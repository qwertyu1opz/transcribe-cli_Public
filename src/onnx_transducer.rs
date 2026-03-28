use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView3, Axis, Ix1, Ix3};
use ort::value::TensorRef;

use crate::onnx_ctc::{CtcVocabulary, ExecutionMode, VocabularyOptions, build_session};

pub struct OnnxTransducerRuntime {
    encoder: ort::session::Session,
    decoder_joint: ort::session::Session,
    vocabulary: CtcVocabulary,
    state_spec: DecoderStateSpec,
    max_tokens_per_step: usize,
}

impl OnnxTransducerRuntime {
    pub fn new(
        encoder_path: &Path,
        decoder_joint_path: &Path,
        vocab_path: &Path,
        execution: &ExecutionMode,
        vocabulary_options: VocabularyOptions,
        max_tokens_per_step: usize,
    ) -> Result<Self> {
        let vocabulary = CtcVocabulary::from_text_file(vocab_path, vocabulary_options)?;
        let encoder = build_session(encoder_path, execution)?;
        let decoder_joint = build_session(decoder_joint_path, execution)?;
        let state_spec = DecoderStateSpec::from_session(&decoder_joint)?;

        Ok(Self {
            encoder,
            decoder_joint,
            vocabulary,
            state_spec,
            max_tokens_per_step: max_tokens_per_step.max(1),
        })
    }

    pub fn transcribe_features(
        &mut self,
        features: ArrayView3<'_, f32>,
        lengths: ArrayView1<'_, i64>,
    ) -> Result<String> {
        let (encoder_out, encoder_lengths) = self.encode(features, lengths)?;
        let encodings = encoder_out.index_axis(Axis(0), 0);
        let encoded_len = encoder_lengths.get(0).copied().unwrap_or_default().max(0) as usize;

        self.decode_tdt(encodings, encoded_len)
    }

    fn encode(
        &mut self,
        features: ArrayView3<'_, f32>,
        lengths: ArrayView1<'_, i64>,
    ) -> Result<(Array3<f32>, Array1<i64>)> {
        let outputs = self.encoder.run(ort::inputs![
            TensorRef::from_array_view(features)?,
            TensorRef::from_array_view(lengths)?,
        ])?;
        let encoder_out = outputs[0]
            .try_extract_array::<f32>()
            .context("failed to extract transducer encoder output tensor")?;
        let encoder_out = encoder_out
            .into_dimensionality::<Ix3>()
            .map_err(|error| anyhow!("unexpected transducer encoder output shape: {error}"))?
            .permuted_axes([0, 2, 1])
            .to_owned();

        let encoder_lengths = outputs[1]
            .try_extract_array::<i64>()
            .context("failed to extract transducer encoder lengths tensor")?;
        let encoder_lengths = encoder_lengths
            .into_dimensionality::<Ix1>()
            .map_err(|error| anyhow!("unexpected transducer length output shape: {error}"))?
            .to_owned();

        Ok((encoder_out, encoder_lengths))
    }

    fn decode_tdt(
        &mut self,
        encodings: ndarray::ArrayView2<'_, f32>,
        encoded_len: usize,
    ) -> Result<String> {
        let valid_len = encoded_len.min(encodings.len_of(Axis(0)));
        if valid_len == 0 {
            return Ok(String::new());
        }

        let blank_id = self.vocabulary.blank_id();
        let vocab_size = self.vocabulary.len();
        let mut state = self.state_spec.initial_state();
        let mut tokens = Vec::new();
        let mut t = 0usize;
        let mut emitted_tokens = 0usize;

        while t < valid_len {
            let frame = encodings.index_axis(Axis(0), t);
            let previous_token = tokens.last().copied().unwrap_or(blank_id) as i32;
            let (output, next_state) = self.decode_step(frame, previous_token, &state)?;

            if output.len() <= vocab_size {
                bail!(
                    "unexpected Parakeet TDT output width {}; expected more than vocabulary size {}",
                    output.len(),
                    vocab_size
                );
            }

            let (token_logits, duration_logits) = output.split_at(vocab_size);
            let token = argmax(token_logits).context("Parakeet TDT emitted empty token logits")?;
            let step =
                argmax(duration_logits).context("Parakeet TDT emitted empty duration logits")?;

            if token != blank_id {
                state = next_state;
                tokens.push(token);
                emitted_tokens += 1;
            }

            if step > 0 {
                t += step;
                emitted_tokens = 0;
            } else if token == blank_id || emitted_tokens == self.max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok(self.vocabulary.decode_ids(&tokens))
    }

    fn decode_step(
        &mut self,
        encoder_frame: ndarray::ArrayView1<'_, f32>,
        previous_token: i32,
        state: &DecoderState,
    ) -> Result<(Vec<f32>, DecoderState)> {
        let frame_width = encoder_frame.len();
        let mut encoder_outputs = Array3::<f32>::zeros((1, frame_width, 1));
        for (index, value) in encoder_frame.iter().copied().enumerate() {
            encoder_outputs[(0, index, 0)] = value;
        }
        let targets = Array2::<i32>::from_shape_vec((1, 1), vec![previous_token])
            .context("failed to build transducer target tensor")?;
        let target_length = Array1::<i32>::from_vec(vec![1]);

        let outputs = self.decoder_joint.run(ort::inputs![
            TensorRef::from_array_view(encoder_outputs.view())?,
            TensorRef::from_array_view(targets.view())?,
            TensorRef::from_array_view(target_length.view())?,
            TensorRef::from_array_view(state.state1.view())?,
            TensorRef::from_array_view(state.state2.view())?,
        ])?;

        let logits = outputs[0]
            .try_extract_array::<f32>()
            .context("failed to extract transducer decoder output tensor")?;
        let logits = logits.iter().copied().collect::<Vec<_>>();

        let state1 = outputs[2]
            .try_extract_array::<f32>()
            .context("failed to extract transducer recurrent state 1")?;
        let state1 = state1
            .into_dimensionality::<Ix3>()
            .map_err(|error| anyhow!("unexpected transducer state_1 shape: {error}"))?
            .to_owned();

        let state2 = outputs[3]
            .try_extract_array::<f32>()
            .context("failed to extract transducer recurrent state 2")?;
        let state2 = state2
            .into_dimensionality::<Ix3>()
            .map_err(|error| anyhow!("unexpected transducer state_2 shape: {error}"))?
            .to_owned();

        Ok((logits, DecoderState { state1, state2 }))
    }
}

#[derive(Clone, Debug)]
struct DecoderStateSpec {
    state1_layers: usize,
    state1_hidden: usize,
    state2_layers: usize,
    state2_hidden: usize,
}

impl DecoderStateSpec {
    fn from_session(session: &ort::session::Session) -> Result<Self> {
        let (state1_layers, state1_hidden) = infer_state_shape(session, "input_states_1")?;
        let (state2_layers, state2_hidden) = infer_state_shape(session, "input_states_2")?;
        Ok(Self {
            state1_layers,
            state1_hidden,
            state2_layers,
            state2_hidden,
        })
    }

    fn initial_state(&self) -> DecoderState {
        DecoderState {
            state1: Array3::<f32>::zeros((self.state1_layers, 1, self.state1_hidden)),
            state2: Array3::<f32>::zeros((self.state2_layers, 1, self.state2_hidden)),
        }
    }
}

#[derive(Clone, Debug)]
struct DecoderState {
    state1: Array3<f32>,
    state2: Array3<f32>,
}

fn infer_state_shape(session: &ort::session::Session, input_name: &str) -> Result<(usize, usize)> {
    let input = session
        .inputs
        .iter()
        .find(|input| input.name == input_name)
        .with_context(|| format!("transducer model does not define `{input_name}`"))?;
    let shape = input
        .input_type
        .tensor_shape()
        .with_context(|| format!("transducer input `{input_name}` is not a tensor"))?;

    if shape.len() != 3 {
        bail!(
            "transducer input `{input_name}` has unsupported rank {}; expected 3",
            shape.len()
        );
    }

    let layers = usize::try_from(shape[0])
        .ok()
        .filter(|value| *value > 0)
        .with_context(|| {
            format!(
                "transducer input `{input_name}` has unsupported layer dimension {}",
                shape[0]
            )
        })?;
    let hidden = usize::try_from(shape[2])
        .ok()
        .filter(|value| *value > 0)
        .with_context(|| {
            format!(
                "transducer input `{input_name}` has unsupported hidden dimension {}",
                shape[2]
            )
        })?;
    Ok((layers, hidden))
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
}
