use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzReader;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;
use std::io::Cursor;

use crate::model::ModelChoice;
use crate::onnx_ctc::{ExecutionMode, VocabularyOptions};
use crate::onnx_transducer::OnnxTransducerRuntime;

const ONNX_ASR_FBANKS: &[u8] = include_bytes!("../assets/onnx_asr_fbanks.npz");

pub struct Parakeet {
    runtime: OnnxTransducerRuntime,
    config: ParakeetConfig,
}

impl Parakeet {
    pub fn new(model_dir: &Path, choice: ModelChoice, execution: &ExecutionMode) -> Result<Self> {
        let config_path = model_dir.join(choice.config_file());
        let vocab_path = model_dir.join(choice.vocab_file());
        let encoder_path = model_dir.join(choice.onnx_file(execution.compute_type()));
        let decoder_joint_path = model_dir.join(
            choice
                .secondary_onnx_file(execution.compute_type())
                .context("Parakeet TDT backend requires a decoder_joint ONNX file")?,
        );

        let config = ParakeetConfig::read(&config_path)?;
        let runtime = OnnxTransducerRuntime::new(
            &encoder_path,
            &decoder_joint_path,
            &vocab_path,
            execution,
            VocabularyOptions::default(),
            config.max_tokens_per_step,
        )?;

        Ok(Self { runtime, config })
    }

    pub fn sampling_rate(&self) -> usize {
        self.config.sample_rate
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let mut transcript = String::new();
        self.transcribe_with_callback(samples, |_, _, chunk_text| {
            if !transcript.is_empty() {
                transcript.push('\n');
            }
            transcript.push_str(chunk_text);
            Ok(())
        })?;
        Ok(transcript)
    }

    pub fn transcribe_with_callback<F>(&mut self, samples: &[f32], mut on_chunk: F) -> Result<()>
    where
        F: FnMut(usize, usize, &str) -> Result<()>,
    {
        if samples.is_empty() {
            return Ok(());
        }

        let chunk_ranges = self.chunk_ranges(samples.len());
        let total_chunks = chunk_ranges.len();

        for (chunk_index, range) in chunk_ranges.into_iter().enumerate() {
            let chunk_text = self
                .transcribe_chunk(&samples[range.clone()])
                .with_context(|| format!("failed to transcribe chunk {}", chunk_index + 1))?;
            let chunk_text = chunk_text.trim();
            if chunk_text.is_empty() {
                continue;
            }
            on_chunk(chunk_index + 1, total_chunks, chunk_text)?;
        }

        Ok(())
    }

    fn transcribe_chunk(&mut self, samples: &[f32]) -> Result<String> {
        let (features, lengths) = self.extract_features(samples)?;
        self.runtime
            .transcribe_features(features.view(), lengths.view())
    }

    fn chunk_ranges(&self, total_samples: usize) -> Vec<Range<usize>> {
        let max_samples = self.config.max_chunk_samples();
        if total_samples <= max_samples {
            return vec![0..total_samples];
        }

        let mut ranges = Vec::new();
        let mut start = 0usize;

        while start < total_samples {
            let end = (start + max_samples).min(total_samples);
            ranges.push(start..end);
            if end == total_samples {
                break;
            }
            start = end;
        }

        ranges
    }

    fn extract_features(&self, samples: &[f32]) -> Result<(Array3<f32>, Array1<i64>)> {
        let sample_rate = self.config.sample_rate;
        if sample_rate == 0 {
            bail!("invalid Parakeet config: sample rate is zero");
        }
        let n_mels = self.config.n_mels;
        let win_length = self.config.win_length;
        let hop_length = self.config.hop_length;
        let n_fft = self.config.n_fft;

        if n_mels == 0 || win_length == 0 || hop_length == 0 || n_fft == 0 {
            bail!("invalid Parakeet preprocessing config");
        }

        let emphasized = apply_preemphasis(samples, self.config.preemphasis);
        let padded = zero_pad_waveform(&emphasized, n_fft / 2);
        let frame_count = if padded.len() <= n_fft {
            1
        } else {
            1 + (padded.len() - n_fft) / hop_length
        };

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window = padded_hann_window(win_length, n_fft);
        let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; (n_fft / 2) + 1];
        let mut log_mel = vec![0.0_f32; n_mels * frame_count];

        for frame_index in 0..frame_count {
            let start = frame_index * hop_length;
            let end = (start + n_fft).min(padded.len());

            for value in &mut buffer {
                value.re = 0.0;
                value.im = 0.0;
            }

            for (sample_index, &sample) in padded[start..end].iter().enumerate() {
                buffer[sample_index].re = sample as f64 * window[sample_index];
            }

            fft.process_with_scratch(&mut buffer, &mut scratch);

            for (power_bin, fft_bin) in power.iter_mut().zip(buffer.iter()) {
                *power_bin = fft_bin.norm_sqr();
            }

            for (mel_index, filter_row) in self.config.mel_filters.outer_iter().enumerate() {
                let mut sum = 0.0;
                for (weight, magnitude) in filter_row.iter().zip(power.iter()) {
                    sum += *weight * *magnitude;
                }
                log_mel[(frame_index * n_mels) + mel_index] =
                    (sum as f32 + self.config.log_zero_guard_value).ln();
            }
        }

        let features_len = (samples.len() / hop_length).max(1);
        let normalized = normalize_log_mel_per_feature(&log_mel, frame_count, n_mels, features_len);

        let features = Array3::from_shape_vec((1, n_mels, frame_count), normalized)
            .context("failed to build Parakeet feature tensor")?;
        let lengths = Array1::from_vec(vec![features_len.min(frame_count) as i64]);
        Ok((features, lengths))
    }
}

fn apply_preemphasis(samples: &[f32], preemphasis: f32) -> Vec<f32> {
    let mut emphasized = Vec::with_capacity(samples.len());
    let mut previous = 0.0_f32;

    for &sample in samples {
        emphasized.push(sample - (preemphasis * previous));
        previous = sample;
    }

    emphasized
}

fn hann_window(win_length: usize) -> Vec<f64> {
    let denominator = win_length as f64;
    (0..win_length)
        .map(|index| {
            0.5 * (1.0 - f64::cos((2.0 * std::f64::consts::PI * index as f64) / denominator))
        })
        .collect()
}

fn padded_hann_window(win_length: usize, n_fft: usize) -> Vec<f64> {
    let base = hann_window(win_length);
    let pad = (n_fft.saturating_sub(win_length)) / 2;
    let mut padded = vec![0.0_f64; n_fft];
    for (index, value) in base.into_iter().enumerate() {
        padded[pad + index] = value;
    }
    padded
}

fn zero_pad_waveform(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = vec![0.0_f32; samples.len() + (pad * 2)];
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    padded
}

fn normalize_log_mel_per_feature(
    log_mel: &[f32],
    frame_count: usize,
    n_mels: usize,
    features_len: usize,
) -> Vec<f32> {
    let valid_frames = features_len.min(frame_count).max(1);
    let mut mean = vec![0.0_f32; n_mels];
    let mut var = vec![0.0_f32; n_mels];

    for frame_index in 0..valid_frames {
        let frame = &log_mel[(frame_index * n_mels)..((frame_index + 1) * n_mels)];
        for (mel_index, value) in frame.iter().enumerate() {
            mean[mel_index] += *value;
        }
    }
    for value in &mut mean {
        *value /= valid_frames as f32;
    }

    if valid_frames > 1 {
        for frame_index in 0..valid_frames {
            let frame = &log_mel[(frame_index * n_mels)..((frame_index + 1) * n_mels)];
            for (mel_index, value) in frame.iter().enumerate() {
                let centered = *value - mean[mel_index];
                var[mel_index] += centered * centered;
            }
        }
        for value in &mut var {
            *value /= (valid_frames - 1) as f32;
        }
    }

    let mut normalized = vec![0.0_f32; n_mels * frame_count];
    for frame_index in 0..valid_frames {
        let src = &log_mel[(frame_index * n_mels)..((frame_index + 1) * n_mels)];
        for mel_index in 0..n_mels {
            let std = var[mel_index].sqrt() + 1e-5;
            normalized[(mel_index * frame_count) + frame_index] =
                (src[mel_index] - mean[mel_index]) / std;
        }
    }

    normalized
}

#[derive(Debug)]
struct ParakeetConfig {
    sample_rate: usize,
    n_mels: usize,
    win_length: usize,
    hop_length: usize,
    n_fft: usize,
    preemphasis: f32,
    log_zero_guard_value: f32,
    max_chunk_seconds: usize,
    max_tokens_per_step: usize,
    mel_filters: Array2<f64>,
}

impl ParakeetConfig {
    fn read(path: &Path) -> Result<Self> {
        let config = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read `{}`", path.display()))?;
        let raw: RawParakeetConfig = serde_json::from_str(&config)
            .with_context(|| format!("failed to parse `{}`", path.display()))?;

        let sample_rate = 16_000usize;
        let n_mels = raw.features_size.unwrap_or(80);
        let n_fft = 512usize;
        let win_length = 400usize;
        let hop_length = 160usize;
        let preemphasis = 0.97_f32;
        let mel_filters = load_nemo_filterbank(n_mels)?;

        Ok(Self {
            sample_rate,
            n_mels,
            win_length,
            hop_length,
            n_fft,
            preemphasis,
            log_zero_guard_value: 2f32.powi(-24),
            max_chunk_seconds: 30,
            max_tokens_per_step: raw.max_tokens_per_step.unwrap_or(10),
            mel_filters,
        })
    }

    fn max_chunk_samples(&self) -> usize {
        self.sample_rate * self.max_chunk_seconds
    }
}

fn load_nemo_filterbank(n_mels: usize) -> Result<Array2<f64>> {
    let reader = Cursor::new(ONNX_ASR_FBANKS);
    let mut archive =
        NpzReader::new(reader).context("failed to open bundled onnx-asr filter bank archive")?;
    let filterbank_name = match n_mels {
        80 => "nemo80.npy",
        128 => "nemo128.npy",
        other => bail!("unsupported Parakeet feature size {other}; expected 80 or 128"),
    };
    let fbanks: Array2<f32> = archive.by_name(filterbank_name).with_context(|| {
        format!("failed to load `{filterbank_name}` filter bank from bundled onnx-asr archive")
    })?;
    Ok(fbanks.reversed_axes().mapv(f64::from))
}

#[derive(Debug, Deserialize)]
struct RawParakeetConfig {
    features_size: Option<usize>,
    max_tokens_per_step: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::apply_preemphasis;

    #[test]
    fn applies_preemphasis_progressively() {
        let emphasized = apply_preemphasis(&[1.0, 0.5, 0.25], 0.5);
        assert_eq!(emphasized, vec![1.0, 0.0, 0.0]);
    }
}
