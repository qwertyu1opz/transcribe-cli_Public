use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mel_spec::mel::mel;
use ndarray::{Array1, Array2, Array3};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;

use crate::model::ModelChoice;
use crate::onnx_ctc::{ExecutionMode, OnnxCtcRuntime, VocabularyOptions};

const MIN_LOG_VALUE: f64 = 1e-9;
const MAX_LOG_VALUE: f64 = 1e9;

pub struct GigaAm {
    runtime: OnnxCtcRuntime,
    config: GigaAmConfig,
}

impl GigaAm {
    pub fn new(model_dir: &Path, choice: ModelChoice, execution: &ExecutionMode) -> Result<Self> {
        let yaml_path = model_dir.join(choice.config_file());
        let vocab_path = model_dir.join(choice.vocab_file());
        let onnx_path = model_dir.join(choice.onnx_file(execution.compute_type()));

        let config = GigaAmConfig::read(&yaml_path)?;
        let runtime = OnnxCtcRuntime::new(
            &onnx_path,
            &vocab_path,
            execution,
            VocabularyOptions::default(),
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
            bail!("invalid GigaAM config: sample rate is zero");
        }
        let n_mels = self.config.n_mels;
        let win_length = self.config.win_length;
        let hop_length = self.config.hop_length;
        let n_fft = self.config.n_fft;

        if n_mels == 0 || win_length == 0 || hop_length == 0 || n_fft == 0 {
            bail!("invalid GigaAM preprocessing config");
        }
        if self.config.center {
            bail!("unsupported GigaAM config: center=true is not implemented");
        }

        let frame_count = if samples.len() <= win_length {
            1
        } else {
            1 + (samples.len() - win_length) / hop_length
        };

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window = hann_window(win_length);
        let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; (n_fft / 2) + 1];
        let mut features = vec![0.0_f32; n_mels * frame_count];

        for frame_index in 0..frame_count {
            let start = frame_index * hop_length;
            let end = (start + win_length).min(samples.len());

            for value in &mut buffer {
                value.re = 0.0;
                value.im = 0.0;
            }

            for (sample_index, &sample) in samples[start..end].iter().enumerate() {
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
                features[(mel_index * frame_count) + frame_index] =
                    sum.clamp(MIN_LOG_VALUE, MAX_LOG_VALUE).ln() as f32;
            }
        }

        let features = Array3::from_shape_vec((1, n_mels, frame_count), features)
            .context("failed to build GigaAM feature tensor")?;
        let lengths = Array1::from_vec(vec![frame_count as i64]);
        Ok((features, lengths))
    }
}

fn hann_window(win_length: usize) -> Vec<f64> {
    let denominator = win_length as f64;
    (0..win_length)
        .map(|index| {
            0.5 * (1.0 - f64::cos((2.0 * std::f64::consts::PI * index as f64) / denominator))
        })
        .collect()
}

#[derive(Debug)]
struct GigaAmConfig {
    sample_rate: usize,
    n_mels: usize,
    win_length: usize,
    hop_length: usize,
    n_fft: usize,
    center: bool,
    max_input_frames: usize,
    mel_filters: Array2<f64>,
}

impl GigaAmConfig {
    fn read(path: &Path) -> Result<Self> {
        let yaml = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read `{}`", path.display()))?;
        let raw: RawGigaAmConfig = serde_yaml::from_str(&yaml)
            .with_context(|| format!("failed to parse `{}`", path.display()))?;

        let preprocessor = raw
            .preprocessor
            .context("GigaAM config does not define a preprocessor section")?;
        let sample_rate = preprocessor
            .sample_rate
            .unwrap_or(raw.sample_rate.unwrap_or(16_000));
        let encoder = raw
            .encoder
            .context("GigaAM config does not define an encoder section")?;
        let n_fft = preprocessor
            .n_fft
            .unwrap_or(preprocessor.win_length.unwrap_or(320));
        let n_mels = preprocessor.features.unwrap_or(64);
        let mel_filters = mel(sample_rate as f64, n_fft, n_mels, None, None, true, false);

        Ok(Self {
            sample_rate: sample_rate as usize,
            n_mels,
            win_length: preprocessor.win_length.unwrap_or(n_fft),
            hop_length: preprocessor
                .hop_length
                .unwrap_or(sample_rate as usize / 100),
            n_fft,
            center: preprocessor.center.unwrap_or(false),
            max_input_frames: encoder.pos_emb_max_len.unwrap_or(5_000),
            mel_filters,
        })
    }

    fn safe_chunk_frames(&self) -> usize {
        self.max_input_frames.saturating_sub(64).max(512)
    }

    fn max_chunk_samples(&self) -> usize {
        self.win_length
            + self
                .hop_length
                .saturating_mul(self.safe_chunk_frames().saturating_sub(1))
    }
}

#[derive(Debug, Deserialize)]
struct RawGigaAmConfig {
    sample_rate: Option<usize>,
    preprocessor: Option<RawPreprocessorConfig>,
    encoder: Option<RawEncoderConfig>,
}

#[derive(Debug, Deserialize)]
struct RawPreprocessorConfig {
    sample_rate: Option<usize>,
    features: Option<usize>,
    win_length: Option<usize>,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
    center: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawEncoderConfig {
    pos_emb_max_len: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::GigaAmConfig;
    use ndarray::Array2;

    #[test]
    fn safe_chunk_frames_reserve_positional_headroom() {
        let config = GigaAmConfig {
            sample_rate: 16_000,
            n_mels: 64,
            win_length: 320,
            hop_length: 160,
            n_fft: 512,
            center: false,
            max_input_frames: 5_000,
            mel_filters: Array2::zeros((64, 257)),
        };

        assert_eq!(config.safe_chunk_frames(), 4_936);
        assert_eq!(config.max_chunk_samples(), 789_920);
    }
}
