use std::collections::VecDeque;

use anyhow::{Result, bail};

const DEFAULT_FRAME_MS: usize = 30;
const DEFAULT_PRE_ROLL_MS: usize = 240;
const DEFAULT_MIN_SPEECH_MS: usize = 180;
const DEFAULT_MAX_SILENCE_MS: usize = 420;
const DEFAULT_MIN_SEGMENT_MS: usize = 240;
const DEFAULT_CONTINUATION_MS: usize = 400;

#[derive(Clone, Debug)]
pub struct DynamicChunkConfig {
    pub frame_samples: usize,
    pub pre_roll_samples: usize,
    pub min_speech_frames: usize,
    pub max_silence_frames: usize,
    pub min_segment_samples: usize,
    pub max_segment_samples: usize,
    pub continuation_samples: usize,
    pub base_start_rms: f32,
    pub base_continue_rms: f32,
    pub max_zero_crossing_rate: f32,
    pub max_continue_zero_crossing_rate: f32,
    pub max_silence_rms: f32,
}

impl DynamicChunkConfig {
    pub fn for_live_stream(sample_rate: usize, model_window_samples: usize) -> Result<Self> {
        let mut config = Self::new(sample_rate, model_window_samples, 6, 0.010, 0.006)?;
        config.max_silence_frames = (720 / DEFAULT_FRAME_MS).max(1);
        config.continuation_samples =
            (sample_rate * 900 / 1000).min(config.max_segment_samples / 2);
        Ok(config)
    }

    fn new(
        sample_rate: usize,
        model_window_samples: usize,
        max_segment_seconds: usize,
        base_start_rms: f32,
        base_continue_rms: f32,
    ) -> Result<Self> {
        if sample_rate == 0 {
            bail!("dynamic chunk sample rate must be greater than zero");
        }
        if model_window_samples == 0 {
            bail!("dynamic chunk model window size must be greater than zero");
        }

        let frame_samples = (sample_rate * DEFAULT_FRAME_MS / 1000).max(1);
        let pre_roll_samples = sample_rate * DEFAULT_PRE_ROLL_MS / 1000;
        let min_speech_frames = (DEFAULT_MIN_SPEECH_MS / DEFAULT_FRAME_MS).max(1);
        let max_silence_frames = (DEFAULT_MAX_SILENCE_MS / DEFAULT_FRAME_MS).max(1);
        let min_segment_samples = sample_rate * DEFAULT_MIN_SEGMENT_MS / 1000;
        let max_segment_samples = (sample_rate * max_segment_seconds).min(model_window_samples);
        let continuation_samples =
            (sample_rate * DEFAULT_CONTINUATION_MS / 1000).min(max_segment_samples / 3);

        Ok(Self {
            frame_samples,
            pre_roll_samples,
            min_speech_frames,
            max_silence_frames,
            min_segment_samples,
            max_segment_samples,
            continuation_samples,
            base_start_rms,
            base_continue_rms,
            max_zero_crossing_rate: 0.22,
            max_continue_zero_crossing_rate: 0.32,
            max_silence_rms: 0.0025,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DynamicChunk {
    pub samples: Vec<f32>,
}

#[derive(Debug)]
pub struct DynamicChunkEngine {
    config: DynamicChunkConfig,
    pending_samples: VecDeque<f32>,
    pre_roll_samples: VecDeque<f32>,
    active_segment: Vec<f32>,
    conditioner: RealtimeAudioConditioner,
    in_speech: bool,
    speech_run_frames: usize,
    silence_run_frames: usize,
    processed_samples: usize,
    segment_start_sample: usize,
    noise_floor_rms: f32,
}

impl DynamicChunkEngine {
    pub fn new(config: DynamicChunkConfig) -> Self {
        let noise_floor_rms = config.base_continue_rms * 0.5;
        Self {
            config,
            pending_samples: VecDeque::new(),
            pre_roll_samples: VecDeque::new(),
            active_segment: Vec::new(),
            conditioner: RealtimeAudioConditioner::new(noise_floor_rms),
            in_speech: false,
            speech_run_frames: 0,
            silence_run_frames: 0,
            processed_samples: 0,
            segment_start_sample: 0,
            noise_floor_rms,
        }
    }

    pub fn push_audio(&mut self, samples: &[f32]) -> Vec<DynamicChunk> {
        self.pending_samples.extend(samples.iter().copied());
        let mut chunks = Vec::new();

        while self.pending_samples.len() >= self.config.frame_samples {
            let raw_frame = self
                .pending_samples
                .drain(..self.config.frame_samples)
                .collect::<Vec<_>>();
            let conditioned_frame = self.conditioner.process_frame(&raw_frame);
            self.process_frame(raw_frame, conditioned_frame, false, &mut chunks);
        }

        chunks
    }

    pub fn finish_audio(&mut self) -> Vec<DynamicChunk> {
        let mut chunks = Vec::new();

        if !self.pending_samples.is_empty() {
            let raw_frame = self.pending_samples.drain(..).collect::<Vec<_>>();
            let conditioned_frame = self.conditioner.process_frame(&raw_frame);
            self.process_frame(raw_frame, conditioned_frame, true, &mut chunks);
        }

        if self.in_speech {
            if let Some(chunk) = self.emit_segment() {
                chunks.push(chunk);
            }
        }

        chunks
    }

    fn process_frame(
        &mut self,
        raw_frame: Vec<f32>,
        conditioned_frame: Vec<f32>,
        finalize: bool,
        chunks: &mut Vec<DynamicChunk>,
    ) {
        if raw_frame.is_empty() {
            return;
        }

        let metrics = FrameMetrics::analyze(&conditioned_frame);
        let is_speech = self.classify_frame(&metrics, self.in_speech);
        self.processed_samples += raw_frame.len();

        if self.in_speech {
            self.active_segment.extend_from_slice(&raw_frame);

            if is_speech {
                self.silence_run_frames = 0;
            } else {
                self.silence_run_frames += 1;
            }

            if self.active_segment.len() >= self.config.max_segment_samples {
                if let Some(chunk) = self.emit_segment() {
                    chunks.push(chunk);
                }
                self.keep_continuation();
                self.silence_run_frames = if is_speech { 0 } else { 1 };
                self.in_speech = true;
            } else if self.silence_run_frames >= self.config.max_silence_frames {
                if let Some(chunk) = self.emit_segment() {
                    chunks.push(chunk);
                }
                self.reset_detection_state();
                self.update_noise_floor(&metrics);
            }

            if finalize && self.in_speech {
                if let Some(chunk) = self.emit_segment() {
                    chunks.push(chunk);
                }
                self.reset_detection_state();
            }

            return;
        }

        self.push_pre_roll(&raw_frame);

        if is_speech {
            self.speech_run_frames += 1;
            if self.speech_run_frames >= self.config.min_speech_frames {
                self.in_speech = true;
                self.silence_run_frames = 0;
                self.segment_start_sample = self.processed_samples - self.pre_roll_samples.len();
                self.active_segment = self.pre_roll_samples.iter().copied().collect();
                self.pre_roll_samples.clear();
            }
        } else {
            self.speech_run_frames = 0;
            self.update_noise_floor(&metrics);
        }

        if finalize {
            self.pre_roll_samples.clear();
        }
    }

    fn classify_frame(&self, metrics: &FrameMetrics, continuing: bool) -> bool {
        let start_rms = (self.noise_floor_rms * 2.7 + 0.001).max(self.config.base_start_rms);
        let continue_rms = (self.noise_floor_rms * 1.6 + 0.0007).max(self.config.base_continue_rms);
        let start_zcr_limit = self.config.max_zero_crossing_rate + metrics.activity_ratio * 0.06;
        let continue_zcr_limit =
            self.config.max_continue_zero_crossing_rate + metrics.activity_ratio * 0.05;

        if continuing {
            metrics.rms >= continue_rms
                && metrics.peak >= continue_rms * 1.35
                && metrics.activity_ratio >= 0.12
                && (metrics.zero_crossing_rate <= continue_zcr_limit
                    || (metrics.activity_ratio >= 0.20 && metrics.peak_to_rms >= 1.55))
        } else {
            metrics.rms >= start_rms
                && metrics.peak >= start_rms * 1.7
                && metrics.activity_ratio >= 0.18
                && (metrics.zero_crossing_rate <= start_zcr_limit
                    || (metrics.activity_ratio >= 0.30 && metrics.peak_to_rms >= 1.75))
        }
    }

    fn update_noise_floor(&mut self, metrics: &FrameMetrics) {
        if metrics.rms > self.config.max_silence_rms {
            return;
        }

        self.noise_floor_rms = if self.noise_floor_rms == 0.0 {
            metrics.rms
        } else {
            self.noise_floor_rms * 0.95 + metrics.rms * 0.05
        };
    }

    fn push_pre_roll(&mut self, frame: &[f32]) {
        self.pre_roll_samples.extend(frame.iter().copied());
        if self.pre_roll_samples.len() > self.config.pre_roll_samples {
            let overflow = self.pre_roll_samples.len() - self.config.pre_roll_samples;
            self.pre_roll_samples.drain(..overflow);
        }
    }

    fn emit_segment(&mut self) -> Option<DynamicChunk> {
        if self.active_segment.len() < self.config.min_segment_samples {
            self.active_segment.clear();
            return None;
        }

        let samples = std::mem::take(&mut self.active_segment);
        Some(DynamicChunk { samples })
    }

    fn keep_continuation(&mut self) {
        let tail_len = self
            .config
            .continuation_samples
            .min(self.active_segment.len());
        let start = self.active_segment.len().saturating_sub(tail_len);
        self.segment_start_sample += start;
        self.active_segment = self.active_segment[start..].to_vec();
    }

    fn reset_detection_state(&mut self) {
        self.in_speech = false;
        self.speech_run_frames = 0;
        self.silence_run_frames = 0;
        self.active_segment.clear();
        self.pre_roll_samples.clear();
    }
}

#[derive(Debug)]
struct FrameMetrics {
    rms: f32,
    peak: f32,
    activity_ratio: f32,
    peak_to_rms: f32,
    zero_crossing_rate: f32,
}

impl FrameMetrics {
    fn analyze(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self {
                rms: 0.0,
                peak: 0.0,
                activity_ratio: 0.0,
                peak_to_rms: 0.0,
                zero_crossing_rate: 0.0,
            };
        }

        let peak = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0f32, f32::max);
        let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        let dead_zone = (peak * 0.10).max(rms * 0.25).max(0.0015);
        let activity_samples = samples
            .iter()
            .filter(|sample| sample.abs() >= dead_zone)
            .count();
        let activity_ratio = activity_samples as f32 / samples.len() as f32;

        let mut zero_crossings = 0usize;
        let mut previous_sign = 0i8;
        for &sample in samples {
            let sign = if sample >= dead_zone {
                1
            } else if sample <= -dead_zone {
                -1
            } else {
                0
            };

            if sign == 0 {
                continue;
            }

            if previous_sign != 0 && previous_sign != sign {
                zero_crossings += 1;
            }
            previous_sign = sign;
        }

        let zero_crossing_rate = zero_crossings as f32 / samples.len() as f32;
        let peak_to_rms = if rms > 0.0 { peak / rms } else { 0.0 };

        Self {
            rms,
            peak,
            activity_ratio,
            peak_to_rms,
            zero_crossing_rate,
        }
    }
}

#[derive(Debug)]
struct RealtimeAudioConditioner {
    previous_input: f32,
    previous_output: f32,
    noise_floor: f32,
}

impl RealtimeAudioConditioner {
    fn new(initial_noise_floor: f32) -> Self {
        Self {
            previous_input: 0.0,
            previous_output: 0.0,
            noise_floor: initial_noise_floor.max(0.0005),
        }
    }

    fn process_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        if frame.is_empty() {
            return Vec::new();
        }

        // Cheap DC/hum reduction plus adaptive soft gate.
        let mut filtered = Vec::with_capacity(frame.len());
        let mut rms_accumulator = 0.0f32;
        let mut peak = 0.0f32;

        for &sample in frame {
            let filtered_sample = sample - self.previous_input + 0.97 * self.previous_output;
            self.previous_input = sample;
            self.previous_output = filtered_sample;
            peak = peak.max(filtered_sample.abs());
            rms_accumulator += filtered_sample * filtered_sample;
            filtered.push(filtered_sample);
        }

        let frame_rms = (rms_accumulator / filtered.len() as f32).sqrt();
        let likely_noise =
            frame_rms <= self.noise_floor * 2.4 + 0.0015 && peak <= self.noise_floor * 6.0 + 0.012;

        if likely_noise {
            self.noise_floor = self.noise_floor * 0.96 + frame_rms * 0.04;
        } else {
            self.noise_floor = self.noise_floor * 0.995;
        }

        let floor_gain = if likely_noise { 0.16 } else { 0.34 };
        let gate_threshold = if likely_noise {
            self.noise_floor * 2.2 + 0.001
        } else {
            self.noise_floor * 1.6 + 0.0007
        };
        let knee = gate_threshold * 1.1 + 0.0008;
        for sample in &mut filtered {
            let magnitude = sample.abs();
            let gain = if magnitude <= gate_threshold {
                floor_gain
            } else if magnitude < gate_threshold + knee {
                floor_gain + (1.0 - floor_gain) * ((magnitude - gate_threshold) / knee)
            } else {
                1.0
            };
            *sample *= gain;
        }

        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::{DynamicChunkConfig, DynamicChunkEngine, RealtimeAudioConditioner};

    fn speech_like_samples(len: usize, amplitude: f32) -> Vec<f32> {
        let pattern = [0.0, 0.35, 0.8, 0.45, 0.05, -0.3, -0.75, -0.4];
        (0..len)
            .map(|index| amplitude * pattern[index % pattern.len()])
            .collect()
    }

    #[test]
    fn skips_silence_and_emits_speech_segments() {
        let config = DynamicChunkConfig::for_live_stream(1_000, 10_000).expect("config");
        let mut engine = DynamicChunkEngine::new(config);
        let silence = vec![0.0; 500];
        let speech = speech_like_samples(800, 0.12);
        let tail = vec![0.0; 600];

        let mut chunks = engine.push_audio(&silence);
        chunks.extend(engine.push_audio(&speech));
        chunks.extend(engine.push_audio(&tail));
        chunks.extend(engine.finish_audio());

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].samples.len() >= 800);
    }

    #[test]
    fn splits_long_speech_with_partial_chunk() {
        let mut config = DynamicChunkConfig::for_live_stream(1_000, 10_000).expect("config");
        config.max_segment_samples = 1_200;
        config.continuation_samples = 200;
        let mut engine = DynamicChunkEngine::new(config);
        let speech = speech_like_samples(3_200, 0.14);

        let mut chunks = engine.push_audio(&speech);
        chunks.extend(engine.finish_audio());

        assert!(chunks.len() >= 2);
    }

    #[test]
    fn conditioner_attenuates_quiet_background_noise() {
        let mut conditioner = RealtimeAudioConditioner::new(0.001);
        let input = vec![0.002; 64];

        let output = conditioner.process_frame(&input);

        let input_energy = input.iter().map(|sample| sample * sample).sum::<f32>();
        let output_energy = output.iter().map(|sample| sample * sample).sum::<f32>();
        assert!(output_energy < input_energy * 0.5);
    }
}
