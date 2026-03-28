use std::collections::VecDeque;

use anyhow::{Result, bail};

const DEFAULT_FILE_OVERLAP_SECONDS: usize = 6;
const DEFAULT_MAX_DEDUP_WORDS: usize = 24;
const DEFAULT_HOLD_WORDS: usize = 8;

#[derive(Clone, Debug)]
pub struct StreamConfig {
    pub sample_rate: usize,
    pub window_samples: usize,
    pub overlap_samples: usize,
    pub max_dedup_words: usize,
    pub hold_words: usize,
}

impl StreamConfig {
    pub fn for_model(sample_rate: usize, window_samples: usize) -> Result<Self> {
        Self::with_overlap_seconds(sample_rate, window_samples, DEFAULT_FILE_OVERLAP_SECONDS)
    }

    fn with_overlap_seconds(
        sample_rate: usize,
        window_samples: usize,
        overlap_seconds: usize,
    ) -> Result<Self> {
        if sample_rate == 0 {
            bail!("stream sample rate must be greater than zero");
        }
        if window_samples == 0 {
            bail!("stream window size must be greater than zero");
        }

        let overlap_samples = (sample_rate * overlap_seconds).min(window_samples / 2);

        Ok(Self {
            sample_rate,
            window_samples,
            overlap_samples,
            max_dedup_words: DEFAULT_MAX_DEDUP_WORDS,
            hold_words: DEFAULT_HOLD_WORDS,
        })
    }

    pub fn step_samples(&self) -> usize {
        self.window_samples
            .saturating_sub(self.overlap_samples)
            .max(1)
    }

    pub fn overlap_seconds(&self) -> usize {
        self.overlap_samples / self.sample_rate
    }

    pub fn window_count(&self, total_samples: usize) -> usize {
        if total_samples == 0 {
            return 0;
        }
        total_samples.div_ceil(self.step_samples())
    }
}

#[derive(Clone, Debug)]
pub struct StreamChunk {
    pub index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub samples: Vec<f32>,
    pub is_partial: bool,
}

impl StreamChunk {
    pub fn status(&self, total_chunks: usize, overlap_seconds: usize) -> String {
        let partial_suffix = if self.is_partial { " / tail" } else { "" };
        format!(
            "chunk {}/{} / samples {}..{} / {}s overlap{}",
            self.index + 1,
            total_chunks.max(1),
            self.start_sample,
            self.end_sample,
            overlap_seconds,
            partial_suffix
        )
    }
}

#[derive(Debug)]
pub struct StreamEngine {
    config: StreamConfig,
    audio_buffer: VecDeque<f32>,
    next_chunk_index: usize,
    next_start_sample: usize,
}

impl StreamEngine {
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            audio_buffer: VecDeque::new(),
            next_chunk_index: 0,
            next_start_sample: 0,
        }
    }

    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    pub fn push_audio(&mut self, samples: &[f32]) -> Vec<StreamChunk> {
        self.audio_buffer.extend(samples.iter().copied());

        let mut chunks = Vec::new();
        while self.audio_buffer.len() >= self.config.window_samples {
            chunks.push(self.build_chunk(self.config.window_samples, false));
            self.advance_audio();
        }

        chunks
    }

    pub fn finish_audio(&mut self) -> Option<StreamChunk> {
        if self.audio_buffer.is_empty() {
            return None;
        }

        let remaining = self.audio_buffer.len();
        let chunk = self.build_chunk(remaining, true);
        self.audio_buffer.clear();
        self.next_chunk_index += 1;
        self.next_start_sample += self.config.step_samples();
        Some(chunk)
    }

    fn build_chunk(&self, len: usize, is_partial: bool) -> StreamChunk {
        let samples = self
            .audio_buffer
            .iter()
            .take(len)
            .copied()
            .collect::<Vec<_>>();
        StreamChunk {
            index: self.next_chunk_index,
            start_sample: self.next_start_sample,
            end_sample: self.next_start_sample + len,
            samples,
            is_partial,
        }
    }

    fn advance_audio(&mut self) {
        let step = self.config.step_samples();
        let to_drop = step.min(self.audio_buffer.len());
        self.audio_buffer.drain(..to_drop);
        self.next_chunk_index += 1;
        self.next_start_sample += step;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamTextUpdate {
    pub confirmed_text: Option<String>,
    pub preview_text: Option<String>,
}

#[derive(Debug)]
pub struct StreamTranscriptController {
    max_dedup_words: usize,
    carry_hold_words: usize,
    confirmed_tail: String,
}

impl StreamTranscriptController {
    pub fn new(config: &StreamConfig) -> Self {
        Self {
            max_dedup_words: config.max_dedup_words,
            carry_hold_words: config.hold_words.min(4),
            confirmed_tail: String::new(),
        }
    }

    pub fn preview_text(&self, raw_text: &str) -> Option<String> {
        let preview = trim_stream_overlap(&self.confirmed_tail, raw_text, self.max_dedup_words);
        if preview.is_empty() {
            None
        } else {
            Some(preview)
        }
    }

    pub fn resolve_next_window(
        &mut self,
        previous_text: &str,
        current_text: &str,
        boundary_is_silent: bool,
    ) -> StreamTextUpdate {
        let previous_clean =
            trim_stream_overlap(&self.confirmed_tail, previous_text, self.max_dedup_words);
        let current_clean =
            trim_stream_overlap(&self.confirmed_tail, current_text, self.max_dedup_words);

        let previous_words = split_words(&previous_clean);
        let current_words = split_words(&current_clean);
        let overlap_words = find_word_overlap(
            previous_words.as_slice(),
            current_words.as_slice(),
            self.max_dedup_words,
        );
        let carry_words = if overlap_words > 0 || boundary_is_silent {
            0
        } else {
            self.carry_hold_words.min(previous_words.len())
        };
        let unresolved_words = overlap_words.max(carry_words);
        let confirmed_words = if unresolved_words >= previous_words.len() {
            &[][..]
        } else {
            &previous_words[..previous_words.len() - unresolved_words]
        };

        let confirmed_text = if confirmed_words.is_empty() {
            None
        } else {
            let text = confirmed_words.join(" ");
            self.confirm_text(&text)
        };

        let preview_text = if overlap_words > 0 || carry_words == 0 {
            self.preview_text(&current_clean)
        } else {
            let carry_prefix = previous_words[previous_words.len() - carry_words..].join(" ");
            let combined =
                combine_preview_texts(&carry_prefix, &current_clean, self.max_dedup_words);
            let preview =
                trim_stream_overlap(&self.confirmed_tail, &combined, self.max_dedup_words);
            if preview.is_empty() {
                None
            } else {
                Some(preview)
            }
        };

        StreamTextUpdate {
            confirmed_text,
            preview_text,
        }
    }

    pub fn finalize_window(&mut self, text: &str) -> Option<String> {
        let final_text = trim_stream_overlap(&self.confirmed_tail, text, self.max_dedup_words);
        self.confirm_text(&final_text)
    }

    fn confirm_text(&mut self, text: &str) -> Option<String> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }

        self.confirmed_tail = merge_stream_tail(&self.confirmed_tail, text, self.max_dedup_words);
        Some(text.to_string())
    }
}

fn trim_stream_overlap(previous_tail: &str, current: &str, max_dedup_words: usize) -> String {
    let current_words = current.split_whitespace().collect::<Vec<_>>();
    if current_words.is_empty() {
        return String::new();
    }

    let previous_words = previous_tail.split_whitespace().collect::<Vec<_>>();
    let max_overlap = previous_words
        .len()
        .min(current_words.len())
        .min(max_dedup_words);

    for overlap in (1..=max_overlap).rev() {
        let previous_slice = &previous_words[previous_words.len() - overlap..];
        let current_slice = &current_words[..overlap];

        if words_match(previous_slice, current_slice) {
            return current_words[overlap..].join(" ");
        }
    }

    current.trim().to_string()
}

fn merge_stream_tail(previous_tail: &str, current: &str, max_dedup_words: usize) -> String {
    let mut words = previous_tail
        .split_whitespace()
        .chain(current.split_whitespace())
        .collect::<Vec<_>>();

    if words.len() > max_dedup_words {
        words = words.split_off(words.len() - max_dedup_words);
    }

    words.join(" ")
}

fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

fn find_word_overlap(previous: &[String], current: &[String], max_dedup_words: usize) -> usize {
    let max_overlap = previous.len().min(current.len()).min(max_dedup_words);

    for overlap in (1..=max_overlap).rev() {
        let previous_slice = &previous[previous.len() - overlap..];
        let current_slice = &current[..overlap];

        if previous_slice
            .iter()
            .zip(current_slice.iter())
            .all(|(left, right)| normalize_word(left) == normalize_word(right))
        {
            return overlap;
        }
    }

    0
}

fn combine_preview_texts(left: &str, right: &str, max_dedup_words: usize) -> String {
    if left.trim().is_empty() {
        return right.trim().to_string();
    }
    if right.trim().is_empty() {
        return left.trim().to_string();
    }

    let overlap = trim_stream_overlap(left, right, max_dedup_words);
    if overlap.is_empty() {
        format!("{} {}", left.trim(), right.trim())
    } else if overlap.trim() == right.trim() {
        format!("{} {}", left.trim(), overlap.trim())
    } else {
        format!("{} {}", left.trim(), overlap.trim())
    }
}

fn words_match(previous: &[&str], current: &[&str]) -> bool {
    previous.len() == current.len()
        && previous
            .iter()
            .zip(current.iter())
            .all(|(left, right)| normalize_word(left) == normalize_word(right))
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(|character| character.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{StreamConfig, StreamEngine, StreamTranscriptController};

    #[test]
    fn emits_full_and_partial_audio_windows() {
        let config = StreamConfig {
            sample_rate: 16_000,
            window_samples: 6,
            overlap_samples: 2,
            max_dedup_words: 24,
            hold_words: 3,
        };
        let mut engine = StreamEngine::new(config);

        let full = engine.push_audio(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let tail = engine.finish_audio().expect("tail chunk");

        assert_eq!(full.len(), 1);
        assert_eq!(full[0].samples, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(full[0].start_sample, 0);
        assert_eq!(full[0].end_sample, 6);
        assert_eq!(tail.samples, vec![4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(tail.start_sample, 4);
        assert!(tail.is_partial);
    }

    #[test]
    fn computes_window_count_for_incremental_stream() {
        let config = StreamConfig::for_model(16_000, 16_000).expect("config");

        assert_eq!(config.window_count(0), 0);
        assert_eq!(config.window_count(16_000), 2);
        assert_eq!(config.window_count(24_000), 3);
    }

    #[test]
    fn keeps_full_model_window_for_file_stream() {
        let config = StreamConfig::for_model(16_000, 480_000).expect("config");

        assert_eq!(config.window_samples, 480_000);
        assert_eq!(config.overlap_samples, 96_000);
    }

    #[test]
    fn confirms_overlap_at_window_boundary_without_losing_words() {
        let config = StreamConfig {
            sample_rate: 16_000,
            window_samples: 10,
            overlap_samples: 4,
            max_dedup_words: 24,
            hold_words: 8,
        };
        let mut controller = StreamTranscriptController::new(&config);

        let update =
            controller.resolve_next_window("hello brave new world", "new world again there", false);

        assert_eq!(update.confirmed_text, Some("hello brave".to_string()));
        assert_eq!(
            update.preview_text,
            Some("new world again there".to_string())
        );
    }

    #[test]
    fn delays_commit_when_boundary_has_no_text_overlap() {
        let config = StreamConfig {
            sample_rate: 16_000,
            window_samples: 10,
            overlap_samples: 4,
            max_dedup_words: 24,
            hold_words: 8,
        };
        let mut controller = StreamTranscriptController::new(&config);

        let update = controller.resolve_next_window(
            "hello brave new world",
            "something else entirely",
            false,
        );

        assert_eq!(update.confirmed_text, None);
        assert_eq!(
            update.preview_text,
            Some("hello brave new world something else entirely".to_string())
        );
    }

    #[test]
    fn finalizes_pending_window_against_confirmed_tail() {
        let config = StreamConfig {
            sample_rate: 16_000,
            window_samples: 10,
            overlap_samples: 4,
            max_dedup_words: 24,
            hold_words: 8,
        };
        let mut controller = StreamTranscriptController::new(&config);
        let update =
            controller.resolve_next_window("hello brave new world", "new world again there", false);
        assert_eq!(update.confirmed_text, Some("hello brave".to_string()));

        assert_eq!(
            controller.finalize_window("new world again there"),
            Some("new world again there".to_string())
        );
    }
}
