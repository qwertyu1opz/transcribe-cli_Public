#[cfg(target_os = "linux")]
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
use std::sync::OnceLock;
use std::thread;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{ArrayBase, ArrayView1, ArrayView3, Axis, Ix3, ViewRepr, s};
use ort::{
    execution_providers::cuda::CUDAExecutionProvider,
    init_from,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

use crate::model::ModelComputeType;
#[cfg(target_os = "linux")]
use crate::model::default_ort_runtime_root_directory;
#[cfg(target_os = "linux")]
static ORT_RUNTIME_PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();
static ORT_RUNTIME_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct ExecutionMode {
    use_gpu: bool,
    compute_type: ModelComputeType,
    gpu_device: Option<i32>,
}

impl ExecutionMode {
    pub fn from_cli(
        use_gpu: bool,
        gpu_device: i32,
        requested_compute_type: Option<ModelComputeType>,
    ) -> Result<Self> {
        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                if gpu_device < 0 {
                    bail!("CUDA device index must be non-negative");
                }

                return Ok(Self {
                    use_gpu: true,
                    compute_type: requested_compute_type.unwrap_or(ModelComputeType::Float32),
                    gpu_device: Some(gpu_device),
                });
            }

            #[cfg(not(feature = "gpu"))]
            {
                let _ = gpu_device;
                let _ = requested_compute_type;
                bail!(
                    "--gpu requires a build with ONNX Runtime CUDA support, for example `cargo install --features cuda`"
                );
            }
        }

        Ok(Self {
            use_gpu: false,
            compute_type: requested_compute_type.unwrap_or(ModelComputeType::Int8),
            gpu_device: None,
        })
    }

    pub fn compute_type(&self) -> ModelComputeType {
        self.compute_type
    }

    pub fn gpu_device(&self) -> Option<i32> {
        self.gpu_device
    }

    pub fn device_label(&self) -> &'static str {
        if self.use_gpu { "cuda" } else { "cpu" }
    }

    pub fn compute_type_label(&self) -> &'static str {
        self.compute_type.label()
    }
}

#[derive(Clone, Debug)]
pub struct VocabularyOptions {
    pub blank_token: &'static str,
    pub word_boundary_token: Option<&'static str>,
}

impl Default for VocabularyOptions {
    fn default() -> Self {
        Self {
            blank_token: "<blk>",
            word_boundary_token: Some("▁"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CtcVocabulary {
    entries: Vec<String>,
    blank_id: usize,
}

impl CtcVocabulary {
    pub fn from_text_file(path: &Path, options: VocabularyOptions) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open CTC vocabulary `{}`", path.display()))?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        let boundary = options.word_boundary_token.map(ToOwned::to_owned);

        for line in reader.lines() {
            let line = line.with_context(|| {
                format!("failed to read vocabulary line from `{}`", path.display())
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let mut parts = trimmed.rsplitn(2, ' ');
            let index = parts
                .next()
                .context("vocabulary entry does not contain an index")?
                .parse::<usize>()
                .with_context(|| format!("failed to parse vocabulary index from `{trimmed}`"))?;
            let token = parts
                .next()
                .context("vocabulary entry does not contain a token")?
                .to_string();
            let token = if let Some(boundary) = boundary.as_deref() {
                token.replace(boundary, " ")
            } else {
                token
            };

            if entries.len() <= index {
                entries.resize(index + 1, String::new());
            }
            entries[index] = token;
        }

        let blank_id = entries
            .iter()
            .position(|token| token == options.blank_token)
            .with_context(|| {
                format!(
                    "CTC vocabulary does not contain the blank token `{}`",
                    options.blank_token
                )
            })?;

        Ok(Self { entries, blank_id })
    }

    pub fn decode_ids(&self, token_ids: &[usize]) -> String {
        let mut text = String::new();
        for &token_id in token_ids {
            if token_id == self.blank_id {
                continue;
            }

            if let Some(token) = self.entries.get(token_id) {
                let trimmed = token.trim_start_matches(' ');
                if trimmed.len() != token.len() && !text.is_empty() && !text.ends_with(' ') {
                    text.push(' ');
                }
                text.push_str(trimmed);
            }
        }
        text.trim_start().to_string()
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn decode_logits_with_lengths(
        &self,
        logits: &ArrayBase<ViewRepr<&f32>, Ix3>,
        output_lengths: Option<ArrayView1<'_, i64>>,
    ) -> String {
        if logits.is_empty() {
            return String::new();
        }

        let sequence_len = logits.len_of(Axis(1));
        let valid_frames = output_lengths
            .and_then(|lengths| lengths.get(0).copied())
            .map(|length| length.max(0) as usize)
            .unwrap_or(sequence_len)
            .min(sequence_len);
        let logits = logits.slice(s![0, 0..valid_frames, ..]);
        let mut token_ids = Vec::new();
        let mut previous = self.blank_id;

        for frame in logits.outer_iter() {
            let Some((token_id, _)) = frame
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
            else {
                continue;
            };

            if token_id != previous && token_id != self.blank_id {
                token_ids.push(token_id);
            }
            previous = token_id;
        }

        self.decode_ids(&token_ids)
    }
}

pub struct OnnxCtcRuntime {
    session: Session,
    vocabulary: CtcVocabulary,
}

impl OnnxCtcRuntime {
    pub fn new(
        onnx_path: &Path,
        vocab_path: &Path,
        execution: &ExecutionMode,
        vocabulary_options: VocabularyOptions,
    ) -> Result<Self> {
        let vocabulary = CtcVocabulary::from_text_file(vocab_path, vocabulary_options)?;
        let session = build_session(onnx_path, execution)?;
        Ok(Self {
            session,
            vocabulary,
        })
    }

    pub fn transcribe_features(
        &mut self,
        features: ArrayView3<'_, f32>,
        lengths: ArrayView1<'_, i64>,
    ) -> Result<String> {
        self.transcribe_features_with_output_lengths(features, lengths, None)
    }

    pub fn transcribe_features_with_output_lengths(
        &mut self,
        features: ArrayView3<'_, f32>,
        lengths: ArrayView1<'_, i64>,
        output_lengths: Option<ArrayView1<'_, i64>>,
    ) -> Result<String> {
        let outputs = self.session.run(ort::inputs![
            TensorRef::from_array_view(features)?,
            TensorRef::from_array_view(lengths)?,
        ])?;
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .context("failed to extract CTC logits tensor")?;
        let logits = logits
            .into_dimensionality::<Ix3>()
            .map_err(|error| anyhow!("unexpected CTC output shape: {error}"))?;

        Ok(self
            .vocabulary
            .decode_logits_with_lengths(&logits, output_lengths))
    }
}

pub(crate) fn build_session(onnx_path: &Path, execution: &ExecutionMode) -> Result<Session> {
    ensure_ort_runtime_initialized()?;

    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(false)?
        .with_memory_pattern(true)?
        .with_intra_threads(available_threads())?;

    if let Some(gpu_device) = execution.gpu_device() {
        builder = builder.with_execution_providers([CUDAExecutionProvider::default()
            .with_device_id(gpu_device)
            .build()
            .error_on_failure()])?;
    }

    builder.commit_from_file(onnx_path).with_context(|| {
        format!(
            "failed to create ONNX session from `{}`",
            onnx_path.display()
        )
    })
}

pub(crate) fn available_threads() -> usize {
    thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn ensure_ort_runtime_initialized() -> Result<()> {
    match ORT_RUNTIME_INITIALIZED.get_or_init(|| {
        #[cfg(target_os = "linux")]
        let runtime_path = find_ort_runtime_library().map_err(|error| format!("{error:#}"))?;
        #[cfg(not(target_os = "linux"))]
        let runtime_path =
            find_ort_runtime_library_fallback().map_err(|error| format!("{error:#}"))?;

        init_from(runtime_path.display().to_string())
            .commit()
            .map(|_| ())
            .map_err(|error| format!("{error:#}"))
    }) {
        Ok(()) => Ok(()),
        Err(error) => bail!(error.clone()),
    }
}

#[cfg(not(target_os = "linux"))]
fn find_ort_runtime_library_fallback() -> Result<std::path::PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        return Ok(std::path::PathBuf::from(path));
    }
    bail!(
        "failed to locate a dynamic ONNX Runtime library; set ORT_DYLIB_PATH to libonnxruntime.so"
    )
}

#[cfg(target_os = "linux")]
fn find_ort_runtime_library() -> Result<PathBuf> {
    match ORT_RUNTIME_PATH
        .get_or_init(|| find_ort_runtime_library_impl().map_err(|error| format!("{error:#}")))
    {
        Ok(path) => Ok(path.clone()),
        Err(error) => bail!(error.clone()),
    }
}

#[cfg(target_os = "linux")]
fn find_ort_runtime_library_impl() -> Result<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(explicit_path) = env::var("ORT_DYLIB_PATH") {
        candidates.push(PathBuf::from(explicit_path));
    }

    if let Ok(explicit_root) = env::var("TRANSCRIBE_ORT_ROOT") {
        let root = PathBuf::from(explicit_root);
        candidates.push(root.join("libonnxruntime.so"));
        candidates.push(root.join("libonnxruntime.so.1"));
        candidates.push(root.join("libonnxruntime.so.1.24.0"));
        candidates.push(root.join("libonnxruntime.so.1.25.0"));
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("libonnxruntime.so"));
            candidates.push(parent.join("libonnxruntime.so.1"));
            candidates.push(parent.join("libonnxruntime.so.1.24.0"));
            candidates.push(parent.join("libonnxruntime.so.1.25.0"));
        }
    }

    if let Ok(runtime_root) = default_ort_runtime_root_directory() {
        let current_dir = runtime_root.join("current/onnxruntime/capi");
        candidates.push(current_dir.join("libonnxruntime.so"));
        candidates.push(current_dir.join("libonnxruntime.so.1"));
        candidates.push(current_dir.join("libonnxruntime.so.1.24.0"));
        candidates.push(current_dir.join("libonnxruntime.so.1.25.0"));

        if let Ok(entries) = std::fs::read_dir(&runtime_root) {
            let mut runtime_dirs = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path().join("onnxruntime/capi"))
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>();
            runtime_dirs.sort_by(|left, right| right.cmp(left));
            for dir in runtime_dirs {
                candidates.push(dir.join("libonnxruntime.so"));
                candidates.push(dir.join("libonnxruntime.so.1"));
                candidates.push(dir.join("libonnxruntime.so.1.24.0"));
                candidates.push(dir.join("libonnxruntime.so.1.25.0"));
            }
        }
    }

    candidates.into_iter().find(|path| path.is_file()).with_context(|| {
        "failed to locate a dynamic ONNX Runtime library (`libonnxruntime.so*`); use `--gpu` once so transcribe-cli can populate `<binary_dir>/transcribe_sandbox/ort-cuda13-nightly`, or set ORT_DYLIB_PATH"
    })
}

#[cfg(test)]
mod tests {
    use super::{CtcVocabulary, VocabularyOptions};
    use ndarray::arr3;

    #[test]
    fn converts_vocabulary_tokens_into_text() {
        let vocabulary = CtcVocabulary {
            entries: vec![
                " ".to_string(),
                "п".to_string(),
                "р".to_string(),
                "и".to_string(),
                "<blk>".to_string(),
            ],
            blank_id: 4,
        };

        let text = vocabulary.decode_ids(&[1, 2, 3, 0, 1]);
        assert_eq!(text, "при п");
    }

    #[test]
    fn preserves_raw_spacing_tokens() {
        let vocabulary = CtcVocabulary {
            entries: vec![" ".to_string(), "а".to_string(), "<blk>".to_string()],
            blank_id: 2,
        };

        let text = vocabulary.decode_ids(&[0, 1, 0, 1]);
        assert_eq!(text, "а а");
    }

    #[test]
    fn supports_disabling_word_boundary_rewrite() {
        let vocabulary = CtcVocabulary {
            entries: vec!["_".to_string(), "a".to_string(), "<blk>".to_string()],
            blank_id: 2,
        };

        let text = vocabulary.decode_ids(&[0, 1]);
        assert_eq!(text, "_a");
    }

    #[test]
    fn default_vocabulary_options_match_sentencepiece_style_ctc() {
        let options = VocabularyOptions::default();
        assert_eq!(options.blank_token, "<blk>");
        assert_eq!(options.word_boundary_token, Some("▁"));
    }

    #[test]
    fn replaces_word_boundary_prefix_inside_tokens_from_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, "▁the 0\ncat 1\n<blk> 2\n").expect("write vocab");

        let vocabulary =
            CtcVocabulary::from_text_file(&path, VocabularyOptions::default()).expect("vocab");

        let text = vocabulary.decode_ids(&[0, 1]);
        assert_eq!(text, "thecat");
    }

    #[test]
    fn keeps_word_boundaries_for_sentencepiece_tokens() {
        let vocabulary = CtcVocabulary {
            entries: vec![
                " the".to_string(),
                "re".to_string(),
                " cat".to_string(),
                "<blk>".to_string(),
            ],
            blank_id: 3,
        };

        let text = vocabulary.decode_ids(&[0, 1, 2]);
        assert_eq!(text, "there cat");
    }

    #[test]
    fn ignores_logits_past_valid_output_length() {
        let vocabulary = CtcVocabulary {
            entries: vec![" a".to_string(), " b".to_string(), "<blk>".to_string()],
            blank_id: 2,
        };
        let logits = arr3(&[[[10.0, 0.0, -1.0], [9.0, 0.0, -1.0], [0.0, 10.0, -1.0]]]);
        let lengths = ndarray::arr1(&[2_i64]);

        let text = vocabulary.decode_logits_with_lengths(&logits.view(), Some(lengths.view()));
        assert_eq!(text, "a");
    }
}
