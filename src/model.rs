use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelChoice {
    GigaamV3,
    ParakeetTdt06bV2,
}

impl ModelChoice {
    pub fn cli_name(self) -> &'static str {
        match self {
            Self::GigaamV3 => "gigaam-v3",
            Self::ParakeetTdt06bV2 => "parakeet-tdt-0.6b-v2",
        }
    }

    pub fn runtime_name(self) -> &'static str {
        match self {
            Self::GigaamV3 => "v3_e2e_ctc",
            Self::ParakeetTdt06bV2 => "parakeet-tdt-0.6b-v2",
        }
    }

    pub fn repo_id(self) -> &'static str {
        match self {
            Self::GigaamV3 => "istupakov/gigaam-v3-onnx",
            Self::ParakeetTdt06bV2 => "istupakov/parakeet-tdt-0.6b-v2-onnx",
        }
    }

    pub fn cache_dir_name(self) -> &'static str {
        match self {
            Self::GigaamV3 => "gigaam-v3",
            Self::ParakeetTdt06bV2 => "parakeet-tdt-0.6b-v2",
        }
    }

    pub fn config_file(self) -> &'static str {
        match self {
            Self::GigaamV3 => "v3_e2e_ctc.yaml",
            Self::ParakeetTdt06bV2 => "config.json",
        }
    }

    pub fn vocab_file(self) -> &'static str {
        match self {
            Self::GigaamV3 => "v3_e2e_ctc_vocab.txt",
            Self::ParakeetTdt06bV2 => "vocab.txt",
        }
    }

    pub fn onnx_file(self, compute_type: ModelComputeType) -> &'static str {
        match self {
            Self::GigaamV3 => match compute_type {
                ModelComputeType::Float32 => "v3_e2e_ctc.onnx",
                ModelComputeType::Int8 => "v3_e2e_ctc.int8.onnx",
            },
            Self::ParakeetTdt06bV2 => match compute_type {
                ModelComputeType::Float32 => "encoder-model.onnx",
                ModelComputeType::Int8 => "encoder-model.int8.onnx",
            },
        }
    }

    pub fn secondary_onnx_file(self, compute_type: ModelComputeType) -> Option<&'static str> {
        match self {
            Self::GigaamV3 => None,
            Self::ParakeetTdt06bV2 => Some(match compute_type {
                ModelComputeType::Float32 => "decoder_joint-model.onnx",
                ModelComputeType::Int8 => "decoder_joint-model.int8.onnx",
            }),
        }
    }

    pub fn extra_required_files(self, compute_type: ModelComputeType) -> &'static [&'static str] {
        match self {
            Self::GigaamV3 => &[],
            Self::ParakeetTdt06bV2 => match compute_type {
                ModelComputeType::Float32 => {
                    &["encoder-model.onnx.data", "decoder_joint-model.onnx"]
                }
                ModelComputeType::Int8 => &["decoder_joint-model.int8.onnx"],
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelComputeType {
    Float32,
    Int8,
}

impl ModelComputeType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Int8 => "int8",
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct HfModelResponse {
    #[serde(default)]
    pub(crate) siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct HfSibling {
    pub(crate) rfilename: String,
    pub(crate) size: Option<u64>,
    pub(crate) lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct HfLfs {
    pub(crate) size: Option<u64>,
}

impl HfSibling {
    pub(crate) fn expected_size(&self) -> Option<u64> {
        self.size
            .or_else(|| self.lfs.as_ref().and_then(|lfs| lfs.size))
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelConfig {
    pub model_name: Option<String>,
    pub model_class: Option<String>,
    pub model_type: Option<String>,
    pub sample_rate: Option<u32>,
    pub features: Option<u32>,
    pub win_length: Option<u32>,
    pub hop_length: Option<u32>,
    pub n_fft: Option<u32>,
    pub center: Option<bool>,
    pub encoder_layers: Option<u32>,
    pub d_model: Option<u32>,
    pub num_classes: Option<u32>,
    pub subsampling_factor: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawGigaAmModelConfig {
    model_name: Option<String>,
    model_class: Option<String>,
    sample_rate: Option<u32>,
    preprocessor: Option<RawPreprocessorConfig>,
    encoder: Option<RawEncoderConfig>,
    head: Option<RawHeadConfig>,
}

#[derive(Debug, Deserialize)]
struct RawPreprocessorConfig {
    sample_rate: Option<u32>,
    features: Option<u32>,
    win_length: Option<u32>,
    hop_length: Option<u32>,
    n_fft: Option<u32>,
    center: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawEncoderConfig {
    n_layers: Option<u32>,
    d_model: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawHeadConfig {
    num_classes: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawParakeetModelConfig {
    model_type: Option<String>,
    features_size: Option<u32>,
    subsampling_factor: Option<u32>,
}

pub fn model_directory(choice: ModelChoice, models_root: Option<&Path>) -> Result<PathBuf> {
    let models_dir = model_root_directory(models_root)?;
    Ok(models_dir.join(choice.cache_dir_name()))
}

pub fn sandbox_directory() -> Result<PathBuf> {
    Ok(binary_directory()?.join("transcribe_sandbox"))
}

pub fn default_model_root_directory() -> Result<PathBuf> {
    Ok(sandbox_directory()?.join("models"))
}

#[cfg(target_os = "linux")]
pub fn default_ort_runtime_root_directory() -> Result<PathBuf> {
    Ok(sandbox_directory()?.join("ort-cuda13-nightly"))
}

pub fn binary_directory() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("failed to resolve executable path")?;
    executable
        .parent()
        .map(Path::to_path_buf)
        .context("failed to resolve executable directory")
}

fn model_root_directory(models_root: Option<&Path>) -> Result<PathBuf> {
    if let Some(models_root) = models_root {
        return Ok(models_root.to_path_buf());
    }

    default_model_root_directory()
}

pub fn read_model_config(model_dir: &Path, choice: ModelChoice) -> Result<ModelConfig> {
    match choice {
        ModelChoice::GigaamV3 => read_gigaam_model_config(model_dir, choice),
        ModelChoice::ParakeetTdt06bV2 => read_parakeet_model_config(model_dir, choice),
    }
}

fn read_gigaam_model_config(model_dir: &Path, choice: ModelChoice) -> Result<ModelConfig> {
    let config_path = model_dir.join(choice.config_file());
    let config_contents = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read `{}`", config_path.display()))?;
    let raw: RawGigaAmModelConfig = serde_yaml::from_str(&config_contents)
        .with_context(|| format!("failed to parse `{}`", config_path.display()))?;

    Ok(ModelConfig {
        model_name: raw.model_name,
        model_class: raw.model_class,
        model_type: None,
        sample_rate: raw
            .preprocessor
            .as_ref()
            .and_then(|config| config.sample_rate)
            .or(raw.sample_rate),
        features: raw.preprocessor.as_ref().and_then(|config| config.features),
        win_length: raw
            .preprocessor
            .as_ref()
            .and_then(|config| config.win_length),
        hop_length: raw
            .preprocessor
            .as_ref()
            .and_then(|config| config.hop_length),
        n_fft: raw.preprocessor.as_ref().and_then(|config| config.n_fft),
        center: raw.preprocessor.as_ref().and_then(|config| config.center),
        encoder_layers: raw.encoder.as_ref().and_then(|config| config.n_layers),
        d_model: raw.encoder.as_ref().and_then(|config| config.d_model),
        num_classes: raw.head.as_ref().and_then(|config| config.num_classes),
        subsampling_factor: None,
    })
}

fn read_parakeet_model_config(model_dir: &Path, choice: ModelChoice) -> Result<ModelConfig> {
    let config_path = model_dir.join(choice.config_file());
    let config_contents = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read `{}`", config_path.display()))?;
    let raw: RawParakeetModelConfig = serde_json::from_str(&config_contents)
        .with_context(|| format!("failed to parse `{}`", config_path.display()))?;

    Ok(ModelConfig {
        model_name: Some(String::from("Parakeet TDT 0.6B V2")),
        model_class: None,
        model_type: raw.model_type,
        sample_rate: Some(16_000),
        features: raw.features_size,
        win_length: Some(400),
        hop_length: Some(160),
        n_fft: Some(512),
        center: Some(false),
        encoder_layers: None,
        d_model: None,
        num_classes: None,
        subsampling_factor: raw.subsampling_factor,
    })
}

pub fn remove_model_with_artifacts(choice: ModelChoice, models_root: &Path) -> Result<usize> {
    if !models_root.exists() {
        return Ok(0);
    }

    let model_name = choice.cache_dir_name();
    let mut removed = 0;

    for entry in std::fs::read_dir(models_root)
        .with_context(|| format!("failed to read `{}`", models_root.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to inspect `{}`", models_root.display()))?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if !matches_model_artifact(file_name, model_name) {
            continue;
        }

        remove_path(&path)?;
        removed += 1;
    }

    Ok(removed)
}

pub fn remove_all_models(models_root: &Path) -> Result<bool> {
    if !models_root.exists() {
        return Ok(false);
    }

    std::fs::remove_dir_all(models_root)
        .with_context(|| format!("failed to remove `{}`", models_root.display()))?;
    Ok(true)
}

fn matches_model_artifact(file_name: &str, model_name: &str) -> bool {
    file_name == model_name
        || file_name.starts_with(&format!("{model_name}."))
        || file_name.starts_with(&format!("{model_name}-"))
}

fn remove_path(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect `{}`", path.display()))?;

    if metadata.is_dir() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("failed to remove directory `{}`", path.display()))?;
    } else {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove file `{}`", path.display()))?;
    }

    Ok(())
}
