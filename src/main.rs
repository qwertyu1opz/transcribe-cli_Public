mod audio;
mod dynamic_chunk;
mod gigaam;
mod model;
mod model_download;
mod onnx_ctc;
mod onnx_transducer;
mod parakeet;
mod rest;
mod server;
mod transcribe;
mod video;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::audio::prepare_audio_source;
use crate::model::{
    ModelChoice, ModelComputeType, default_model_root_directory, read_model_config,
    remove_all_models, remove_model_with_artifacts,
};
use crate::model_download::ensure_model_downloaded;
#[cfg(target_os = "linux")]
use crate::model_download::ensure_ort_runtime_downloaded;
use crate::onnx_ctc::ExecutionMode;
use crate::server::run_server;
use crate::transcribe::{
    run_gigaam_transcription, run_live_gigaam_transcription, run_live_parakeet_transcription,
    run_parakeet_transcription,
};

#[derive(Debug, Parser)]
#[command(
    name = "transcribe-cli",
    version,
    about = "Native Rust transcription CLI with GigaAM v3 ONNX"
)]
struct Cli {
    #[arg(
        long,
        value_name = "PORT",
        help = "Start the REST server on the given port instead of running a single CLI transcription"
    )]
    server: Option<u16>,

    #[arg(
        long,
        value_name = "DIR",
        help = "Model storage directory; defaults to <binary_dir>/transcribe_sandbox/models"
    )]
    models_dir: Option<PathBuf>,

    #[arg(long, help = "Use Nvidia GPU execution")]
    gpu: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "CUDA device index to use with --gpu"
    )]
    gpu_device: i32,

    #[arg(
        long,
        value_enum,
        help = "Override compute type; accepted values are auto, int8, float32"
    )]
    compute_type: Option<CliComputeType>,

    #[arg(
        long,
        alias = "laungage",
        value_name = "LANG",
        help = "Language selects the backend: default ru uses GigaAM, en uses Parakeet-TDT v2"
    )]
    language: Option<String>,

    #[arg(
        long,
        help = "Print transcript chunk-by-chunk while the model is decoding"
    )]
    stream: bool,

    #[arg(
        long,
        help = "Treat MEDIA as a live HTTP/HTTPS media stream and transcribe it incrementally"
    )]
    live: bool,

    #[arg(
        long,
        help = "Enable lightweight VAD segmentation for --live so only speech-like chunks are sent to the model"
    )]
    vad: bool,

    #[arg(
        long,
        conflicts_with = "remove_all",
        help = "Remove the selected model and related leftovers from the models directory"
    )]
    remove_model: bool,

    #[arg(long, help = "Remove the entire models directory")]
    remove_all: bool,

    #[arg(value_name = "MEDIA", help = "Path or URL to an audio or video file")]
    audio: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliComputeType {
    Auto,
    #[value(name = "int8")]
    Int8,
    #[value(name = "float32")]
    Float32,
}

impl CliComputeType {
    fn gigaam_type(self) -> Option<ModelComputeType> {
        match self {
            Self::Auto => None,
            Self::Int8 => Some(ModelComputeType::Int8),
            Self::Float32 => Some(ModelComputeType::Float32),
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let language = resolve_requested_language(cli.language.as_deref())?;
    let model_choice = select_model_for_language(&language);

    let default_models_root = default_model_root_directory()?;
    let models_root = cli
        .models_dir
        .as_deref()
        .unwrap_or(default_models_root.as_path());

    if cli.remove_all {
        let removed = remove_all_models(models_root)?;
        if removed {
            println!("removed models directory `{}`", models_root.display());
        } else {
            println!(
                "models directory `{}` does not exist",
                models_root.display()
            );
        }
        return Ok(());
    }

    if cli.remove_model {
        let removed_entries = remove_model_with_artifacts(model_choice, models_root)?;
        if removed_entries > 0 {
            println!(
                "removed model `{}` and {} related artifact(s) from `{}`",
                model_choice.cli_name(),
                removed_entries,
                models_root.display()
            );
        } else {
            println!(
                "no artifacts found for model `{}` in `{}`",
                model_choice.cli_name(),
                models_root.display()
            );
        }
        return Ok(());
    }

    if let Some(port) = cli.server {
        run_server(port, models_root.to_path_buf()).await?;
        return Ok(());
    }

    let audio_input = cli.audio.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "a media path or URL is required unless --server, --remove-model or --remove-all is used"
        )
    })?;

    if cli.vad && !cli.live {
        anyhow::bail!("`--vad` is currently supported only together with `--live`");
    }

    let execution = ExecutionMode::from_cli(
        cli.gpu,
        cli.gpu_device,
        cli.compute_type.and_then(CliComputeType::gigaam_type),
    )?;
    #[cfg(target_os = "linux")]
    ensure_ort_runtime_downloaded().await?;
    let model_dir = ensure_model_downloaded(
        model_choice,
        Some(execution.compute_type()),
        Some(models_root),
    )
    .await?;
    let model_config = read_model_config(&model_dir, model_choice)?;
    print_model_parameters(
        model_choice,
        &model_dir,
        models_root,
        execution.device_label(),
        execution.compute_type_label(),
        &language,
        execution.gpu_device(),
        &model_config,
    );

    match model_choice {
        ModelChoice::GigaamV3 if cli.live => {
            run_live_gigaam_transcription(
                audio_input,
                model_choice,
                &model_dir,
                &execution,
                Some(&language),
                cli.vad,
            )
            .await
        }
        ModelChoice::ParakeetTdt06bV2 if cli.live => {
            run_live_parakeet_transcription(
                audio_input,
                model_choice,
                &model_dir,
                &execution,
                Some(&language),
                cli.vad,
            )
            .await
        }
        ModelChoice::GigaamV3 => {
            let prepare_bar = ProgressBar::new_spinner();
            prepare_bar.set_style(ProgressStyle::with_template(
                "  preparing media  {spinner:.green} {msg}",
            )?);
            prepare_bar.enable_steady_tick(std::time::Duration::from_millis(80));
            prepare_bar.set_message(audio_input.to_string());
            let audio = prepare_audio_source(audio_input).await?;
            prepare_bar.finish_with_message("media ready");

            run_gigaam_transcription(
                &audio,
                model_choice,
                &model_dir,
                &execution,
                Some(&language),
                cli.stream,
            )
            .await
        }
        ModelChoice::ParakeetTdt06bV2 => {
            let prepare_bar = ProgressBar::new_spinner();
            prepare_bar.set_style(ProgressStyle::with_template(
                "  preparing media  {spinner:.green} {msg}",
            )?);
            prepare_bar.enable_steady_tick(std::time::Duration::from_millis(80));
            prepare_bar.set_message(audio_input.to_string());
            let audio = prepare_audio_source(audio_input).await?;
            prepare_bar.finish_with_message("media ready");

            run_parakeet_transcription(
                &audio,
                model_choice,
                &model_dir,
                &execution,
                Some(&language),
                cli.stream,
            )
            .await
        }
    }
}

fn resolve_requested_language(language: Option<&str>) -> Result<String> {
    let normalized = language
        .map(str::trim)
        .filter(|language| !language.is_empty())
        .unwrap_or("ru")
        .to_ascii_lowercase();

    match normalized.as_str() {
        "ru" | "rus" | "auto" => Ok(String::from("ru")),
        "en" | "eng" => Ok(String::from("en")),
        other => anyhow::bail!(
            "unsupported language `{other}`; currently supported values are `ru` and `en`"
        ),
    }
}

fn select_model_for_language(language: &str) -> ModelChoice {
    match language {
        "en" => ModelChoice::ParakeetTdt06bV2,
        _ => ModelChoice::GigaamV3,
    }
}

fn print_model_parameters(
    model_choice: ModelChoice,
    model_dir: &std::path::Path,
    models_root: &std::path::Path,
    device_label: &str,
    compute_type_label: &str,
    language_label: &str,
    gpu_device: Option<i32>,
    model_config: &crate::model::ModelConfig,
) {
    println!();
    println!("Model parameters");
    println!("  requested model : {}", model_choice.cli_name());
    println!("  runtime model   : {}", model_choice.runtime_name());
    println!("  repo            : {}", model_choice.repo_id());
    println!("  models root     : {}", models_root.display());
    println!("  local path      : {}", model_dir.display());
    println!("  device          : {device_label}");
    println!("  compute type    : {compute_type_label}");
    println!("  language        : {language_label}");
    if let Some(gpu_device) = gpu_device {
        println!("  gpu device      : {gpu_device}");
    }
    if let Some(model_name) = model_config.model_name.as_deref() {
        println!("  model name      : {model_name}");
    }
    if let Some(model_class) = model_config.model_class.as_deref() {
        println!("  model class     : {model_class}");
    }
    if let Some(model_type) = model_config.model_type.as_deref() {
        println!("  model type      : {model_type}");
    }
    if let Some(sample_rate) = model_config.sample_rate {
        println!("  sample rate     : {sample_rate} Hz");
    }
    if let Some(features) = model_config.features {
        println!("  mel bins        : {features}");
    }
    if let Some(win_length) = model_config.win_length {
        println!("  win length      : {win_length}");
    }
    if let Some(hop_length) = model_config.hop_length {
        println!("  hop length      : {hop_length}");
    }
    if let Some(n_fft) = model_config.n_fft {
        println!("  n_fft           : {n_fft}");
    }
    if let Some(center) = model_config.center {
        println!("  center          : {center}");
    }
    if let Some(encoder_layers) = model_config.encoder_layers {
        println!("  encoder layers  : {encoder_layers}");
    }
    if let Some(d_model) = model_config.d_model {
        println!("  d_model         : {d_model}");
    }
    if let Some(num_classes) = model_config.num_classes {
        println!("  num classes     : {num_classes}");
    }
    if let Some(subsampling_factor) = model_config.subsampling_factor {
        println!("  subsampling     : {subsampling_factor}");
    }
}
