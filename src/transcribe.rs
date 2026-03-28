use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use reqwest::header::CONTENT_TYPE;
use tempfile::Builder;
use tokio::task;
use url::Url;

use crate::audio::{AudioMetadata, PreparedAudio, inspect_audio_file, prepare_audio_source};
use crate::dynamic_chunk::{DynamicChunkConfig, DynamicChunkEngine};
use crate::gigaam::GigaAm;
use crate::model::ModelChoice;
use crate::model_download::ensure_model_downloaded;
#[cfg(target_os = "linux")]
use crate::model_download::ensure_ort_runtime_downloaded;
use crate::onnx_ctc::ExecutionMode;
use crate::parakeet::Parakeet;

const LIVE_DECODE_INTERVAL: Duration = Duration::from_millis(450);
const LIVE_MIN_DECODE_BYTES: usize = 64 * 1024;
const LIVE_NATIVE_CHUNK_SECONDS: usize = 6;

pub struct RestTranscriptionResult {
    pub transcript: String,
    pub audio_metadata: AudioMetadata,
}

pub async fn run_gigaam_transcription(
    audio: &PreparedAudio,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    stream_output: bool,
) -> Result<()> {
    validate_gigaam_language(language)?;

    let loading = ProgressBar::new_spinner();
    loading.set_style(
        ProgressStyle::with_template("  loading model     {spinner:.green} {msg}")
            .context("failed to configure loading spinner")?,
    );
    loading.enable_steady_tick(Duration::from_millis(80));
    loading.set_message(model_dir.display().to_string());

    let mut gigaam = GigaAm::new(model_dir, model_choice, execution)
        .context("failed to initialize GigaAM backend")?;
    loading.finish_with_message("model loaded");

    print_audio_parameters(
        audio,
        model_dir,
        execution.device_label(),
        execution.compute_type_label(),
        execution.gpu_device(),
        "ONNX Runtime",
    );

    if gigaam.sampling_rate() != audio.metadata.target_sample_rate as usize {
        bail!(
            "audio was resampled to {} Hz but the GigaAM model expects {} Hz",
            audio.metadata.target_sample_rate,
            gigaam.sampling_rate()
        );
    }

    if stream_output {
        let processing = make_processing_spinner(&audio.display_name)?;
        let mut output_state = StreamOutputState::default();
        gigaam
            .transcribe_with_callback(&audio.samples, |_, _, chunk_text| {
                stream_print_chunk(&processing, &mut output_state, chunk_text)
            })
            .context("GigaAM transcription failed")?;
        processing.finish_with_message("processing complete");
        finish_stream_output(&output_state);
    } else {
        let processing = make_processing_spinner(&audio.display_name)?;
        let transcript = gigaam
            .transcribe(&audio.samples)
            .context("GigaAM transcription failed")?;
        processing.finish_with_message("processing complete");
        print_transcript(transcript.trim());
    }
    Ok(())
}

pub async fn run_parakeet_transcription(
    audio: &PreparedAudio,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    stream_output: bool,
) -> Result<()> {
    validate_parakeet_language(language)?;

    let loading = ProgressBar::new_spinner();
    loading.set_style(
        ProgressStyle::with_template("  loading model     {spinner:.green} {msg}")
            .context("failed to configure loading spinner")?,
    );
    loading.enable_steady_tick(Duration::from_millis(80));
    loading.set_message(model_dir.display().to_string());

    let mut parakeet = Parakeet::new(model_dir, model_choice, execution)
        .context("failed to initialize Parakeet backend")?;
    loading.finish_with_message("model loaded");

    print_audio_parameters(
        audio,
        model_dir,
        execution.device_label(),
        execution.compute_type_label(),
        execution.gpu_device(),
        "ONNX Runtime",
    );

    if parakeet.sampling_rate() != audio.metadata.target_sample_rate as usize {
        bail!(
            "audio was resampled to {} Hz but the Parakeet model expects {} Hz",
            audio.metadata.target_sample_rate,
            parakeet.sampling_rate()
        );
    }

    if stream_output {
        let processing = make_processing_spinner(&audio.display_name)?;
        let mut output_state = StreamOutputState::default();
        parakeet
            .transcribe_with_callback(&audio.samples, |_, _, chunk_text| {
                stream_print_chunk(&processing, &mut output_state, chunk_text)
            })
            .context("Parakeet TDT transcription failed")?;
        processing.finish_with_message("processing complete");
        finish_stream_output(&output_state);
    } else {
        let processing = make_processing_spinner(&audio.display_name)?;
        let transcript = parakeet
            .transcribe(&audio.samples)
            .context("Parakeet TDT transcription failed")?;
        processing.finish_with_message("processing complete");
        print_transcript(transcript.trim());
    }
    Ok(())
}

pub async fn transcribe_media_for_rest(
    media: &str,
    language: &str,
    model_choice: ModelChoice,
    models_root: &Path,
    execution: &ExecutionMode,
) -> Result<RestTranscriptionResult> {
    #[cfg(target_os = "linux")]
    ensure_ort_runtime_downloaded().await?;

    let model_dir = ensure_model_downloaded(
        model_choice,
        Some(execution.compute_type()),
        Some(models_root),
    )
    .await?;
    let audio = prepare_audio_source(media).await?;
    let transcript =
        transcribe_prepared_audio(&audio, model_choice, &model_dir, execution, Some(language))
            .await?;

    Ok(RestTranscriptionResult {
        transcript,
        audio_metadata: audio.metadata.clone(),
    })
}

pub async fn stream_media_for_rest<F, S>(
    media: &str,
    language: &str,
    model_choice: ModelChoice,
    models_root: &Path,
    execution: &ExecutionMode,
    mut on_chunk: F,
    mut on_status: S,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
    S: FnMut(String),
{
    #[cfg(target_os = "linux")]
    ensure_ort_runtime_downloaded().await?;

    on_status(String::from("resolving model"));
    let model_dir = ensure_model_downloaded(
        model_choice,
        Some(execution.compute_type()),
        Some(models_root),
    )
    .await?;

    on_status(String::from("preparing media"));
    let audio = prepare_audio_source(media).await?;
    on_status(String::from("media ready"));

    stream_prepared_audio(
        &audio,
        model_choice,
        &model_dir,
        execution,
        Some(language),
        &mut on_chunk,
    )
    .await
}

pub async fn run_live_gigaam_transcription(
    source_url: &str,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    use_vad: bool,
) -> Result<()> {
    validate_gigaam_language(language)?;

    let loading = ProgressBar::new_spinner();
    loading.set_style(
        ProgressStyle::with_template("  loading model     {spinner:.green} {msg}")
            .context("failed to configure loading spinner")?,
    );
    loading.enable_steady_tick(Duration::from_millis(80));
    loading.set_message(model_dir.display().to_string());

    let mut gigaam = GigaAm::new(model_dir, model_choice, execution)
        .context("failed to initialize GigaAM backend")?;
    loading.finish_with_message("model loaded");
    let processing = make_processing_spinner(source_url)?;
    let mut output_state = StreamOutputState::default();

    run_live_transcription_loop(
        source_url,
        model_dir,
        execution,
        gigaam.sampling_rate(),
        "ONNX Runtime",
        use_vad,
        |samples| gigaam.transcribe(samples),
        |chunk_text| stream_print_chunk(&processing, &mut output_state, chunk_text),
        |status| processing.set_message(status),
    )
    .await
    .context("GigaAM live transcription failed")?;
    processing.finish_with_message("processing complete");
    finish_stream_output(&output_state);
    Ok(())
}

pub async fn run_live_parakeet_transcription(
    source_url: &str,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    use_vad: bool,
) -> Result<()> {
    validate_parakeet_language(language)?;

    let loading = ProgressBar::new_spinner();
    loading.set_style(
        ProgressStyle::with_template("  loading model     {spinner:.green} {msg}")
            .context("failed to configure loading spinner")?,
    );
    loading.enable_steady_tick(Duration::from_millis(80));
    loading.set_message(model_dir.display().to_string());

    let mut parakeet = Parakeet::new(model_dir, model_choice, execution)
        .context("failed to initialize Parakeet backend")?;
    loading.finish_with_message("model loaded");
    let processing = make_processing_spinner(source_url)?;
    let mut output_state = StreamOutputState::default();

    run_live_transcription_loop(
        source_url,
        model_dir,
        execution,
        parakeet.sampling_rate(),
        "ONNX Runtime",
        use_vad,
        |samples| parakeet.transcribe(samples),
        |chunk_text| stream_print_chunk(&processing, &mut output_state, chunk_text),
        |status| processing.set_message(status),
    )
    .await
    .context("Parakeet live transcription failed")?;
    processing.finish_with_message("processing complete");
    finish_stream_output(&output_state);
    Ok(())
}

pub async fn stream_live_gigaam_transcription<F, S>(
    source_url: &str,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    use_vad: bool,
    on_chunk: F,
    on_status: S,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
    S: FnMut(String),
{
    validate_gigaam_language(language)?;
    let mut gigaam = GigaAm::new(model_dir, model_choice, execution)
        .context("failed to initialize GigaAM backend")?;

    run_live_transcription_loop(
        source_url,
        model_dir,
        execution,
        gigaam.sampling_rate(),
        "ONNX Runtime",
        use_vad,
        |samples| gigaam.transcribe(samples),
        on_chunk,
        on_status,
    )
    .await
    .context("GigaAM live transcription failed")
}

pub async fn stream_live_parakeet_transcription<F, S>(
    source_url: &str,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    use_vad: bool,
    on_chunk: F,
    on_status: S,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
    S: FnMut(String),
{
    validate_parakeet_language(language)?;
    let mut parakeet = Parakeet::new(model_dir, model_choice, execution)
        .context("failed to initialize Parakeet backend")?;

    run_live_transcription_loop(
        source_url,
        model_dir,
        execution,
        parakeet.sampling_rate(),
        "ONNX Runtime",
        use_vad,
        |samples| parakeet.transcribe(samples),
        on_chunk,
        on_status,
    )
    .await
    .context("Parakeet live transcription failed")
}

async fn transcribe_prepared_audio(
    audio: &PreparedAudio,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
) -> Result<String> {
    match model_choice {
        ModelChoice::GigaamV3 => {
            validate_gigaam_language(language)?;
            let mut gigaam = GigaAm::new(model_dir, model_choice, execution)
                .context("failed to initialize GigaAM backend")?;

            if gigaam.sampling_rate() != audio.metadata.target_sample_rate as usize {
                bail!(
                    "audio was resampled to {} Hz but the GigaAM model expects {} Hz",
                    audio.metadata.target_sample_rate,
                    gigaam.sampling_rate()
                );
            }

            gigaam.transcribe(&audio.samples)
        }
        ModelChoice::ParakeetTdt06bV2 => {
            validate_parakeet_language(language)?;
            let mut parakeet = Parakeet::new(model_dir, model_choice, execution)
                .context("failed to initialize Parakeet backend")?;

            if parakeet.sampling_rate() != audio.metadata.target_sample_rate as usize {
                bail!(
                    "audio was resampled to {} Hz but the Parakeet model expects {} Hz",
                    audio.metadata.target_sample_rate,
                    parakeet.sampling_rate()
                );
            }

            parakeet.transcribe(&audio.samples)
        }
    }
}

async fn stream_prepared_audio<F>(
    audio: &PreparedAudio,
    model_choice: ModelChoice,
    model_dir: &Path,
    execution: &ExecutionMode,
    language: Option<&str>,
    mut on_chunk: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
{
    match model_choice {
        ModelChoice::GigaamV3 => {
            validate_gigaam_language(language)?;
            let mut gigaam = GigaAm::new(model_dir, model_choice, execution)
                .context("failed to initialize GigaAM backend")?;

            if gigaam.sampling_rate() != audio.metadata.target_sample_rate as usize {
                bail!(
                    "audio was resampled to {} Hz but the GigaAM model expects {} Hz",
                    audio.metadata.target_sample_rate,
                    gigaam.sampling_rate()
                );
            }

            gigaam.transcribe_with_callback(&audio.samples, |_, _, chunk_text| on_chunk(chunk_text))
        }
        ModelChoice::ParakeetTdt06bV2 => {
            validate_parakeet_language(language)?;
            let mut parakeet = Parakeet::new(model_dir, model_choice, execution)
                .context("failed to initialize Parakeet backend")?;

            if parakeet.sampling_rate() != audio.metadata.target_sample_rate as usize {
                bail!(
                    "audio was resampled to {} Hz but the Parakeet model expects {} Hz",
                    audio.metadata.target_sample_rate,
                    parakeet.sampling_rate()
                );
            }

            parakeet
                .transcribe_with_callback(&audio.samples, |_, _, chunk_text| on_chunk(chunk_text))
        }
    }
}

fn validate_gigaam_language(language: Option<&str>) -> Result<()> {
    let Some(language) = language
        .map(str::trim)
        .filter(|language| !language.is_empty())
    else {
        return Ok(());
    };

    let normalized = language.to_ascii_lowercase();
    if matches!(normalized.as_str(), "ru" | "rus" | "auto") {
        return Ok(());
    }

    bail!("GigaAM v3 community ONNX backend is currently Russian-only; use `--language ru`")
}

fn validate_parakeet_language(language: Option<&str>) -> Result<()> {
    let Some(language) = language
        .map(str::trim)
        .filter(|language| !language.is_empty())
    else {
        bail!("Parakeet English backend requires `--language en`");
    };

    let normalized = language.to_ascii_lowercase();
    if matches!(normalized.as_str(), "en" | "eng") {
        return Ok(());
    }

    bail!("Parakeet TDT backend is enabled only with `--language en`")
}

async fn run_live_transcription_loop<F, C, S>(
    source_url: &str,
    model_dir: &Path,
    execution: &ExecutionMode,
    sample_rate: usize,
    backend_name: &str,
    use_vad: bool,
    mut transcribe_segment: F,
    mut on_chunk: C,
    mut on_status: S,
) -> Result<()>
where
    F: FnMut(&[f32]) -> Result<String>,
    C: FnMut(&str) -> Result<()>,
    S: FnMut(String),
{
    let url = Url::parse(source_url)
        .with_context(|| format!("failed to parse live URL `{source_url}`"))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("`--live` currently supports only http/https URLs");
    }

    print_live_audio_parameters(
        source_url,
        model_dir,
        execution.device_label(),
        execution.compute_type_label(),
        execution.gpu_device(),
        backend_name,
        sample_rate as u32,
    );

    let mut vad_engine = build_live_vad_engine(use_vad, sample_rate)?;
    let client = Client::builder()
        .user_agent("transcribe-cli/0.1.0")
        .build()
        .context("failed to build HTTP client")?;
    let response = client
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("failed to open live stream `{url}`"))?
        .error_for_status()
        .with_context(|| format!("live stream returned an error for `{url}`"))?;

    if live_stream_looks_like_raw_pcm(response.headers()) {
        return run_raw_live_transcription_loop(
            source_url,
            model_dir,
            execution,
            backend_name,
            sample_rate,
            response,
            vad_engine.as_mut(),
            transcribe_segment,
            &mut on_chunk,
            &mut on_status,
        )
        .await;
    }

    if live_stream_looks_like_html(response.headers()) {
        bail!(
            "`--live` expected a media stream, but `{source_url}` returned HTML; if this is your local mic server, use the stream endpoint such as `{source_url}/stream`"
        );
    }

    let mut stream = response.bytes_stream();
    let temp_file = Builder::new()
        .prefix("transcribe-cli-live-")
        .suffix(".media")
        .tempfile()
        .context("failed to create temporary live media snapshot")?;
    let temp_path = temp_file.path().to_path_buf();
    let mut snapshot_file = temp_file
        .reopen()
        .context("failed to reopen temporary live media snapshot")?;

    let mut last_decoded_samples = 0usize;
    let mut bytes_since_decode = 0usize;
    let mut saw_successful_decode = false;
    let mut last_decode_attempt = Instant::now();
    let mut last_decode_error: Option<anyhow::Error> = None;
    let mut pending_samples = Vec::new();
    let live_chunk_samples = sample_rate * LIVE_NATIVE_CHUNK_SECONDS;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed to read bytes from live stream")?;
        if chunk.is_empty() {
            continue;
        }

        snapshot_file
            .write_all(chunk.as_ref())
            .context("failed to append bytes to live media snapshot")?;
        snapshot_file
            .flush()
            .context("failed to flush live media snapshot")?;
        bytes_since_decode += chunk.len();

        if bytes_since_decode < LIVE_MIN_DECODE_BYTES
            && last_decode_attempt.elapsed() < LIVE_DECODE_INTERVAL
        {
            continue;
        }

        last_decode_attempt = Instant::now();
        bytes_since_decode = 0;

        match decode_incremental_live_snapshot(&temp_path, sample_rate as u32, last_decoded_samples)
            .await
        {
            Ok(Some((metadata, new_samples, total_samples))) => {
                saw_successful_decode = true;
                last_decoded_samples = total_samples;
                last_decode_error = None;
                process_live_native_samples(
                    &mut pending_samples,
                    vad_engine.as_mut(),
                    &new_samples,
                    live_chunk_samples,
                    &mut transcribe_segment,
                    &mut on_chunk,
                    false,
                )?;

                on_status(format_live_status(
                    metadata.as_ref(),
                    total_samples,
                    sample_rate,
                ));
            }
            Ok(None) => {}
            Err(error) => {
                last_decode_error = Some(error);
            }
        }
    }

    match decode_incremental_live_snapshot(&temp_path, sample_rate as u32, last_decoded_samples)
        .await
    {
        Ok(Some((_metadata, new_samples, _total_samples))) => {
            saw_successful_decode = true;
            process_live_native_samples(
                &mut pending_samples,
                vad_engine.as_mut(),
                &new_samples,
                live_chunk_samples,
                &mut transcribe_segment,
                &mut on_chunk,
                false,
            )?;
        }
        Ok(None) => {}
        Err(error) => {
            last_decode_error = Some(error);
        }
    }

    process_live_native_samples(
        &mut pending_samples,
        vad_engine.as_mut(),
        &[],
        live_chunk_samples,
        &mut transcribe_segment,
        &mut on_chunk,
        true,
    )?;

    if !saw_successful_decode {
        if let Some(error) = last_decode_error {
            return Err(error).context("failed to decode buffered live media stream");
        }
        bail!("failed to decode buffered live media stream");
    }

    Ok(())
}

async fn run_raw_live_transcription_loop<F, C, S>(
    source_url: &str,
    model_dir: &Path,
    execution: &ExecutionMode,
    backend_name: &str,
    sample_rate: usize,
    response: reqwest::Response,
    mut vad_engine: Option<&mut DynamicChunkEngine>,
    mut transcribe_segment: F,
    mut on_chunk: C,
    mut on_status: S,
) -> Result<()>
where
    F: FnMut(&[f32]) -> Result<String>,
    C: FnMut(&str) -> Result<()>,
    S: FnMut(String),
{
    let mut stream = response.bytes_stream();
    let mut raw_decoder = RawLiveAudioDecoder::new(48_000, sample_rate as u32);
    let mut total_output_samples = 0usize;
    let mut pending_samples = Vec::new();
    let live_chunk_samples = sample_rate * LIVE_NATIVE_CHUNK_SECONDS;

    print_live_raw_audio_parameters(
        source_url,
        model_dir,
        execution.device_label(),
        execution.compute_type_label(),
        execution.gpu_device(),
        backend_name,
        48_000,
        sample_rate as u32,
    );

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed to read bytes from raw live stream")?;
        if chunk.is_empty() {
            continue;
        }

        let decoded_samples = raw_decoder.push_bytes(chunk.as_ref());
        if decoded_samples.is_empty() {
            continue;
        }

        total_output_samples += decoded_samples.len();
        process_live_native_samples(
            &mut pending_samples,
            vad_engine.as_deref_mut(),
            &decoded_samples,
            live_chunk_samples,
            &mut transcribe_segment,
            &mut on_chunk,
            false,
        )?;
        on_status(format!(
            "live / {:.1}s buffered / raw f32le",
            total_output_samples as f64 / sample_rate as f64
        ));
    }

    let tail_samples = raw_decoder.finish();
    if !tail_samples.is_empty() {
        process_live_native_samples(
            &mut pending_samples,
            vad_engine.as_deref_mut(),
            &tail_samples,
            live_chunk_samples,
            &mut transcribe_segment,
            &mut on_chunk,
            false,
        )?;
    }

    process_live_native_samples(
        &mut pending_samples,
        vad_engine.as_deref_mut(),
        &[],
        live_chunk_samples,
        &mut transcribe_segment,
        &mut on_chunk,
        true,
    )?;
    Ok(())
}

async fn decode_incremental_live_snapshot(
    snapshot_path: &Path,
    target_sample_rate: u32,
    last_decoded_samples: usize,
) -> Result<Option<(Option<AudioMetadata>, Vec<f32>, usize)>> {
    let snapshot_path = snapshot_path.to_path_buf();
    let inspect_result =
        task::spawn_blocking(move || inspect_audio_file(&snapshot_path, target_sample_rate))
            .await
            .context("failed to join live media decode task")?;

    let (metadata, mut samples) = inspect_result?;
    if samples.len() <= last_decoded_samples {
        return Ok(None);
    }

    let total_samples = samples.len();
    let new_samples = if last_decoded_samples == 0 {
        samples
    } else {
        samples.split_off(last_decoded_samples)
    };
    Ok(Some((Some(metadata), new_samples, total_samples)))
}

fn live_stream_looks_like_raw_pcm(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.starts_with("application/octet-stream"))
        .unwrap_or(false)
}

fn live_stream_looks_like_html(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.starts_with("text/html"))
        .unwrap_or(false)
}

fn process_live_native_samples<F, C>(
    pending_samples: &mut Vec<f32>,
    mut vad_engine: Option<&mut DynamicChunkEngine>,
    new_samples: &[f32],
    live_chunk_samples: usize,
    transcribe_segment: &mut F,
    on_chunk: &mut C,
    finalize: bool,
) -> Result<()>
where
    F: FnMut(&[f32]) -> Result<String>,
    C: FnMut(&str) -> Result<()>,
{
    if let Some(vad_engine) = vad_engine.as_deref_mut() {
        if !new_samples.is_empty() {
            for chunk in vad_engine.push_audio(new_samples) {
                transcribe_live_native_chunk(&chunk.samples, transcribe_segment, on_chunk)?;
            }
        }

        if finalize {
            for chunk in vad_engine.finish_audio() {
                transcribe_live_native_chunk(&chunk.samples, transcribe_segment, on_chunk)?;
            }
        }

        return Ok(());
    }

    if !new_samples.is_empty() {
        pending_samples.extend_from_slice(new_samples);
    }

    while pending_samples.len() >= live_chunk_samples {
        let tail = pending_samples.split_off(live_chunk_samples);
        let chunk_samples = std::mem::replace(pending_samples, tail);
        transcribe_live_native_chunk(&chunk_samples, transcribe_segment, on_chunk)?;
    }

    if finalize && !pending_samples.is_empty() {
        let chunk_samples = std::mem::take(pending_samples);
        transcribe_live_native_chunk(&chunk_samples, transcribe_segment, on_chunk)?;
    }

    Ok(())
}

fn build_live_vad_engine(use_vad: bool, sample_rate: usize) -> Result<Option<DynamicChunkEngine>> {
    if !use_vad {
        return Ok(None);
    }

    let model_window_samples = sample_rate * LIVE_NATIVE_CHUNK_SECONDS;
    let config = DynamicChunkConfig::for_live_stream(sample_rate, model_window_samples)?;
    Ok(Some(DynamicChunkEngine::new(config)))
}

fn transcribe_live_native_chunk<F, C>(
    chunk_samples: &[f32],
    transcribe_segment: &mut F,
    on_chunk: &mut C,
) -> Result<()>
where
    F: FnMut(&[f32]) -> Result<String>,
    C: FnMut(&str) -> Result<()>,
{
    if chunk_samples.is_empty() {
        return Ok(());
    }

    let text = transcribe_segment(chunk_samples)?;
    let text = text.trim();
    if text.is_empty() {
        return Ok(());
    }

    on_chunk(text)?;
    Ok(())
}

fn make_processing_spinner(message: &str) -> Result<ProgressBar> {
    println!("{}", "=".repeat(72));
    let processing = ProgressBar::new_spinner();
    processing.set_style(
        ProgressStyle::with_template("  processing        {spinner:.green} {msg}")
            .context("failed to configure processing spinner")?,
    );
    processing.enable_steady_tick(Duration::from_millis(80));
    processing.set_message(message.to_string());
    Ok(processing)
}

fn print_live_audio_parameters(
    source_url: &str,
    model_dir: &Path,
    device_label: &str,
    compute_type_label: &str,
    gpu_device: Option<i32>,
    backend_name: &str,
    target_sample_rate: u32,
) {
    println!();
    println!("Audio parameters");
    println!("backend: {backend_name} / {device_label} ({compute_type_label})");
    if let Some(gpu_device) = gpu_device {
        println!("gpu id : {gpu_device}");
    }
    println!("source : {source_url}");
    println!("length : live");
    println!("model rate : {target_sample_rate} Hz");
    println!("model path : {}", model_dir.display());
    println!();
}

fn print_live_raw_audio_parameters(
    source_url: &str,
    model_dir: &Path,
    device_label: &str,
    compute_type_label: &str,
    gpu_device: Option<i32>,
    backend_name: &str,
    source_sample_rate: u32,
    target_sample_rate: u32,
) {
    println!();
    println!("Audio parameters");
    println!("backend: {backend_name} / {device_label} ({compute_type_label})");
    if let Some(gpu_device) = gpu_device {
        println!("gpu id : {gpu_device}");
    }
    println!("source : {source_url}");
    println!("length : live");
    println!("input rate : {source_sample_rate} Hz");
    println!("model rate : {target_sample_rate} Hz");
    println!("channels   : 1 -> mono");
    println!("codec      : raw F32Le");
    println!("model path : {}", model_dir.display());
    println!();
}

fn print_audio_parameters(
    audio: &PreparedAudio,
    model_dir: &Path,
    device_label: &str,
    compute_type_label: &str,
    gpu_device: Option<i32>,
    backend_name: &str,
) {
    println!();
    println!("Audio parameters");
    println!("backend: {backend_name} / {device_label} ({compute_type_label})");
    if let Some(gpu_device) = gpu_device {
        println!("gpu id : {gpu_device}");
    }
    println!("source : {}", audio.display_name);
    if let Some(duration) = audio.metadata.duration {
        println!("length : {}", format_duration(duration.as_secs_f64()));
    }
    if let Some(source_rate) = audio.metadata.source_sample_rate {
        println!("input rate : {source_rate} Hz");
    }
    println!("model rate : {} Hz", audio.metadata.target_sample_rate);
    if let Some(channels) = audio.metadata.channels {
        println!("channels   : {channels} -> mono");
    }
    println!("codec      : {}", audio.metadata.codec);
    println!("model path : {}", model_dir.display());
    println!();
}

fn print_transcript(transcript: &str) {
    print_transcript_header();
    if !transcript.is_empty() {
        println!("{transcript}");
    }
}

fn print_transcript_header() {
    println!("{}", "=".repeat(72));
    println!();
}

fn format_duration(seconds: f64) -> String {
    let total_millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let total_seconds = total_millis / 1000;
    let millis = total_millis % 1000;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    format!("{minutes:02}:{seconds:02}.{millis:03}")
}

fn format_live_status(
    metadata: Option<&AudioMetadata>,
    total_samples: usize,
    sample_rate: usize,
) -> String {
    let seconds = if sample_rate == 0 {
        0.0
    } else {
        total_samples as f64 / sample_rate as f64
    };

    match metadata {
        Some(metadata) => {
            let source_rate = metadata
                .source_sample_rate
                .map(|rate| format!("{rate} Hz"))
                .unwrap_or_else(|| String::from("unknown rate"));
            let channels = metadata
                .channels
                .map(|channels| channels.to_string())
                .unwrap_or_else(|| String::from("?"));
            format!(
                "live / {:.1}s buffered / {} / {} ch",
                seconds, source_rate, channels
            )
        }
        None => format!("live / {:.1}s buffered", seconds),
    }
}

#[derive(Debug)]
struct RawLiveAudioDecoder {
    leftover_bytes: Vec<u8>,
    pending_source_samples: Vec<f32>,
    source_sample_rate: u32,
    target_sample_rate: u32,
}

impl RawLiveAudioDecoder {
    fn new(source_sample_rate: u32, target_sample_rate: u32) -> Self {
        Self {
            leftover_bytes: Vec::new(),
            pending_source_samples: Vec::new(),
            source_sample_rate,
            target_sample_rate,
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<f32> {
        if bytes.is_empty() {
            return Vec::new();
        }

        self.leftover_bytes.extend_from_slice(bytes);
        let usable_len = self.leftover_bytes.len() - (self.leftover_bytes.len() % 4);
        if usable_len == 0 {
            return Vec::new();
        }

        for chunk in self.leftover_bytes[..usable_len].chunks_exact(4) {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.pending_source_samples.push(sample.clamp(-1.0, 1.0));
        }
        self.leftover_bytes.drain(..usable_len);

        self.take_resampled_output(false)
    }

    fn finish(&mut self) -> Vec<f32> {
        self.leftover_bytes.clear();
        self.take_resampled_output(true)
    }

    fn take_resampled_output(&mut self, finalize: bool) -> Vec<f32> {
        if self.pending_source_samples.is_empty() {
            return Vec::new();
        }

        if self.source_sample_rate == self.target_sample_rate {
            return std::mem::take(&mut self.pending_source_samples);
        }

        if self.source_sample_rate == 48_000 && self.target_sample_rate == 16_000 {
            let usable = if finalize {
                self.pending_source_samples.len() / 3 * 3
            } else {
                (self.pending_source_samples.len() / 3 * 3).saturating_sub(3)
            };
            if usable < 3 {
                return Vec::new();
            }

            let mut output = Vec::with_capacity(usable / 3);
            for trio in self.pending_source_samples[..usable].chunks_exact(3) {
                output.push(((trio[0] + trio[1] + trio[2]) / 3.0).clamp(-1.0, 1.0));
            }
            self.pending_source_samples.drain(..usable);
            return output;
        }

        let output = crate::audio::linear_resample(
            &self.pending_source_samples,
            self.source_sample_rate,
            self.target_sample_rate,
        );
        self.pending_source_samples.clear();
        output
    }
}

#[derive(Default)]
struct StreamOutputState {
    printed_any: bool,
}

fn stream_print_chunk(
    processing: &ProgressBar,
    output_state: &mut StreamOutputState,
    chunk_text: &str,
) -> Result<()> {
    processing.suspend(|| -> Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{chunk_text}").context("failed to write stream chunk")?;
        stdout.flush().context("failed to flush stream output")?;
        Ok(())
    })?;
    output_state.printed_any = true;
    Ok(())
}

fn finish_stream_output(output_state: &StreamOutputState) {
    if output_state.printed_any {
        println!();
    }
}
