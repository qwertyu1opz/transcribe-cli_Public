use std::convert::Infallible;
use std::path::PathBuf;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::UnboundedReceiverStream};

use crate::model::{ModelChoice, ModelComputeType};
use crate::model_download::ensure_model_downloaded;
#[cfg(target_os = "linux")]
use crate::model_download::ensure_ort_runtime_downloaded;
use crate::transcribe::{
    RestTranscriptionResult, stream_live_gigaam_transcription, stream_live_parakeet_transcription,
    stream_media_for_rest, transcribe_media_for_rest,
};

#[derive(Clone)]
pub struct RestState {
    pub models_root: PathBuf,
}

pub fn build_router(state: RestState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/transcribe", post(transcribe))
        .route("/v1/transcribe/stream", post(transcribe_stream))
        .route("/v1/live/transcribe", post(live_transcribe))
        .with_state(state)
}

async fn root() -> Json<ApiInfo> {
    Json(ApiInfo {
        service: "transcribe-cli",
        version: env!("CARGO_PKG_VERSION"),
        endpoints: vec![
            String::from("GET /health"),
            String::from("POST /v1/transcribe"),
            String::from("POST /v1/transcribe/stream"),
            String::from("POST /v1/live/transcribe"),
        ],
    })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "transcribe-cli",
    })
}

async fn transcribe(
    State(state): State<RestState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let language = normalize_language(request.language.as_deref())
        .map_err(|error| bad_request(error.to_string()))?;
    let execution = request
        .execution()
        .map_err(|error| bad_request(error.to_string()))?;
    let model_choice = select_model_for_language(&language);

    let result = transcribe_media_for_rest(
        &request.media,
        &language,
        model_choice,
        &state.models_root,
        &execution,
    )
    .await
    .map_err(|error| internal_error(format!("{error:#}")))?;

    Ok(Json(TranscribeResponse::from_result(
        request.media,
        language,
        model_choice,
        &execution,
        result,
    )))
}

async fn transcribe_stream(
    State(state): State<RestState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<
    Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>,
    (StatusCode, Json<ErrorResponse>),
> {
    let language = normalize_language(request.language.as_deref())
        .map_err(|error| bad_request(error.to_string()))?;
    let execution = request
        .execution()
        .map_err(|error| bad_request(error.to_string()))?;
    let model_choice = select_model_for_language(&language);
    let models_root = state.models_root.clone();
    let media = request.media.clone();

    let (sender, receiver) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

    tokio::spawn(async move {
        let _ = sender.send(Ok(
            Event::default()
                .event("ready")
                .data(format!("starting stream transcription for {}", media)),
        ));

        let result: anyhow::Result<()> = stream_media_for_rest(
            &media,
            &language,
            model_choice,
            &models_root,
            &execution,
            {
                let sender = sender.clone();
                move |chunk| {
                    sender
                        .send(Ok(Event::default().event("chunk").data(chunk.to_string())))
                        .map_err(|error| anyhow::anyhow!("failed to stream transcription chunk: {error}"))
                }
            },
            {
                let sender = sender.clone();
                move |status| {
                    let _ = sender.send(Ok(Event::default().event("status").data(status)));
                }
            },
        )
        .await;

        match result {
            Ok(()) => {
                let _ = sender.send(Ok(Event::default().event("done").data("complete")));
            }
            Err(error) => {
                let _ = sender.send(Ok(Event::default()
                    .event("error")
                    .data(format!("{error:#}").replace('\n', " "))));
            }
        }
    });

    let stream = UnboundedReceiverStream::new(receiver).map(|event| event);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn live_transcribe(
    State(state): State<RestState>,
    Json(request): Json<LiveTranscribeRequest>,
) -> Result<
    Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>,
    (StatusCode, Json<ErrorResponse>),
> {
    let language = normalize_language(request.language.as_deref())
        .map_err(|error| bad_request(error.to_string()))?;
    let execution = request
        .execution()
        .map_err(|error| bad_request(error.to_string()))?;
    let model_choice = select_model_for_language(&language);
    let models_root = state.models_root.clone();
    let media = request.media.clone();
    let use_vad = request.vad.unwrap_or(false);

    let (sender, receiver) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

    tokio::spawn(async move {
        let _ = sender.send(Ok(Event::default()
            .event("ready")
            .data(format!("starting live transcription for {}", media))));

        let result: anyhow::Result<()> = async {
            #[cfg(target_os = "linux")]
            ensure_ort_runtime_downloaded().await?;

            let model_dir = ensure_model_downloaded(
                model_choice,
                Some(execution.compute_type()),
                Some(&models_root),
            )
            .await?;

            let chunk_sender = sender.clone();
            match model_choice {
                ModelChoice::GigaamV3 => {
                    stream_live_gigaam_transcription(
                        &media,
                        model_choice,
                        &model_dir,
                        &execution,
                        Some(&language),
                        use_vad,
                        move |chunk| {
                            chunk_sender
                                .send(Ok(Event::default().event("chunk").data(chunk.to_string())))
                                .map_err(|error| {
                                    anyhow::anyhow!("failed to stream live chunk: {error}")
                                })
                        },
                        {
                            let sender = sender.clone();
                            move |status| {
                                let _ =
                                    sender.send(Ok(Event::default().event("status").data(status)));
                            }
                        },
                    )
                    .await?
                }
                ModelChoice::ParakeetTdt06bV2 => {
                    stream_live_parakeet_transcription(
                        &media,
                        model_choice,
                        &model_dir,
                        &execution,
                        Some(&language),
                        use_vad,
                        move |chunk| {
                            chunk_sender
                                .send(Ok(Event::default().event("chunk").data(chunk.to_string())))
                                .map_err(|error| {
                                    anyhow::anyhow!("failed to stream live chunk: {error}")
                                })
                        },
                        {
                            let sender = sender.clone();
                            move |status| {
                                let _ =
                                    sender.send(Ok(Event::default().event("status").data(status)));
                            }
                        },
                    )
                    .await?
                }
            }

            Ok(())
        }
        .await;

        match result {
            Ok(()) => {
                let _ = sender.send(Ok(Event::default().event("done").data("complete")));
            }
            Err(error) => {
                let _ = sender.send(Ok(Event::default()
                    .event("error")
                    .data(format!("{error:#}").replace('\n', " "))));
            }
        }
    });

    let stream = UnboundedReceiverStream::new(receiver).map(|event| event);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Serialize)]
struct ApiInfo {
    service: &'static str,
    version: &'static str,
    endpoints: Vec<String>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

#[derive(Debug, Deserialize)]
struct TranscribeRequest {
    media: String,
    language: Option<String>,
    gpu: Option<bool>,
    gpu_device: Option<i32>,
    compute_type: Option<RestComputeType>,
}

impl TranscribeRequest {
    fn execution(&self) -> anyhow::Result<crate::onnx_ctc::ExecutionMode> {
        crate::onnx_ctc::ExecutionMode::from_cli(
            self.gpu.unwrap_or(false),
            self.gpu_device.unwrap_or(0),
            self.compute_type
                .and_then(RestComputeType::model_compute_type),
        )
    }
}

#[derive(Debug, Deserialize)]
struct LiveTranscribeRequest {
    media: String,
    language: Option<String>,
    gpu: Option<bool>,
    gpu_device: Option<i32>,
    compute_type: Option<RestComputeType>,
    vad: Option<bool>,
}

impl LiveTranscribeRequest {
    fn execution(&self) -> anyhow::Result<crate::onnx_ctc::ExecutionMode> {
        crate::onnx_ctc::ExecutionMode::from_cli(
            self.gpu.unwrap_or(false),
            self.gpu_device.unwrap_or(0),
            self.compute_type
                .and_then(RestComputeType::model_compute_type),
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RestComputeType {
    Auto,
    Int8,
    Float32,
}

impl RestComputeType {
    fn model_compute_type(self) -> Option<ModelComputeType> {
        match self {
            Self::Auto => None,
            Self::Int8 => Some(ModelComputeType::Int8),
            Self::Float32 => Some(ModelComputeType::Float32),
        }
    }
}

#[derive(Serialize)]
struct TranscribeResponse {
    transcript: String,
    source: String,
    language: String,
    requested_model: &'static str,
    runtime_model: &'static str,
    device: &'static str,
    compute_type: &'static str,
    target_sample_rate: u32,
    source_sample_rate: Option<u32>,
    channels: Option<u16>,
    codec: String,
    duration_seconds: Option<f64>,
}

impl TranscribeResponse {
    fn from_result(
        source: String,
        language: String,
        model_choice: ModelChoice,
        execution: &crate::onnx_ctc::ExecutionMode,
        result: RestTranscriptionResult,
    ) -> Self {
        Self {
            transcript: result.transcript,
            source,
            language,
            requested_model: model_choice.cli_name(),
            runtime_model: model_choice.runtime_name(),
            device: execution.device_label(),
            compute_type: execution.compute_type_label(),
            target_sample_rate: result.audio_metadata.target_sample_rate,
            source_sample_rate: result.audio_metadata.source_sample_rate,
            channels: result.audio_metadata.channels,
            codec: result.audio_metadata.codec,
            duration_seconds: result
                .audio_metadata
                .duration
                .map(|duration| duration.as_secs_f64()),
        }
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn normalize_language(language: Option<&str>) -> anyhow::Result<String> {
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

fn bad_request(error: String) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }))
}

fn internal_error(error: String) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error }),
    )
}
