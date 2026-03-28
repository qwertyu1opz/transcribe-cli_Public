use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, CodecParameters, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tempfile::{Builder, NamedTempFile};
use tokio::fs;
use url::Url;

use crate::video::is_video_file;

pub struct PreparedAudio {
    pub display_name: String,
    pub metadata: AudioMetadata,
    pub samples: Vec<f32>,
    _temp_file: Option<NamedTempFile>,
}

#[derive(Debug, Clone)]
pub struct AudioMetadata {
    pub source_sample_rate: Option<u32>,
    pub target_sample_rate: u32,
    pub channels: Option<u16>,
    pub duration: Option<Duration>,
    pub codec: String,
}

pub async fn prepare_audio_source(input: &str) -> Result<PreparedAudio> {
    prepare_audio_source_for_rate(input, 16_000).await
}

pub async fn prepare_audio_source_for_rate(
    input: &str,
    target_sample_rate: u32,
) -> Result<PreparedAudio> {
    if let Ok(url) = Url::parse(input) {
        if matches!(url.scheme(), "http" | "https") {
            return download_remote_audio(url, target_sample_rate).await;
        }
    }

    let path = resolve_local_audio_path(input).await?;
    let (metadata, samples) = inspect_audio_file(&path, target_sample_rate)?;

    Ok(PreparedAudio {
        display_name: path.display().to_string(),
        metadata,
        samples,
        _temp_file: None,
    })
}

async fn resolve_local_audio_path(input: &str) -> Result<std::path::PathBuf> {
    let requested_path = Path::new(input);
    let resolved_path = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current working directory")?
            .join(requested_path)
    };

    fs::canonicalize(&resolved_path).await.with_context(|| {
        format!(
            "failed to resolve audio path `{}` from current directory",
            input
        )
    })
}

async fn download_remote_audio(url: Url, target_sample_rate: u32) -> Result<PreparedAudio> {
    let suffix = url
        .path_segments()
        .and_then(|segments| segments.last())
        .and_then(|name| Path::new(name).extension())
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{ext}"))
        .unwrap_or_else(|| ".audio".to_string());

    let mut temp_file = Builder::new()
        .prefix("transcribe-cli-")
        .suffix(&suffix)
        .tempfile()
        .context("failed to create temporary audio file")?;

    let client = Client::builder()
        .user_agent("transcribe-cli/0.1.0")
        .build()
        .context("failed to build HTTP client")?;
    let response = client
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("failed to download audio from `{url}`"))?
        .error_for_status()
        .with_context(|| format!("audio download returned an error for `{url}`"))?;

    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("failed to read audio body from `{url}`"))?;
    temp_file
        .write_all(bytes.as_ref())
        .context("failed to save downloaded audio")?;

    let local_path = temp_file.path().to_path_buf();
    let (metadata, samples) = inspect_audio_file(&local_path, target_sample_rate)?;

    Ok(PreparedAudio {
        display_name: url.to_string(),
        metadata,
        samples,
        _temp_file: Some(temp_file),
    })
}

pub(crate) fn inspect_audio_file(
    path: &Path,
    target_sample_rate: u32,
) -> Result<(AudioMetadata, Vec<f32>)> {
    let file = File::open(path)
        .with_context(|| format!("failed to open audio file `{}`", path.display()))?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();

    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }

    let probe = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| friendly_media_probe_error(path, &error))?;

    let mut format = probe.format;
    let (mut codec_params, track_id) = select_audio_track(format.as_ref()).with_context(|| {
        if is_video_file(path) {
            format!(
                "failed to find a supported audio track in video input `{}`",
                path.display()
            )
        } else {
            format!(
                "failed to find a supported audio track in `{}`",
                path.display()
            )
        }
    })?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("failed to create audio decoder")?;
    let mut source_sample_rate = codec_params.sample_rate;
    let mut source_channels = codec_params.channels.map(|channels| channels.count());
    let mut mono_samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("audio stream reset is required and is not supported")
            }
            Err(error) => {
                return Err(friendly_media_runtime_error(
                    path,
                    "failed to read audio data",
                    &error,
                ));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => {
                return Err(friendly_media_runtime_error(
                    path,
                    "failed to decode audio data",
                    &error,
                ));
            }
        };

        let spec = *decoded.spec();
        if source_sample_rate.is_none() {
            source_sample_rate = Some(spec.rate);
            codec_params.with_sample_rate(spec.rate);
        }
        if source_channels.is_none() {
            source_channels = Some(spec.channels.count());
            codec_params.with_channels(spec.channels);
        }

        let source_channels =
            source_channels.context("audio stream does not expose channel information")?;
        let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buffer.copy_interleaved_ref(decoded);

        for frame in sample_buffer.samples().chunks(source_channels) {
            let mono = frame.iter().copied().sum::<f32>() / source_channels as f32;
            mono_samples.push(mono.clamp(-1.0, 1.0));
        }
    }

    let source_sample_rate =
        source_sample_rate.context("audio stream does not expose a sample rate")?;
    let duration = codec_params
        .n_frames
        .zip(codec_params.sample_rate)
        .map(|(frames, sample_rate)| Duration::from_secs_f64(frames as f64 / sample_rate as f64));
    let resampled_samples = if source_sample_rate == target_sample_rate {
        mono_samples
    } else {
        linear_resample(&mono_samples, source_sample_rate, target_sample_rate)
    };

    Ok((
        extract_metadata(
            &codec_params,
            duration,
            source_sample_rate,
            target_sample_rate,
        ),
        resampled_samples,
    ))
}

fn friendly_media_probe_error(path: &Path, error: &SymphoniaError) -> anyhow::Error {
    let reason = match error {
        SymphoniaError::IoError(io_error)
            if io_error.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            "input appears incomplete or truncated"
        }
        SymphoniaError::Unsupported(_)
        | SymphoniaError::LimitError(_)
        | SymphoniaError::DecodeError(_) => {
            "unsupported or malformed media container"
        }
        _ => "unsupported or malformed media input",
    };

    anyhow::anyhow!(
        "failed to parse audio input `{}`: {reason}",
        path.display()
    )
}

fn friendly_media_runtime_error(
    path: &Path,
    action: &str,
    error: &SymphoniaError,
) -> anyhow::Error {
    let reason = match error {
        SymphoniaError::IoError(io_error)
            if io_error.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            "input appears incomplete or truncated"
        }
        SymphoniaError::Unsupported(_) | SymphoniaError::LimitError(_) => {
            "media format uses unsupported features"
        }
        SymphoniaError::DecodeError(_) => "audio frames could not be decoded",
        SymphoniaError::ResetRequired => "stream reset is required and is not supported",
        _ => "media input could not be processed",
    };

    anyhow::anyhow!("{action} in `{}`: {reason}", path.display())
}

fn select_audio_track(format: &dyn FormatReader) -> Result<(CodecParameters, u32)> {
    let decoder_options = DecoderOptions::default();

    select_audio_track_with(format, |codec_params| {
        codec_params.codec != CODEC_TYPE_NULL
            && symphonia::default::get_codecs()
                .make(codec_params, &decoder_options)
                .is_ok()
    })
}

fn select_audio_track_with<F>(
    format: &dyn FormatReader,
    is_supported: F,
) -> Result<(CodecParameters, u32)>
where
    F: Fn(&CodecParameters) -> bool,
{
    if let Some(track) = format
        .tracks()
        .iter()
        .find(|track| is_supported(&track.codec_params))
    {
        return Ok((track.codec_params.clone(), track.id));
    }

    bail!("media input does not contain a decodable audio track")
}

fn extract_metadata(
    codec: &CodecParameters,
    duration: Option<Duration>,
    source_sample_rate: u32,
    target_sample_rate: u32,
) -> AudioMetadata {
    AudioMetadata {
        source_sample_rate: Some(source_sample_rate),
        target_sample_rate,
        channels: codec.channels.map(|channels| channels.count() as u16),
        duration,
        codec: format!("{:?}", codec.codec),
    }
}

pub(crate) fn linear_resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }

    let ratio = target_rate as f64 / source_rate as f64;
    let output_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut resampled = Vec::with_capacity(output_len);

    for index in 0..output_len {
        let source_position = index as f64 / ratio;
        let left_index = source_position.floor() as usize;
        let right_index = (left_index + 1).min(samples.len().saturating_sub(1));
        let fraction = (source_position - left_index as f64) as f32;
        let left = samples[left_index];
        let right = samples[right_index];
        resampled.push(left + (right - left) * fraction);
    }

    resampled
}
