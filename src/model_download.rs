use std::collections::BTreeSet;
#[cfg(target_os = "linux")]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use reqwest::header::RANGE;
use tokio::fs;
use tokio::io::AsyncWriteExt;
#[cfg(target_os = "linux")]
use zip::ZipArchive;

#[cfg(target_os = "linux")]
use crate::model::default_ort_runtime_root_directory;
use crate::model::{HfModelResponse, HfSibling, ModelChoice, ModelComputeType, model_directory};

const CURL_DOWNLOAD_THRESHOLD: u64 = 8 * 1024 * 1024;
#[cfg(target_os = "linux")]
const ORT_CUDA13_RUNTIME_VERSION: &str = "1.24.0.dev20251108004";
#[cfg(target_os = "linux")]
const ORT_CUDA13_RUNTIME_WHEEL_URL: &str = "https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/22685817-9e91-4967-bafe-94c9843c26a9/pypi/download/onnxruntime-gpu/1.24.dev20251108004/onnxruntime_gpu-1.24.0.dev20251108004-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
#[cfg(target_os = "linux")]
const ORT_REQUIRED_LIBRARIES: &[&str] = &[
    "libonnxruntime.so.1.24.0",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
];

pub async fn ensure_model_downloaded(
    choice: ModelChoice,
    compute_type: Option<ModelComputeType>,
    models_root: Option<&Path>,
) -> Result<PathBuf> {
    let target_dir = model_directory(choice, models_root)?;
    fs::create_dir_all(&target_dir)
        .await
        .with_context(|| format!("failed to create `{}`", target_dir.display()))?;

    let manifest_bar = ProgressBar::new_spinner();
    manifest_bar.set_style(
        ProgressStyle::with_template("  resolving model   {spinner:.green} {msg}")
            .context("failed to configure manifest spinner")?,
    );
    manifest_bar.enable_steady_tick(std::time::Duration::from_millis(80));
    manifest_bar.set_message(choice.repo_id().to_string());

    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .user_agent("transcribe-cli/0.1.0")
        .build()
        .context("failed to build HTTP client")?;
    let manifest = fetch_manifest(&client, choice).await?;
    let files = resolve_model_files(choice, compute_type, manifest)?;
    manifest_bar.finish_with_message(format!("manifest ready [{} files]", files.len()));

    let total_size = files
        .iter()
        .filter_map(|file| file.expected_size)
        .sum::<u64>();
    let download_bar = ProgressBar::new(total_size.max(1));
    download_bar.set_style(
        ProgressStyle::with_template(
            "  downloading model [{bar:40.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta} {msg}",
        )
        .context("failed to configure download progress bar")?
        .progress_chars("=> "),
    );

    for file in files {
        let destination = target_dir.join(&file.relative_path);

        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }

        let existing_size = existing_file_size(&destination).await?;
        if file_is_complete(existing_size, file.expected_size) {
            if let Some(expected_size) = file.expected_size {
                download_bar.inc(expected_size);
            } else if existing_size > 0 {
                download_bar.inc(existing_size);
            }
            continue;
        }

        if existing_size > 0 {
            download_bar.inc(existing_size);
            download_bar.set_message(format!(
                "{} (resuming from {})",
                file.relative_path,
                indicatif::HumanBytes(existing_size)
            ));
        } else {
            download_bar.set_message(file.relative_path.clone());
        }

        download_file(
            &client,
            &file.download_url,
            &destination,
            &download_bar,
            file.expected_size,
            existing_size,
        )
        .await?;
    }

    let message = if let Some(compute_type) = compute_type {
        format!(
            "  downloaded model [{} / {}]",
            choice.runtime_name(),
            compute_type.label()
        )
    } else {
        format!("  downloaded model [{}]", choice.runtime_name())
    };
    download_bar.finish_with_message(message);
    Ok(target_dir)
}

#[cfg(target_os = "linux")]
pub async fn ensure_ort_runtime_downloaded() -> Result<PathBuf> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some()
        || std::env::var_os("TRANSCRIBE_ORT_ROOT").is_some()
    {
        return Ok(default_ort_runtime_root_directory()?);
    }

    let runtime_root = default_ort_runtime_root_directory()?;
    let version_root = runtime_root.join(ORT_CUDA13_RUNTIME_VERSION);
    let capi_dir = version_root.join("onnxruntime/capi");

    if ort_runtime_ready(&capi_dir) {
        refresh_current_runtime_symlink(&runtime_root, &version_root).await?;
        return Ok(runtime_root);
    }

    fs::create_dir_all(&runtime_root)
        .await
        .with_context(|| format!("failed to create `{}`", runtime_root.display()))?;

    let wheel_path = runtime_root.join(format!("onnxruntime_gpu-{ORT_CUDA13_RUNTIME_VERSION}.whl"));
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .user_agent("transcribe-cli/0.1.0")
        .build()
        .context("failed to build HTTP client for ONNX Runtime runtime download")?;

    let staging_root = runtime_root.join(format!(".{}.extract", ORT_CUDA13_RUNTIME_VERSION));
    let mut last_error = None;
    for attempt in 0..2 {
        ensure_ort_runtime_wheel_downloaded(&client, &wheel_path).await?;
        if staging_root.exists() {
            fs::remove_dir_all(&staging_root)
                .await
                .with_context(|| format!("failed to remove `{}`", staging_root.display()))?;
        }

        let extraction_result = tokio::task::spawn_blocking({
            let wheel_path = wheel_path.clone();
            let staging_root = staging_root.clone();
            move || extract_ort_runtime_wheel(&wheel_path, &staging_root)
        })
        .await
        .context("failed to join ORT runtime extraction task")?;

        match extraction_result {
            Ok(()) => {
                last_error = None;
                break;
            }
            Err(error) if attempt == 0 => {
                last_error = Some(error);
                if wheel_path.exists() {
                    fs::remove_file(&wheel_path).await.with_context(|| {
                        format!(
                            "failed to remove corrupt ORT runtime wheel `{}`",
                            wheel_path.display()
                        )
                    })?;
                }
                if staging_root.exists() {
                    fs::remove_dir_all(&staging_root).await.with_context(|| {
                        format!("failed to remove `{}`", staging_root.display())
                    })?;
                }
            }
            Err(error) => {
                last_error = Some(error);
                break;
            }
        }
    }

    if let Some(error) = last_error {
        return Err(error).context(format!(
            "failed to prepare ONNX Runtime CUDA runtime in `{}`",
            runtime_root.display()
        ));
    }

    if version_root.exists() {
        fs::remove_dir_all(&version_root)
            .await
            .with_context(|| format!("failed to replace `{}`", version_root.display()))?;
    }

    fs::rename(&staging_root, &version_root)
        .await
        .with_context(|| {
            format!(
                "failed to move extracted ORT runtime into `{}`",
                version_root.display()
            )
        })?;

    refresh_current_runtime_symlink(&runtime_root, &version_root).await?;
    Ok(runtime_root)
}

#[cfg(target_os = "linux")]
async fn ensure_ort_runtime_wheel_downloaded(client: &Client, wheel_path: &Path) -> Result<()> {
    let wheel_size = existing_file_size(wheel_path).await?;
    if wheel_size > 0 {
        return Ok(());
    }

    let download_bar = ProgressBar::new_spinner();
    download_bar.set_style(
        ProgressStyle::with_template("  downloading ort   {spinner:.green} {msg}")
            .context("failed to configure ORT runtime spinner")?,
    );
    download_bar.enable_steady_tick(std::time::Duration::from_millis(80));
    download_bar.set_message(format!("CUDA 13 runtime {ORT_CUDA13_RUNTIME_VERSION}"));
    download_file(
        client,
        ORT_CUDA13_RUNTIME_WHEEL_URL,
        wheel_path,
        &download_bar,
        None,
        0,
    )
    .await?;
    download_bar.finish_with_message(format!(
        "ort runtime ready [{}]",
        ORT_CUDA13_RUNTIME_VERSION
    ));
    Ok(())
}

#[derive(Debug)]
struct ResolvedModelFile {
    relative_path: String,
    download_url: String,
    expected_size: Option<u64>,
}

async fn fetch_manifest(client: &Client, choice: ModelChoice) -> Result<Vec<HfSibling>> {
    let url = format!("https://huggingface.co/api/models/{}", choice.repo_id());
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to fetch model metadata for `{}`", choice.repo_id()))?
        .error_for_status()
        .with_context(|| format!("model metadata request failed for `{}`", choice.repo_id()))?;

    let parsed: HfModelResponse = response
        .json()
        .await
        .with_context(|| format!("failed to decode manifest for `{}`", choice.repo_id()))?;

    Ok(parsed
        .siblings
        .into_iter()
        .filter(|file| !file.rfilename.ends_with('/'))
        .collect())
}

fn resolve_model_files(
    choice: ModelChoice,
    compute_type: Option<ModelComputeType>,
    manifest: Vec<HfSibling>,
) -> Result<Vec<ResolvedModelFile>> {
    let compute_type = compute_type.unwrap_or(ModelComputeType::Int8);
    let mut required_files: BTreeSet<String> = [
        choice.config_file(),
        choice.vocab_file(),
        choice.onnx_file(compute_type),
    ]
    .into_iter()
    .map(ToOwned::to_owned)
    .collect();
    required_files.extend(
        choice
            .extra_required_files(compute_type)
            .iter()
            .copied()
            .map(ToOwned::to_owned),
    );

    resolve_manifest_sizes(choice, manifest, &required_files)
}

fn resolve_manifest_sizes(
    choice: ModelChoice,
    manifest: Vec<HfSibling>,
    required_files: &BTreeSet<String>,
) -> Result<Vec<ResolvedModelFile>> {
    let mut files = Vec::new();

    for file in manifest {
        if !required_files.contains(&file.rfilename) {
            continue;
        }

        files.push(ResolvedModelFile {
            relative_path: file.rfilename.clone(),
            download_url: file_download_url(choice, &file.rfilename),
            expected_size: file.expected_size(),
        });
    }

    let missing = required_files
        .iter()
        .filter(|required| !files.iter().any(|file| &file.relative_path == *required))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        anyhow::bail!(
            "model repository `{}` is missing required file(s): {}",
            choice.repo_id(),
            missing.join(", ")
        );
    }

    Ok(files)
}

async fn existing_file_size(path: &Path) -> Result<u64> {
    let Ok(metadata) = fs::metadata(path).await else {
        return Ok(0);
    };

    Ok(metadata.len())
}

fn file_is_complete(existing_size: u64, expected_size: Option<u64>) -> bool {
    match expected_size {
        Some(expected_size) => existing_size == expected_size,
        None => existing_size > 0,
    }
}

fn file_download_url(choice: ModelChoice, relative_path: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        choice.repo_id(),
        relative_path
    )
}

async fn download_file(
    client: &Client,
    url: &str,
    destination: &Path,
    progress_bar: &ProgressBar,
    expected_size: Option<u64>,
    existing_size: u64,
) -> Result<()> {
    if expected_size.unwrap_or_default() >= CURL_DOWNLOAD_THRESHOLD {
        match download_with_curl(url, destination, progress_bar, existing_size).await {
            Ok(()) => return Ok(()),
            Err(error) => {
                progress_bar.set_message(format!("curl fallback: {error:#}"));
            }
        }
    }

    let mut request = client.get(url);
    if existing_size > 0 {
        request = request.header(RANGE, format!("bytes={existing_size}-"));
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("failed to download `{url}`"))?;

    let status = response.status();
    if existing_size > 0 {
        if status != reqwest::StatusCode::PARTIAL_CONTENT {
            let response = client
                .get(url)
                .send()
                .await
                .with_context(|| format!("failed to restart download for `{url}`"))?;
            return stream_response_to_file(response, destination, progress_bar, 0).await;
        }
    } else if !status.is_success() {
        let error = response
            .error_for_status()
            .expect_err("non-success status should return an error");
        return Err(error).with_context(|| format!("download failed for `{url}`"));
    }

    stream_response_to_file(response, destination, progress_bar, existing_size).await?;

    if let Some(expected_size) = expected_size {
        let actual_size = existing_file_size(destination).await?;
        if actual_size != expected_size {
            anyhow::bail!(
                "downloaded `{}` but received {} bytes instead of {}",
                destination.display(),
                actual_size,
                expected_size
            );
        }
    }

    Ok(())
}

async fn stream_response_to_file(
    response: reqwest::Response,
    destination: &Path,
    progress_bar: &ProgressBar,
    existing_size: u64,
) -> Result<()> {
    let mut file = if existing_size > 0 {
        fs::OpenOptions::new()
            .append(true)
            .open(destination)
            .await
            .with_context(|| format!("failed to append to `{}`", destination.display()))?
    } else {
        fs::File::create(destination)
            .await
            .with_context(|| format!("failed to create `{}`", destination.display()))?
    };

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("failed to read `{}`", destination.display()))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("failed to write `{}`", destination.display()))?;
        progress_bar.inc(chunk.len() as u64);
    }

    file.flush()
        .await
        .with_context(|| format!("failed to flush `{}`", destination.display()))?;
    Ok(())
}

async fn download_with_curl(
    url: &str,
    destination: &Path,
    progress_bar: &ProgressBar,
    existing_size: u64,
) -> Result<()> {
    let mut command = Command::new("curl");
    command
        .arg("--fail")
        .arg("--location")
        .arg("--silent")
        .arg("--show-error")
        .arg("--output")
        .arg(destination)
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if existing_size > 0 {
        command.arg("--continue-at").arg(existing_size.to_string());
    }

    let output = command
        .output()
        .with_context(|| format!("failed to start curl for `{}`", destination.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "curl failed for `{}`: {}",
            destination.display(),
            stderr.trim()
        );
    }

    let new_size = existing_file_size(destination).await?;
    let downloaded = new_size.saturating_sub(existing_size);
    if downloaded > 0 {
        progress_bar.inc(downloaded);
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn extract_ort_runtime_wheel(wheel_path: &Path, target_root: &Path) -> Result<()> {
    if target_root.exists() {
        std::fs::remove_dir_all(target_root)
            .with_context(|| format!("failed to clear `{}`", target_root.display()))?;
    }
    std::fs::create_dir_all(target_root)
        .with_context(|| format!("failed to create `{}`", target_root.display()))?;

    let wheel = File::open(wheel_path)
        .with_context(|| format!("failed to open `{}`", wheel_path.display()))?;
    let mut archive = ZipArchive::new(wheel)
        .with_context(|| format!("failed to read `{}`", wheel_path.display()))?;

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("failed to access archive entry #{index}"))?;
        let Some(relative_path) = entry.enclosed_name().map(|path| path.to_path_buf()) else {
            continue;
        };

        let relative = relative_path.to_string_lossy();
        if !relative.starts_with("onnxruntime/capi/") {
            continue;
        }

        let Some(file_name) = relative_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !ORT_REQUIRED_LIBRARIES.contains(&file_name) {
            continue;
        }

        let destination = target_root.join(&relative_path);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }

        let mut output = std::fs::File::create(&destination)
            .with_context(|| format!("failed to create `{}`", destination.display()))?;
        std::io::copy(&mut entry, &mut output)
            .with_context(|| format!("failed to extract `{}`", destination.display()))?;
    }

    let capi_dir = target_root.join("onnxruntime/capi");
    ensure_ort_runtime_symlinks(&capi_dir)?;
    if !ort_runtime_ready(&capi_dir) {
        anyhow::bail!(
            "extracted ONNX Runtime wheel `{}` but required CUDA provider libraries were not found",
            wheel_path.display()
        );
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn ensure_ort_runtime_symlinks(capi_dir: &Path) -> Result<()> {
    use std::os::unix::fs::symlink;

    let lib_major = capi_dir.join("libonnxruntime.so.1");
    if lib_major.exists() {
        std::fs::remove_file(&lib_major)
            .with_context(|| format!("failed to replace `{}`", lib_major.display()))?;
    }
    symlink("libonnxruntime.so.1.24.0", &lib_major)
        .with_context(|| format!("failed to create `{}`", lib_major.display()))?;

    let lib_unversioned = capi_dir.join("libonnxruntime.so");
    if lib_unversioned.exists() {
        std::fs::remove_file(&lib_unversioned)
            .with_context(|| format!("failed to replace `{}`", lib_unversioned.display()))?;
    }
    symlink("libonnxruntime.so.1", &lib_unversioned)
        .with_context(|| format!("failed to create `{}`", lib_unversioned.display()))?;

    Ok(())
}

#[cfg(target_os = "linux")]
fn ort_runtime_ready(capi_dir: &Path) -> bool {
    ORT_REQUIRED_LIBRARIES
        .iter()
        .all(|file_name| capi_dir.join(file_name).is_file())
        && capi_dir.join("libonnxruntime.so").exists()
        && capi_dir.join("libonnxruntime.so.1").exists()
}

#[cfg(target_os = "linux")]
async fn refresh_current_runtime_symlink(runtime_root: &Path, version_root: &Path) -> Result<()> {
    use std::os::unix::fs::symlink;

    let current_link = runtime_root.join("current");
    if current_link.exists() || current_link.symlink_metadata().is_ok() {
        fs::remove_file(&current_link)
            .await
            .with_context(|| format!("failed to replace `{}`", current_link.display()))?;
    }

    symlink(
        version_root
            .file_name()
            .context("ORT runtime version path is missing a file name")?,
        &current_link,
    )
    .with_context(|| format!("failed to create `{}`", current_link.display()))?;

    Ok(())
}
