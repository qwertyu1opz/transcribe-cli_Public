# transcribe-cli

`transcribe-cli` is a native Rust transcription tool built around ONNX Runtime.



Current backends:

- `ru` -> `GigaAM v3 e2e CTC`
- `en` -> `Parakeet TDT v2`

It supports:

- local audio and video files
- `http/https` media URLs
- chunked transcript output with `--stream`
- live URL transcription with `--live`
- optional lightweight VAD for live mode with `--vad`
- REST server mode with `--server`
- CPU and NVIDIA GPU execution

REST documentation:

- [REST_API.md](REST_API.md)

## Requirements

- Rust `1.85+`
- Linux is the primary target
- no external `ffmpeg` dependency is required

For GPU runs:

- install with `--features cuda`
- NVIDIA driver must be available
- the project downloads the required ONNX Runtime CUDA libraries into its sandbox automatically

## Install

From crates.io:

```bash
cargo install transcribe-cli --locked
```

With GPU support from crates.io:

```bash
cargo install transcribe-cli --locked --features cuda
```

From a local checkout:

```bash
cargo install --path . --locked
```

With GPU support:

```bash
cargo install --path . --locked --features cuda
```

## First run

Models and runtime files are stored in a sandbox next to the installed binary:

```text
<binary_dir>/transcribe_sandbox/
```

This sandbox is used for:

- downloaded models
- ONNX Runtime shared libraries

If needed, you can override only the model storage directory:

```bash
transcribe-cli --models-dir /path/to/models file.wav
```

## Basic usage

Russian transcription:

```bash
transcribe-cli /path/to/file.wav
transcribe-cli --language ru /path/to/file.mp3
```

English transcription:

```bash
transcribe-cli --language en /path/to/file.wav
```

Video input:

```bash
transcribe-cli movie.mp4
```

Remote media URL:

```bash
transcribe-cli https://example.com/audio.mp3
```

Use GPU:

```bash
transcribe-cli --gpu /path/to/file.wav
transcribe-cli --gpu --gpu-device 0 /path/to/file.wav
```

Override compute type:

```bash
transcribe-cli --compute-type int8 /path/to/file.wav
transcribe-cli --compute-type float32 --gpu /path/to/file.wav
```

Stream transcript while decoding:

```bash
transcribe-cli --stream /path/to/file.wav
transcribe-cli --language en --stream song.mp3
```

Live URL transcription:

```bash
transcribe-cli --live http://127.0.0.1:8765/stream
transcribe-cli --live --gpu --language ru http://127.0.0.1:8765/stream
```

Live URL transcription with VAD:

```bash
transcribe-cli --live --vad http://127.0.0.1:8765/stream
```

Start REST server:

```bash
transcribe-cli --server 8787
```

## Main arguments

- `MEDIA`
  Path or URL to an audio or video source.
- `--language <ru|en>`
  Selects the backend.
- `--gpu`
  Enables NVIDIA GPU execution.
- `--gpu-device <N>`
  Selects CUDA device index.
- `--compute-type <auto|int8|float32>`
  Overrides compute mode.
- `--stream`
  Prints transcript chunk-by-chunk during normal file/URL transcription.
- `--live`
  Treats `MEDIA` as a live `http/https` stream.
- `--vad`
  Enables lightweight speech segmentation for `--live`.
- `--server <PORT>`
  Starts the REST API instead of a one-shot CLI transcription.
- `--models-dir <DIR>`
  Overrides the model directory.
- `--remove-model`
  Removes the selected model and its related artifacts.
- `--remove-all`
  Removes the whole model directory.

## Cleanup

Remove the current language-selected model:

```bash
transcribe-cli --language ru --remove-model
transcribe-cli --language en --remove-model
```

Remove all downloaded models:

```bash
transcribe-cli --remove-all
```

## Notes

- `ru` and `en` are the only supported language values right now.
- `--live` supports direct `http/https` streams, not playlist protocols such as HLS/DASH.
- `--vad` is currently valid only together with `--live`.
- Audio/video decoding is done inside the Rust pipeline through `symphonia`.
- The REST API is documented separately in [REST_API.md](REST_API.md).
