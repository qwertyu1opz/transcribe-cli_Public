# REST API

`transcribe-cli` can run as an HTTP service:

```bash
transcribe-cli --server 8787
```

Base URL:

```text
http://127.0.0.1:8787
```

## Endpoints

- `GET /`
- `GET /health`
- `POST /v1/transcribe`
- `POST /v1/transcribe/stream`
- `POST /v1/live/transcribe`

## GET /

Returns basic service info and endpoint list.

Example:

```bash
curl http://127.0.0.1:8787/
```

## GET /health

Returns a simple health status.

Example:

```bash
curl http://127.0.0.1:8787/health
```

Response:

```json
{
  "status": "ok",
  "service": "transcribe-cli"
}
```

## POST /v1/transcribe

Runs a normal one-shot transcription and returns JSON.

Request body:

```json
{
  "media": "/path/to/file.wav",
  "language": "ru",
  "gpu": false,
  "gpu_device": 0,
  "compute_type": "auto"
}
```

Fields:

- `media`
  Required. Local path or `http/https` URL.
- `language`
  Optional. `ru` or `en`. Default is `ru`.
- `gpu`
  Optional boolean. Default is `false`.
- `gpu_device`
  Optional integer. Default is `0`.
- `compute_type`
  Optional. `auto`, `int8`, or `float32`.

Example:

```bash
curl http://127.0.0.1:8787/v1/transcribe \
  -H 'content-type: application/json' \
  -d '{
    "media": "/home/user/audio.wav",
    "language": "ru",
    "gpu": false,
    "compute_type": "auto"
  }'
```

Successful response:

```json
{
  "transcript": "example transcript",
  "source": "/home/user/audio.wav",
  "language": "ru",
  "requested_model": "gigaam-v3",
  "runtime_model": "v3_e2e_ctc",
  "device": "cpu",
  "compute_type": "int8",
  "target_sample_rate": 16000,
  "source_sample_rate": 48000,
  "channels": 1,
  "codec": "CodecType(...)",
  "duration_seconds": 12.34
}
```

## POST /v1/transcribe/stream

Runs a normal transcription but returns progress as Server-Sent Events.

This is useful when you want chunk-by-chunk output for a file or URL.

Request body is the same as `/v1/transcribe`.

Example:

```bash
curl -N http://127.0.0.1:8787/v1/transcribe/stream \
  -H 'content-type: application/json' \
  -d '{
    "media": "/home/user/audio.wav",
    "language": "ru",
    "gpu": false,
    "compute_type": "auto"
  }'
```

SSE events:

- `ready`
- `status`
- `chunk`
- `done`
- `error`

Typical stream:

```text
event: ready
data: starting stream transcription for /home/user/audio.wav

event: status
data: resolving model

event: status
data: preparing media

event: chunk
data: first transcript chunk

event: chunk
data: second transcript chunk

event: done
data: complete
```

## POST /v1/live/transcribe

Runs live transcription from a direct `http/https` media stream and returns SSE.

Request body:

```json
{
  "media": "http://127.0.0.1:8765/stream",
  "language": "ru",
  "gpu": false,
  "gpu_device": 0,
  "compute_type": "auto",
  "vad": false
}
```

Additional field:

- `vad`
  Optional boolean. Enables lightweight speech segmentation for live mode.

Example:

```bash
curl -N http://127.0.0.1:8787/v1/live/transcribe \
  -H 'content-type: application/json' \
  -d '{
    "media": "http://127.0.0.1:8765/stream",
    "language": "ru",
    "gpu": true,
    "gpu_device": 0,
    "compute_type": "float32",
    "vad": true
  }'
```

SSE events:

- `ready`
- `status`
- `chunk`
- `done`
- `error`

Typical stream:

```text
event: ready
data: starting live transcription for http://127.0.0.1:8765/stream

event: status
data: live / 3.5s buffered / raw f32le

event: chunk
data: recognized live text
```

## Error format

Non-SSE endpoints return JSON errors:

```json
{
  "error": "human-readable message"
}
```

SSE endpoints emit:

```text
event: error
data: human-readable message
```

## Notes

- Supported language values are only `ru` and `en`.
- `ru` uses GigaAM.
- `en` uses Parakeet TDT v2.
- `live` expects a direct stream URL, not a playlist page.
