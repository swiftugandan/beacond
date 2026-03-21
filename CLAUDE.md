# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Beacon

Beacon (`beacond`) is an audio-based spatial awareness toolkit. It uses two complementary matching strategies: Shazam-style hash-based fingerprinting for precise matching and vector similarity signatures for ambient sound matching. Signatures can be computed via a built-in 192-dim spectral envelope or via an optional ML embedding backend (`--features ml-embeddings`) that runs a pre-trained ONNX audio model (e.g., YAMNet, 1024-dim) for better noise robustness. Zones can be defined by ambient sound signatures or ultrasonic beacons. Everything is self-hosted with no cloud dependencies.

## Build & Test Commands

```bash
cargo build                              # Debug build
cargo build --release                    # Release build (binary at ./target/release/beacond)
cargo build --features ml-embeddings     # Build with ONNX embedding support
cargo test                               # Run all tests
cargo test <test_name>                   # Run a single test
cargo run -- <COMMAND>                   # Run in dev mode (e.g., cargo run -- demo)
```

**Linux dependency**: `libasound2-dev` (ALSA headers) is required for mic support.

## Architecture

The binary is `beacond` (defined in `src/main.rs`), which delegates to `beacon::cli::execute()`. The library crate is `beacon`.

### Audio Fingerprinting Pipeline

The core algorithm flows through these modules in order:

1. **`audio.rs`** — WAV decoding, mono mixing, resampling, Butterworth high/low-pass filters. Defines `FrequencyMode` (Audible at 16kHz, Ultrasonic at 96kHz, Full at 96kHz) which controls sample rate and filter cutoffs throughout the system.

2. **`spectrogram.rs`** — Hann-windowed STFT via `rustfft`. Configurable window size (default 1024) and hop size (default 512). Produces magnitude frames.

3. **`fingerprint.rs`** — Constellation map peak extraction across frequency bands, then combinatorial hashing (anchor + target zone pairing) using `xxh3_64`. Produces `Vec<Fingerprint>` where each fingerprint is a `(hash: u64, offset: u32)` pair.

4. **`signature.rs`** — Alternative to hash matching: computes an L2-normalized feature vector. Supports variable-dimension vectors (192-dim spectral or model-dependent ML embeddings). Matching uses cosine similarity (dot product on normalized vectors). Signatures are stored as BLOBs in the zones table with a `signature_dim` column.

5. **`embeddings.rs`** — (feature-gated: `ml-embeddings`) Loads a pre-trained ONNX audio embedding model via `ort`. Resamples audio to 16kHz, zero-pads to 160-sample frame boundaries, runs inference, and L2-normalizes the output. The daemon uses this when `--model-path` is provided; falls back to spectral signatures otherwise.

6. **`database.rs`** — SQLite storage via `rusqlite` (bundled). Stores fingerprint hashes mapped to tracks/zones, plus signature BLOBs. Hash matching uses offset histogram alignment. The monitor loop runs both hash and signature matching in parallel, taking the best result.

### Daemon & CLI

- **`daemon.rs`** — Async TCP server (tokio) speaking newline-delimited JSON on port 18923. Handles recognition, ingestion, zone management, and monitor event broadcasting. Stores PID file at `~/.beacond/beacond.pid`, DB at `~/.beacond/beacond.db`.
- **`cli.rs`** — Clap-derived CLI. All commands work in two modes: if the daemon is running, the CLI connects via TCP; otherwise it opens the database directly.
- **`microphone.rs`** — Live mic capture via CPAL with sample rate negotiation.
- **`spectrogram_server.rs`** — HTTP server (default port 18924) that streams real-time 2048-point FFT frames at 60fps via SSE for browser visualization. Also exposes a zone capture/detection HTTP API (`/zone/capture/start`, `/zone/save`, `/zone/detect/start`, etc.).

### Frequency Modes

Default is **ultrasonic** (18–48 kHz). The `--audible` flag switches to standard audio range (20 Hz–8 kHz). `--full-spectrum` preserves everything. This mode selection propagates through the entire pipeline (sample rate, filters, fingerprinting).

## Daemon Protocol

Newline-delimited JSON over TCP. Request types are tagged via `{"type": "..."}`. Key types: `ping`, `recognize`, `ingest`, `zone_list`, `zone_add`, `zone_remove`, `zone_detect`, `monitor_subscribe`, `shutdown`.
