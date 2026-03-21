//! Real-time spectrogram server with room signature capture & detection.
//!
//! Captures audio from the microphone at 96 kHz via CPAL, computes FFT frames
//! using rustfft, and streams the magnitude data to a browser via Server-Sent Events.
//! The HTML visualization is served from the same HTTP endpoint.
//!
//! Additional endpoints enable capturing room audio signatures, saving them as
//! named zones, and continuously detecting which room the device is in.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::{Arc, Mutex};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

use crate::audio::{AudioSignal, FrequencyMode};
use crate::database::Database;
use crate::fingerprint::Fingerprinter;

/// FFT configuration for the real-time spectrogram.
const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = 1024;
const NUM_BINS: usize = FFT_SIZE / 2 + 1; // 1025

/// How many seconds of audio to keep for zone capture / detection.
const AUDIO_HISTORY_SECS: usize = 10;

/// Shared state for zone capture and detection.
struct ZoneState {
    /// Rolling buffer of mono samples at capture sample rate.
    mono_history: Vec<f32>,
    /// Max samples to keep in history.
    max_history_samples: usize,
    /// Whether we are currently capturing for a new zone.
    capturing: bool,
    /// Accumulated samples during capture.
    capture_buf: Vec<f32>,
    /// Whether continuous detection is enabled.
    detecting: bool,
    /// Last detection result (JSON string).
    last_detection: String,
}

/// Start the spectrogram server and open the browser.
pub async fn run(port: u16, bind: &str) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No audio input device found")?;
    let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());

    // Find highest sample rate config
    let (stream_config, sample_format, sample_rate, channels) = {
        if let Some((cfg, fmt)) = crate::microphone::find_high_rate_config(&device) {
            let sr = cfg.sample_rate.0;
            let ch = cfg.channels as usize;
            (cfg, fmt, sr, ch)
        } else {
            let default_cfg = device
                .default_input_config()
                .context("No input config available")?;
            let sr = default_cfg.sample_rate().0;
            let ch = default_cfg.channels() as usize;
            let fmt = default_cfg.sample_format();
            (default_cfg.into(), fmt, sr, ch)
        }
    };

    println!(
        "  Mic: {} ({}Hz, {}ch, {:?})",
        device_name, sample_rate, channels, sample_format
    );
    println!(
        "  FFT: {} samples, {} bins, {:.1} Hz/bin",
        FFT_SIZE,
        NUM_BINS,
        sample_rate as f32 / FFT_SIZE as f32
    );
    println!("  Nyquist: {} Hz", sample_rate / 2);

    // Ring buffer: audio callback pushes samples, FFT thread consumes them
    let ring: Arc<Mutex<Vec<f32>>> =
        Arc::new(Mutex::new(Vec::with_capacity(sample_rate as usize * 2)));
    let ring_clone = Arc::clone(&ring);

    // Broadcast channel for SSE clients (spectrogram frames)
    let (tx, _) = broadcast::channel::<Vec<u8>>(64);
    let tx_clone = tx.clone();

    // Zone state shared between FFT thread, detection loop, and HTTP handlers
    let zone_state = Arc::new(Mutex::new(ZoneState {
        mono_history: Vec::with_capacity(sample_rate as usize * AUDIO_HISTORY_SECS),
        max_history_samples: sample_rate as usize * AUDIO_HISTORY_SECS,
        capturing: false,
        capture_buf: Vec::new(),
        detecting: false,
        last_detection: String::new(),
    }));

    // Broadcast channel for zone detection events (SSE)
    let (zone_tx, _) = broadcast::channel::<String>(16);

    // Start audio capture
    let err_flag: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let err_clone = Arc::clone(&err_flag);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let buf = Arc::clone(&ring_clone);
            let ef = Arc::clone(&err_clone);
            device
                .build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut b) = buf.lock() {
                            b.extend_from_slice(data);
                        }
                    },
                    move |err| {
                        if let Ok(mut f) = ef.lock() {
                            *f = Some(err.to_string());
                        }
                    },
                    None,
                )
                .context("Failed to build F32 stream")?
        }
        cpal::SampleFormat::I16 => {
            let buf = Arc::clone(&ring_clone);
            let ef = Arc::clone(&err_clone);
            device
                .build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                        if let Ok(mut b) = buf.lock() {
                            b.extend_from_slice(&floats);
                        }
                    },
                    move |err| {
                        if let Ok(mut f) = ef.lock() {
                            *f = Some(err.to_string());
                        }
                    },
                    None,
                )
                .context("Failed to build I16 stream")?
        }
        _ => {
            let buf = Arc::clone(&ring_clone);
            let ef = Arc::clone(&err_clone);
            device
                .build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut b) = buf.lock() {
                            b.extend_from_slice(data);
                        }
                    },
                    move |err| {
                        if let Ok(mut f) = ef.lock() {
                            *f = Some(err.to_string());
                        }
                    },
                    None,
                )
                .context("Failed to build stream")?
        }
    };

    stream.play().context("Failed to start audio stream")?;
    println!("  Audio capture started");

    // FFT processing thread — also feeds mono samples to zone_state
    let fft_ring = Arc::clone(&ring);
    let fft_channels = channels;
    let fft_sr = sample_rate;
    let fft_zone_state = Arc::clone(&zone_state);
    std::thread::spawn(move || {
        run_fft_loop(fft_ring, fft_channels, fft_sr, tx_clone, fft_zone_state);
    });

    // Background zone detection loop
    let detect_zone_state = Arc::clone(&zone_state);
    let detect_zone_tx = zone_tx.clone();
    let detect_sr = sample_rate;
    tokio::spawn(async move {
        run_detection_loop(detect_zone_state, detect_zone_tx, detect_sr).await;
    });

    // Start HTTP server
    let addr = format!("{}:{}", bind, port);
    let listener = TcpListener::bind(&addr)
        .await
        .context(format!("Failed to bind to {}", addr))?;

    println!("  Server: http://{}", addr);
    println!();
    println!("  Opening browser...");

    // Open browser (always use localhost for the local browser)
    let url = format!("http://127.0.0.1:{}", port);
    let _ = std::process::Command::new("open").arg(&url).spawn();

    println!("  Press Ctrl+C to stop");
    println!();

    // Accept connections
    loop {
        let (socket, _addr) = listener.accept().await?;
        let tx = tx.clone();
        let sr = sample_rate;
        let ch = channels;
        let dev = device_name.clone();
        let zs = Arc::clone(&zone_state);
        let ztx = zone_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(socket, tx, sr, ch, &dev, zs, ztx).await {
                log::debug!("Connection error: {}", e);
            }
        });
    }
}

/// Target frame rate for SSE output.
const TARGET_FPS: u32 = 60;

/// FFT processing loop — optimized hot path.
///
/// Perf notes:
///  - Ring buffer drained via `std::mem::take` (O(1), no memmove)
///  - Mono buffer uses cursor index instead of `drain()` (avoids O(n) shift)
///  - Skips `sqrt()`: accumulates mag² and converts via `10*log10` instead of `20*log10(sqrt)`
///  - Pre-allocated output message buffer reused every tick
///  - Hann window applied with direct indexing (auto-vectorizable)
fn run_fft_loop(
    ring: Arc<Mutex<Vec<f32>>>,
    channels: usize,
    sample_rate: u32,
    tx: broadcast::Sender<Vec<u8>>,
    zone_state: Arc<Mutex<ZoneState>>,
) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    // Pre-compute Hann window
    let hann = crate::spectrogram::hann_window(FFT_SIZE);

    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
    let mut mono_buf: Vec<f32> = Vec::with_capacity(sample_rate as usize);
    let mut cursor: usize = 0; // read position in mono_buf (avoids drain)

    // Accumulator: stores mag² (no sqrt needed)
    let mut accum = vec![0.0f32; NUM_BINS];
    let mut accum_count: u32 = 0;

    // Pre-allocated output message (reused every tick)
    let msg_size = 6 + NUM_BINS * 4;
    let mut msg: Vec<u8> = vec![0u8; msg_size];
    msg[0..2].copy_from_slice(&(NUM_BINS as u16).to_le_bytes());
    msg[2..6].copy_from_slice(&sample_rate.to_le_bytes());

    let tick = std::time::Duration::from_micros(1_000_000 / TARGET_FPS as u64);
    let inv_channels = 1.0 / channels as f32;

    // Temp buffer for new mono samples to feed into zone_state
    let mut new_mono: Vec<f32> = Vec::new();

    loop {
        let frame_start = std::time::Instant::now();
        new_mono.clear();

        // Drain ring buffer (O(1) swap)
        {
            let mut r = ring.lock().unwrap();
            if !r.is_empty() {
                let drained = std::mem::take(&mut *r);
                drop(r); // release lock before processing

                // Mix to mono and append
                if channels > 1 {
                    mono_buf.reserve(drained.len() / channels);
                    for chunk in drained.chunks_exact(channels) {
                        let sum: f32 = chunk.iter().sum();
                        let sample = sum * inv_channels;
                        mono_buf.push(sample);
                        new_mono.push(sample);
                    }
                } else {
                    new_mono.extend_from_slice(&drained);
                    mono_buf.extend_from_slice(&drained);
                }
            }
        }

        // Feed mono samples into zone_state for capture/detection
        if !new_mono.is_empty() {
            if let Ok(mut zs) = zone_state.lock() {
                // Append to rolling history
                zs.mono_history.extend_from_slice(&new_mono);
                if zs.mono_history.len() > zs.max_history_samples {
                    let excess = zs.mono_history.len() - zs.max_history_samples;
                    zs.mono_history.drain(..excess);
                }
                // If capturing, also accumulate
                if zs.capturing {
                    zs.capture_buf.extend_from_slice(&new_mono);
                }
            }
        }

        // Process all available FFT frames using cursor (no memmove)
        while cursor + FFT_SIZE <= mono_buf.len() {
            let slice = &mono_buf[cursor..cursor + FFT_SIZE];
            for i in 0..FFT_SIZE {
                fft_buffer[i] = Complex::new(slice[i] * hann[i], 0.0);
            }
            fft.process(&mut fft_buffer);

            // Accumulate mag² (skip sqrt — convert with 10*log10 later)
            for i in 0..NUM_BINS {
                let c = &fft_buffer[i];
                accum[i] += c.re * c.re + c.im * c.im;
            }
            accum_count += 1;
            cursor += HOP_SIZE;
        }

        // Compact mono_buf when cursor is past half (amortized O(1))
        if cursor > mono_buf.len() / 2 && cursor > FFT_SIZE * 4 {
            mono_buf.drain(..cursor);
            cursor = 0;
        }

        // Send one averaged frame per tick
        if accum_count > 0 {
            let inv = 1.0 / accum_count as f32;
            let out = &mut msg[6..];
            for i in 0..NUM_BINS {
                let mag_sq = accum[i] * inv;
                // 10*log10(mag²) = 20*log10(mag), avoids sqrt
                let db = if mag_sq > 1e-30 {
                    10.0 * mag_sq.log10()
                } else {
                    -150.0
                };
                out[i * 4..i * 4 + 4].copy_from_slice(&db.to_le_bytes());
                accum[i] = 0.0;
            }
            accum_count = 0;
            let _ = tx.send(msg.clone());
        }

        // Prevent unbounded growth
        let max_buf = sample_rate as usize * 2;
        if mono_buf.len() - cursor > max_buf {
            let new_start = mono_buf.len() - FFT_SIZE * 2;
            mono_buf.drain(..new_start);
            cursor = 0;
        }

        // Sleep remainder of tick
        let elapsed = frame_start.elapsed();
        if elapsed < tick {
            std::thread::sleep(tick - elapsed);
        }
    }
}

/// Background loop: every ~2 seconds, fingerprint recent audio and match against all zones.
/// Sends all zone matches with probabilities so the UI can display live bars.
async fn run_detection_loop(
    zone_state: Arc<Mutex<ZoneState>>,
    zone_tx: broadcast::Sender<String>,
    sample_rate: u32,
) {
    let detect_duration_secs: usize = 5;
    let detect_samples = sample_rate as usize * detect_duration_secs;
    let fp = Fingerprinter::with_defaults();
    let db_path = crate::daemon::default_db_path();

    // Open the database once; reopen only on error.
    let mut db: Option<Database> = Database::open(&db_path).ok();

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Check if detection is enabled and grab samples
        let samples = {
            let zs = zone_state.lock().unwrap();
            if !zs.detecting {
                continue;
            }
            let history = &zs.mono_history;
            if history.len() < detect_samples {
                continue;
            }
            let start = history.len() - detect_samples;
            history[start..].to_vec()
        };

        let sr = sample_rate;
        let zs_clone = Arc::clone(&zone_state);
        let ztx = zone_tx.clone();

        // Take the DB handle so we can move it into spawn_blocking
        let mut db_handle = db.take();
        let fp_clone = fp.config.clone();
        let db_path_clone = db_path.clone();

        let returned_db = tokio::task::spawn_blocking(move || {
            let signal = AudioSignal::from_samples_with_mode(samples, sr, FrequencyMode::Full);

            let fingerprinter = Fingerprinter::new(fp_clone);
            let fingerprints = fingerprinter.fingerprint(&signal.samples, signal.sample_rate);

            if fingerprints.is_empty() {
                let json = serde_json::json!({
                    "matches": [],
                    "fingerprints": 0,
                    "message": "No fingerprints extracted"
                })
                .to_string();
                if let Ok(mut zs) = zs_clone.lock() {
                    zs.last_detection = json.clone();
                }
                let _ = ztx.send(json);
                return db_handle;
            }

            // Reopen database if needed
            if db_handle.is_none() {
                db_handle = Database::open(&db_path_clone).ok();
            }

            let Some(ref current_db) = db_handle else {
                let json = serde_json::json!({
                    "matches": [],
                    "fingerprints": fingerprints.len(),
                    "message": "Database not available"
                })
                .to_string();
                if let Ok(mut zs) = zs_clone.lock() {
                    zs.last_detection = json.clone();
                }
                let _ = ztx.send(json);
                return db_handle;
            };

            let results = current_db
                .search(&fingerprints, 10, signal.sample_rate)
                .unwrap_or_default();

            // Collect ALL zone matches with their confidence
            let mut zone_matches: Vec<serde_json::Value> = Vec::new();
            for result in &results {
                if let Ok(Some(zone)) = current_db.get_zone_by_track_id(result.track.id) {
                    zone_matches.push(serde_json::json!({
                        "zone": zone.name,
                        "confidence": (result.confidence * 10000.0).round() / 10000.0,
                        "match_count": result.match_count,
                    }));
                }
            }

            let json = serde_json::json!({
                "matches": zone_matches,
                "fingerprints": fingerprints.len(),
            })
            .to_string();

            if let Ok(mut zs) = zs_clone.lock() {
                zs.last_detection = json.clone();
            }
            let _ = ztx.send(json);
            db_handle
        })
        .await
        .ok()
        .flatten();

        // Restore the DB handle for the next iteration
        db = returned_db;
    }
}

/// Handle an HTTP connection: serve the HTML page, SSE streams, or zone API.
async fn handle_connection(
    mut socket: tokio::net::TcpStream,
    tx: broadcast::Sender<Vec<u8>>,
    sample_rate: u32,
    channels: usize,
    device_name: &str,
    zone_state: Arc<Mutex<ZoneState>>,
    zone_tx: broadcast::Sender<String>,
) -> Result<()> {
    use tokio::io::AsyncReadExt;

    // Read HTTP request
    let mut req_buf = vec![0u8; 8192];
    let n = socket.read(&mut req_buf).await?;
    let req = String::from_utf8_lossy(&req_buf[..n]);

    // Parse request line
    let first_line = req.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let method = parts.first().copied().unwrap_or("GET");
    let path = parts.get(1).copied().unwrap_or("/");

    // Extract body for POST requests
    let body = req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();

    match (method, path) {
        ("GET", "/stream") => {
            // SSE endpoint for spectrogram data
            let headers = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\n\
                Connection: keep-alive\r\n\
                Access-Control-Allow-Origin: *\r\n\r\n";
            socket.write_all(headers.as_bytes()).await?;

            let mut rx = tx.subscribe();

            loop {
                match rx.recv().await {
                    Ok(data) => {
                        use std::fmt::Write;
                        let b64 = base64_encode(&data);
                        let mut event = String::new();
                        write!(event, "data: {}\n\n", b64)?;
                        if socket.write_all(event.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        }

        ("GET", "/info") => {
            let info = serde_json::json!({
                "sample_rate": sample_rate,
                "channels": channels,
                "device": device_name,
                "fft_size": FFT_SIZE,
                "num_bins": NUM_BINS,
                "freq_resolution": sample_rate as f32 / FFT_SIZE as f32,
                "nyquist": sample_rate / 2,
            });
            let body = serde_json::to_string(&info)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\n\
                Content-Type: application/json\r\n\
                Content-Length: {}\r\n\
                Access-Control-Allow-Origin: *\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await?;
        }

        // ── Zone API ────────────────────────────────────────────────
        ("POST", "/zone/capture/start") => {
            // Start capturing audio for a new zone
            {
                let mut zs = zone_state.lock().unwrap();
                zs.capturing = true;
                zs.capture_buf.clear();
            }
            send_json(
                &mut socket,
                200,
                &serde_json::json!({"status": "capturing"}),
            )
            .await?;
        }

        ("POST", "/zone/capture/stop") => {
            // Stop capturing
            let sample_count = {
                let mut zs = zone_state.lock().unwrap();
                zs.capturing = false;
                zs.capture_buf.len()
            };
            let duration = sample_count as f32 / sample_rate as f32;
            send_json(
                &mut socket,
                200,
                &serde_json::json!({
                    "status": "stopped",
                    "samples": sample_count,
                    "duration_secs": (duration * 10.0).round() / 10.0
                }),
            )
            .await?;
        }

        ("POST", "/zone/save") => {
            // Save captured audio as a named zone
            // Body: {"name": "kitchen"}
            let parsed: serde_json::Value =
                serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
            let name = parsed["name"].as_str().unwrap_or("").trim().to_string();

            if name.is_empty() {
                send_json(
                    &mut socket,
                    400,
                    &serde_json::json!({"error": "name is required"}),
                )
                .await?;
                return Ok(());
            }

            // Take the captured samples
            let samples = {
                let mut zs = zone_state.lock().unwrap();
                zs.capturing = false;
                std::mem::take(&mut zs.capture_buf)
            };

            if samples.is_empty() {
                send_json(
                    &mut socket,
                    400,
                    &serde_json::json!({"error": "No audio captured. Start capture first."}),
                )
                .await?;
                return Ok(());
            }

            let sr = sample_rate;
            let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
                let duration_secs = samples.len() as f32 / sr as f32;

                // Process through ultrasonic pipeline
                let signal = AudioSignal::from_samples_with_mode(
                    samples,
                    sr,
                    FrequencyMode::Full,
                );

                let fp = Fingerprinter::with_defaults();
                let fingerprints = fp.fingerprint(&signal.samples, signal.sample_rate);

                if fingerprints.is_empty() {
                    return Ok(serde_json::json!({"error": "No fingerprints could be extracted from the captured audio"}));
                }

                let db_path = crate::daemon::default_db_path();
                let mut db = Database::open(&db_path)?;

                // Insert as a track + zone
                let track_id = db.insert_track(
                    &format!("zone:{}", name),
                    "beacon",
                    None,
                    duration_secs,
                    None,
                    &fingerprints,
                )?;

                let sig = crate::signature::Signature::from_samples(
                    &signal.samples,
                    signal.sample_rate,
                );
                let sig_bytes = sig.to_bytes();
                db.insert_zone(&name, track_id, "ultrasonic", Some(&sig_bytes))?;

                Ok(serde_json::json!({
                    "status": "saved",
                    "zone": name,
                    "fingerprints": fingerprints.len(),
                    "signature_dim": sig.vector.len(),
                    "duration_secs": (duration_secs * 10.0).round() / 10.0,
                    "track_id": track_id
                }))
            }).await??;

            let status = if result.get("error").is_some() {
                400
            } else {
                200
            };
            send_json(&mut socket, status, &result).await?;
        }

        ("GET", "/zones") => {
            // List all saved zones
            let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
                let db_path = crate::daemon::default_db_path();
                let db = Database::open(&db_path)?;
                let zones = db.list_zones()?;
                let list: Vec<serde_json::Value> = zones
                    .iter()
                    .map(|z| {
                        serde_json::json!({
                            "name": z.name,
                            "fingerprints": z.fingerprint_count,
                            "duration_secs": z.duration_secs,
                            "added_at": z.added_at,
                        })
                    })
                    .collect();
                Ok(serde_json::json!({"zones": list}))
            })
            .await??;
            send_json(&mut socket, 200, &result).await?;
        }

        (_, p) if p.starts_with("/zone/delete/") => {
            let zone_name = &p["/zone/delete/".len()..];
            let zone_name = urldecode(zone_name);
            let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
                let db_path = crate::daemon::default_db_path();
                let mut db = Database::open(&db_path)?;
                if db.remove_zone(&zone_name)? {
                    Ok(serde_json::json!({"status": "deleted", "zone": zone_name}))
                } else {
                    Ok(serde_json::json!({"error": format!("Zone '{}' not found", zone_name)}))
                }
            })
            .await??;
            let status = if result.get("error").is_some() {
                404
            } else {
                200
            };
            send_json(&mut socket, status, &result).await?;
        }

        ("POST", "/zone/detect/start") => {
            {
                let mut zs = zone_state.lock().unwrap();
                zs.detecting = true;
            }
            send_json(
                &mut socket,
                200,
                &serde_json::json!({"status": "detecting"}),
            )
            .await?;
        }

        ("POST", "/zone/detect/stop") => {
            {
                let mut zs = zone_state.lock().unwrap();
                zs.detecting = false;
                zs.last_detection.clear();
            }
            send_json(&mut socket, 200, &serde_json::json!({"status": "stopped"})).await?;
        }

        ("GET", "/zone/events") => {
            // SSE endpoint for zone detection events
            let headers = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\n\
                Connection: keep-alive\r\n\
                Access-Control-Allow-Origin: *\r\n\r\n";
            socket.write_all(headers.as_bytes()).await?;

            let mut rx = zone_tx.subscribe();

            loop {
                match rx.recv().await {
                    Ok(data) => {
                        let event = format!("data: {}\n\n", data);
                        if socket.write_all(event.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        }

        ("GET", "/zone/detect/once") => {
            // One-shot detection: grab recent audio, fingerprint, match
            let samples = {
                let zs = zone_state.lock().unwrap();
                let history = &zs.mono_history;
                let detect_samples = sample_rate as usize * 5;
                if history.len() < detect_samples {
                    Vec::new()
                } else {
                    let start = history.len() - detect_samples;
                    history[start..].to_vec()
                }
            };

            if samples.is_empty() {
                send_json(
                    &mut socket,
                    200,
                    &serde_json::json!({
                        "detected": false,
                        "message": "Not enough audio accumulated yet"
                    }),
                )
                .await?;
                return Ok(());
            }

            let sr = sample_rate;
            let result = tokio::task::spawn_blocking(move || -> serde_json::Value {
                let signal = AudioSignal::from_samples_with_mode(samples, sr, FrequencyMode::Full);

                let fp = Fingerprinter::with_defaults();
                let fingerprints = fp.fingerprint(&signal.samples, signal.sample_rate);

                if fingerprints.is_empty() {
                    return serde_json::json!({
                        "detected": false,
                        "message": "No fingerprints extracted"
                    });
                }

                let db_path = crate::daemon::default_db_path();
                let db = match Database::open(&db_path) {
                    Ok(db) => db,
                    Err(_) => {
                        return serde_json::json!({
                            "detected": false,
                            "message": "Database not available"
                        })
                    }
                };

                let results = db
                    .search(&fingerprints, 5, signal.sample_rate)
                    .unwrap_or_default();

                for result in &results {
                    if let Ok(Some(zone)) = db.get_zone_by_track_id(result.track.id) {
                        return serde_json::json!({
                            "detected": true,
                            "zone": zone.name,
                            "confidence": (result.confidence * 100.0).round() / 100.0,
                            "match_count": result.match_count,
                        });
                    }
                }

                serde_json::json!({
                    "detected": false,
                    "message": "No zone match"
                })
            })
            .await
            .unwrap_or(serde_json::json!({"detected": false, "message": "Internal error"}));

            send_json(&mut socket, 200, &result).await?;
        }

        _ => {
            // Serve HTML page
            let html = build_html(sample_rate, channels, device_name);
            let resp = format!(
                "HTTP/1.1 200 OK\r\n\
                Content-Type: text/html; charset=utf-8\r\n\
                Content-Length: {}\r\n\r\n{}",
                html.len(),
                html
            );
            socket.write_all(resp.as_bytes()).await?;
        }
    }

    Ok(())
}

/// Send a JSON response.
async fn send_json(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    value: &serde_json::Value,
) -> Result<()> {
    let body = serde_json::to_string(value)?;
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "OK",
    };
    let resp = format!(
        "HTTP/1.1 {} {}\r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        Access-Control-Allow-Origin: *\r\n\r\n{}",
        status,
        status_text,
        body.len(),
        body
    );
    socket.write_all(resp.as_bytes()).await?;
    Ok(())
}

/// Simple percent-decode for URL path segments (handles multi-byte UTF-8).
fn urldecode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut iter = s.bytes();
    while let Some(b) = iter.next() {
        if b == b'%' {
            let h = iter.next().unwrap_or(b'0');
            let l = iter.next().unwrap_or(b'0');
            bytes.push(hex_val(h) * 16 + hex_val(l));
        } else if b == b'+' {
            bytes.push(b' ');
        } else {
            bytes.push(b);
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// Simple base64 encoder (no external dependency).
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3f) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Build the HTML visualization page with room signature UI (embedded in the binary).
fn build_html(sample_rate: u32, _channels: usize, device_name: &str) -> String {
    let nyquist = sample_rate / 2;
    let freq_res = sample_rate as f32 / FFT_SIZE as f32;

    format!(
        r##"<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Beacon — Realtime Spectrogram + Room Detection</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a0a; color: #ccc;
    font-family: -apple-system, 'SF Mono', 'Menlo', monospace;
    overflow: hidden; height: 100vh;
  }}
  #header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 16px; background: #111; border-bottom: 1px solid #222;
    min-height: 44px; position: relative; z-index: 50;
  }}
  #header h1 {{ color: #00ccff; font-size: 14px; font-weight: 600; }}
  #controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  button {{
    background: #222; color: #ccc; border: 1px solid #444; border-radius: 4px;
    padding: 4px 10px; font-size: 11px; font-family: inherit; cursor: pointer;
  }}
  button:hover {{ background: #333; }}
  button.active {{ background: #00ccff22; border-color: #00ccff; color: #00ccff; }}
  button.recording {{ background: #ff440033; border-color: #ff6600; color: #ff6600; animation: pulse 1s infinite; }}
  button.detecting {{ background: #44ff4422; border-color: #44ff44; color: #44ff44; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} }}
  .sep {{ width: 1px; height: 20px; background: #333; }}
  [data-tip] {{ position: relative; display: inline-flex; align-items: center; }}
  .tip {{
    position: absolute; top: calc(100% + 10px); left: 50%; transform: translateX(-50%);
    background: #1e1e1e; color: #ccc; border: 1px solid #444; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; line-height: 1.6; width: max-content;
    max-width: 280px; white-space: normal; pointer-events: none;
    opacity: 0; visibility: hidden; transition: opacity 0.15s, visibility 0.15s;
    z-index: 200; box-shadow: 0 6px 20px #000a; text-align: left;
  }}
  .tip::after {{
    content: ''; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
    border: 7px solid transparent; border-bottom-color: #444;
  }}
  [data-tip]:hover > .tip {{ opacity: 1; visibility: visible; }}
  #main {{ display: flex; flex-direction: column; height: calc(100vh - 44px); }}
  #spectrogram-container {{ flex: 1; position: relative; min-height: 0; }}
  #spectrogram {{ width: 100%; height: 100%; display: block; }}
  #freq-overlay {{
    position: absolute; left: 0; top: 0; bottom: 0; right: 0; pointer-events: none;
  }}
  .freq-label {{
    position: absolute; left: 4px; color: #888; font-size: 10px;
    transform: translateY(-50%); text-shadow: 0 0 4px #000, 0 0 2px #000;
  }}
  .freq-line {{
    position: absolute; left: 45px; right: 0; height: 1px; background: #ffffff12;
  }}
  .freq-line.ultra {{ background: #00ccff33; height: 2px; }}
  #cursor-info {{
    position: absolute; top: 8px; right: 8px; background: #000c;
    padding: 4px 8px; border-radius: 3px; font-size: 11px; color: #aaa;
    pointer-events: none; display: none;
  }}
  /* Zone detection overlay */
  #zone-overlay {{
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    background: #000d; border: 1px solid #333; border-radius: 8px;
    padding: 12px 20px; pointer-events: none;
    min-width: 320px; max-width: 500px; transition: all 0.3s;
  }}
  #zone-overlay.detected {{ border-color: #44ff44; background: #0a1a0dee; }}
  #zone-overlay.no-match {{ border-color: #555; }}
  #zone-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 8px;
  }}
  #zone-best {{ font-size: 18px; font-weight: 700; color: #44ff44; }}
  #zone-overlay.no-match #zone-best {{ color: #555; font-size: 13px; font-weight: 400; }}
  #zone-fps {{ font-size: 10px; color: #555; }}
  .zone-bar-row {{
    display: flex; align-items: center; gap: 8px; margin: 4px 0;
  }}
  .zone-bar-name {{
    font-size: 11px; color: #ccc; min-width: 80px; text-align: right;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .zone-bar-track {{
    flex: 1; height: 16px; background: #1a1a1a; border-radius: 3px;
    overflow: hidden; position: relative;
  }}
  .zone-bar-fill {{
    height: 100%; border-radius: 3px; transition: width 0.4s ease-out;
    background: linear-gradient(90deg, #225522, #44ff44);
  }}
  .zone-bar-fill.best {{
    background: linear-gradient(90deg, #225522, #44ff44);
  }}
  .zone-bar-fill.other {{
    background: linear-gradient(90deg, #222244, #6688cc);
  }}
  .zone-bar-pct {{
    font-size: 11px; color: #aaa; min-width: 48px; text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .zone-bar-pct.best {{ color: #44ff44; font-weight: 600; }}
  .zone-bar-matches {{
    font-size: 9px; color: #555; min-width: 30px;
  }}
  #bottom-panel {{
    height: 220px; min-height: 220px; border-top: 1px solid #222; display: flex;
  }}
  #spectrum-chart {{ flex: 1; position: relative; }}
  #spectrum-canvas {{ width: 100%; height: 100%; display: block; }}
  #peaks-panel {{
    width: 200px; padding: 10px 14px; border-left: 1px solid #222;
    overflow-y: auto; font-size: 11px;
  }}
  #peaks-panel h3 {{ color: #ff8800; font-size: 12px; margin-bottom: 6px; }}
  .peak-row {{
    display: flex; justify-content: space-between; padding: 2px 0;
    border-bottom: 1px solid #1a1a1a;
  }}
  .peak-freq {{ color: #00ccff; }}
  .peak-db {{ color: #ff8800; min-width: 55px; text-align: right; }}
  .peak-source {{ color: #666; font-size: 10px; margin-bottom: 2px; }}
  #connection-status {{ font-size: 10px; padding: 2px 6px; border-radius: 3px; }}
  .connected {{ color: #44ff44; }}
  .disconnected {{ color: #ff4444; }}

  /* Room panel — own column */
  #room-panel {{
    width: 240px; padding: 10px 14px; border-left: 1px solid #222;
    overflow-y: auto; font-size: 11px;
  }}
  #room-panel h3 {{ color: #44ff44; font-size: 12px; margin-bottom: 6px; }}
  .room-capture {{
    display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px;
  }}
  .room-capture input {{
    background: #1a1a1a; border: 1px solid #333; border-radius: 3px;
    color: #ccc; padding: 4px 8px; font-size: 11px; font-family: inherit;
    width: 100%;
  }}
  .room-capture input:focus {{ outline: none; border-color: #00ccff; }}
  .room-actions {{ display: flex; gap: 4px; }}
  .room-actions button {{ flex: 1; font-size: 10px; padding: 4px 6px; }}
  .zone-item {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 0; border-bottom: 1px solid #1a1a1a;
  }}
  .zone-item-name {{ color: #ccc; font-size: 11px; }}
  .zone-item-meta {{ color: #555; font-size: 9px; }}
  .zone-item button {{
    background: none; border: none; color: #ff4444; cursor: pointer;
    font-size: 10px; padding: 2px 4px;
  }}
  .zone-item button:hover {{ color: #ff6666; }}
  #capture-status {{
    font-size: 10px; color: #888; padding: 2px 0;
  }}
  #capture-timer {{
    font-size: 10px; color: #ff6600; font-weight: 600;
  }}
</style>
</head>
<body>

<div id="header">
  <h1>Beacon — {sample_rate}Hz Spectrogram</h1>
  <div id="controls">
    <span style="font-size:10px;color:#44ff44">{device_name_esc}</span>
    <span style="font-size:10px;color:#888">{sample_rate}Hz · {nyquist}Hz Nyquist · {freq_res:.1}Hz/bin</span>
    <span class="sep"></span>
    <span data-tip><button id="btn-zoom-ultra" class="active">Ultrasonic</button>
      <span class="tip">Zoom to 14–{nyquist_k}kHz — the full ultrasonic range.</span></span>
    <span data-tip><button id="btn-zoom-full">Full Range</button>
      <span class="tip">Show all frequencies from 0 to {nyquist_k}kHz.</span></span>
    <span data-tip><button id="btn-zoom-high">24–{nyquist_k}kHz</button>
      <span class="tip">Zoom to the browser-invisible range.</span></span>
    <span class="sep"></span>
    <span data-tip><button id="btn-baseline">Capture Baseline</button>
      <span class="tip">Record ~2 seconds of electronic noise, then subtract it.</span></span>
    <span data-tip><button id="btn-clear" disabled>Clear</button>
      <span class="tip">Discard baseline and return to raw view.</span></span>
    <span data-tip><button id="btn-mode">Raw</button>
      <span class="tip"><b>Raw/Subtract/Ratio</b> — click to cycle.</span></span>
    <span class="sep"></span>
    <span data-tip><button id="btn-gain-down">-</button><span class="tip">Decrease display gain.</span></span>
    <span id="gain-display" style="font-size:10px;color:#888;min-width:50px;text-align:center">Gain: 0</span>
    <span data-tip><button id="btn-gain-up">+</button><span class="tip">Increase display gain.</span></span>
    <span data-tip><button id="btn-pause">Pause</button>
      <span class="tip">Freeze the display.</span></span>
    <span id="connection-status" class="disconnected">connecting...</span>
    <span id="baseline-status" style="font-size:10px;color:#555">no baseline</span>
  </div>
</div>

<div id="main">
  <div id="spectrogram-container">
    <canvas id="spectrogram"></canvas>
    <div id="freq-overlay"></div>
    <div id="cursor-info"></div>
    <div id="zone-overlay" class="no-match" style="display:none">
      <div id="zone-header">
        <div id="zone-best">Listening...</div>
        <div id="zone-fps"></div>
      </div>
      <div id="zone-bars"></div>
    </div>
  </div>
  <div id="bottom-panel">
    <div id="spectrum-chart">
      <canvas id="spectrum-canvas"></canvas>
    </div>
    <div id="peaks-panel">
      <h3>Live Peaks</h3>
      <div id="peaks-list"></div>
    </div>
    <div id="room-panel">
      <h3>Room Signatures</h3>
      <div class="room-capture">
        <input id="room-name" type="text" placeholder="Room name (e.g. kitchen)" maxlength="40">
        <div class="room-actions">
          <button id="btn-capture" onclick="toggleCapture()">Record Room</button>
          <button id="btn-save" onclick="saveZone()" disabled>Save</button>
        </div>
        <div id="capture-status"></div>
        <div id="capture-timer"></div>
      </div>
      <div class="room-actions" style="margin-bottom:8px">
        <button id="btn-detect" onclick="toggleDetect()">Start Detecting</button>
      </div>
      <div id="zones-list"></div>
    </div>
  </div>
</div>

<script>
const SAMPLE_RATE = {sample_rate};
const NYQUIST = {nyquist};
const NUM_BINS = {num_bins};
const FREQ_RES = {freq_res};

// State
let paused = false;
let baseline = null;
let baselineAccum = null;
let baselineFrames = 0;
let capturingBaseline = false;
let displayMode = 'raw';
let gainBoost = 0;
const BASELINE_TARGET = 60;

// Zoom
const ZOOM = {{
  ultra: [14000, {nyquist}],
  full:  [0, {nyquist}],
  high:  [24000, {nyquist}],
}};
let freqLo = 14000, freqHi = {nyquist};

// Canvas
const spectCanvas = document.getElementById('spectrogram');
const spectCtx = spectCanvas.getContext('2d');
const specCanvas = document.getElementById('spectrum-canvas');
const specCtx = specCanvas.getContext('2d');
const freqOverlay = document.getElementById('freq-overlay');
const cursorInfo = document.getElementById('cursor-info');
const peaksList = document.getElementById('peaks-list');
const connStatus = document.getElementById('connection-status');
const baselineStatus = document.getElementById('baseline-status');
const gainDisplay = document.getElementById('gain-display');

// Zone detection elements
const zoneOverlay = document.getElementById('zone-overlay');
const zoneBest = document.getElementById('zone-best');
const zoneFps = document.getElementById('zone-fps');
const zoneBars = document.getElementById('zone-bars');

// Controls
document.getElementById('btn-zoom-ultra').addEventListener('click', () => setZoom('ultra'));
document.getElementById('btn-zoom-full').addEventListener('click', () => setZoom('full'));
document.getElementById('btn-zoom-high').addEventListener('click', () => setZoom('high'));
document.getElementById('btn-baseline').addEventListener('click', startBaseline);
document.getElementById('btn-clear').addEventListener('click', clearBaseline);
document.getElementById('btn-mode').addEventListener('click', cycleMode);
document.getElementById('btn-gain-up').addEventListener('click', () => {{ gainBoost += 5; updateGain(); }});
document.getElementById('btn-gain-down').addEventListener('click', () => {{ gainBoost -= 5; updateGain(); }});
document.getElementById('btn-pause').addEventListener('click', () => {{
  paused = !paused;
  document.getElementById('btn-pause').textContent = paused ? 'Resume' : 'Pause';
  document.getElementById('btn-pause').classList.toggle('active', paused);
}});

function updateGain() {{
  gainBoost = Math.max(-30, Math.min(50, gainBoost));
  gainDisplay.textContent = `Gain: ${{gainBoost > 0 ? '+' : ''}}${{gainBoost}}`;
}}

function setZoom(preset) {{
  [freqLo, freqHi] = ZOOM[preset];
  ['ultra','full','high'].forEach(p => {{
    document.getElementById('btn-zoom-' + p).classList.toggle('active', p === preset);
  }});
  spectCtx.fillStyle = '#000';
  spectCtx.fillRect(0, 0, spectCanvas.width, spectCanvas.height);
  setupFreqLabels();
}}

function cycleMode() {{
  const modes = ['raw', 'subtract', 'ratio'];
  let idx = modes.indexOf(displayMode);
  idx = (idx + 1) % modes.length;
  displayMode = modes[idx];
  const btn = document.getElementById('btn-mode');
  btn.textContent = modes[idx].charAt(0).toUpperCase() + modes[idx].slice(1);
  if (modes[idx] !== 'raw' && !baseline) btn.textContent += ' (need baseline)';
}}

function startBaseline() {{
  baselineAccum = new Float32Array(NUM_BINS);
  baselineFrames = 0;
  capturingBaseline = true;
  document.getElementById('btn-baseline').textContent = 'Capturing...';
  document.getElementById('btn-baseline').classList.add('recording');
}}

function finishBaseline() {{
  capturingBaseline = false;
  baseline = new Float32Array(NUM_BINS);
  for (let i = 0; i < NUM_BINS; i++) baseline[i] = baselineAccum[i] / baselineFrames;
  document.getElementById('btn-baseline').textContent = 'Capture Baseline';
  document.getElementById('btn-baseline').classList.remove('recording');
  document.getElementById('btn-clear').disabled = false;
  baselineStatus.textContent = `baseline OK (${{baselineFrames}}f)`;
  baselineStatus.style.color = '#44ff44';
  displayMode = 'subtract';
  document.getElementById('btn-mode').textContent = 'Subtract';
  spectCtx.fillStyle = '#000';
  spectCtx.fillRect(0, 0, spectCanvas.width, spectCanvas.height);
}}

function clearBaseline() {{
  baseline = null;
  displayMode = 'raw';
  document.getElementById('btn-mode').textContent = 'Raw';
  document.getElementById('btn-clear').disabled = true;
  baselineStatus.textContent = 'no baseline';
  baselineStatus.style.color = '#555';
}}

// ── Room Signature Capture & Detection ──────────────────────

let isCapturing = false;
let captureStartTime = null;
let captureTimerInterval = null;
let isDetecting = false;
let zoneEventSource = null;

async function toggleCapture() {{
  const btn = document.getElementById('btn-capture');
  const status = document.getElementById('capture-status');
  const timer = document.getElementById('capture-timer');

  if (!isCapturing) {{
    // Start capture
    const resp = await fetch('/zone/capture/start', {{ method: 'POST' }});
    if (resp.ok) {{
      isCapturing = true;
      captureStartTime = Date.now();
      btn.textContent = 'Stop Recording';
      btn.classList.add('recording');
      document.getElementById('btn-save').disabled = true;
      status.textContent = 'Recording room audio...';
      status.style.color = '#ff6600';
      captureTimerInterval = setInterval(() => {{
        const secs = ((Date.now() - captureStartTime) / 1000).toFixed(1);
        timer.textContent = secs + 's';
      }}, 100);
    }}
  }} else {{
    // Stop capture
    const resp = await fetch('/zone/capture/stop', {{ method: 'POST' }});
    if (resp.ok) {{
      const data = await resp.json();
      isCapturing = false;
      clearInterval(captureTimerInterval);
      btn.textContent = 'Record Room';
      btn.classList.remove('recording');
      document.getElementById('btn-save').disabled = false;
      status.textContent = `Captured ${{data.duration_secs}}s of audio`;
      status.style.color = '#44ff44';
      timer.textContent = '';
    }}
  }}
}}

async function saveZone() {{
  const nameInput = document.getElementById('room-name');
  const name = nameInput.value.trim();
  const status = document.getElementById('capture-status');

  if (!name) {{
    status.textContent = 'Enter a room name first';
    status.style.color = '#ff4444';
    nameInput.focus();
    return;
  }}

  status.textContent = 'Saving...';
  status.style.color = '#888';
  document.getElementById('btn-save').disabled = true;

  const resp = await fetch('/zone/save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ name }})
  }});

  const data = await resp.json();
  if (data.error) {{
    status.textContent = data.error;
    status.style.color = '#ff4444';
    document.getElementById('btn-save').disabled = false;
  }} else {{
    status.textContent = `Saved "${{name}}" (${{data.fingerprints}} fingerprints)`;
    status.style.color = '#44ff44';
    nameInput.value = '';
    loadZones();
  }}
}}

async function deleteZone(name) {{
  const resp = await fetch('/zone/delete/' + encodeURIComponent(name), {{ method: 'POST' }});
  if (resp.ok) loadZones();
}}

async function loadZones() {{
  try {{
    const resp = await fetch('/zones');
    const data = await resp.json();
    const list = document.getElementById('zones-list');
    if (!data.zones || data.zones.length === 0) {{
      list.innerHTML = '<div style="color:#444;font-size:10px;padding:4px 0">No rooms saved yet</div>';
      return;
    }}
    list.innerHTML = data.zones.map(z => `
      <div class="zone-item">
        <div>
          <div class="zone-item-name">${{z.name}}</div>
          <div class="zone-item-meta">${{z.fingerprints}} fps · ${{z.duration_secs}}s</div>
        </div>
        <button onclick="deleteZone('${{z.name.replace(/'/g, "\\\\'")}}')">x</button>
      </div>
    `).join('');
  }} catch (e) {{
    // DB may not exist yet
  }}
}}

async function toggleDetect() {{
  const btn = document.getElementById('btn-detect');

  if (!isDetecting) {{
    const resp = await fetch('/zone/detect/start', {{ method: 'POST' }});
    if (resp.ok) {{
      isDetecting = true;
      btn.textContent = 'Stop Detecting';
      btn.classList.add('detecting');
      zoneOverlay.style.display = 'block';
      zoneBest.textContent = 'Listening...';
      zoneFps.textContent = '';
      zoneBars.innerHTML = '';
      zoneOverlay.className = 'no-match';

      // Subscribe to continuous detection events
      zoneEventSource = new EventSource('/zone/events');
      zoneEventSource.onmessage = (ev) => {{
        try {{
          const data = JSON.parse(ev.data);
          const matches = data.matches || [];

          zoneFps.textContent = `${{data.fingerprints || 0}} fps`;

          if (matches.length > 0) {{
            // Best match
            const best = matches[0];
            const bestPct = (best.confidence * 100).toFixed(1);
            zoneBest.textContent = best.zone;
            zoneOverlay.className = 'detected';

            // Build probability bars for all matches
            let html = '';
            for (let i = 0; i < matches.length; i++) {{
              const m = matches[i];
              const pct = (m.confidence * 100).toFixed(1);
              const barWidth = Math.min(100, m.confidence * 100);
              const isBest = i === 0;
              html += `<div class="zone-bar-row">
                <div class="zone-bar-name">${{m.zone}}</div>
                <div class="zone-bar-track">
                  <div class="zone-bar-fill ${{isBest ? 'best' : 'other'}}" style="width:${{barWidth}}%"></div>
                </div>
                <div class="zone-bar-pct ${{isBest ? 'best' : ''}}">${{pct}}%</div>
                <div class="zone-bar-matches">${{m.match_count}}</div>
              </div>`;
            }}
            zoneBars.innerHTML = html;
          }} else {{
            zoneBest.textContent = data.message || 'No match';
            zoneOverlay.className = 'no-match';
            zoneBars.innerHTML = '';
          }}
        }} catch (e) {{}}
      }};
    }}
  }} else {{
    if (zoneEventSource) {{ zoneEventSource.close(); zoneEventSource = null; }}
    const resp = await fetch('/zone/detect/stop', {{ method: 'POST' }});
    isDetecting = false;
    btn.textContent = 'Start Detecting';
    btn.classList.remove('detecting');
    zoneOverlay.style.display = 'none';
  }}
}}

// Load zones on startup
loadZones();

// ── Freq labels ──
function setupFreqLabels() {{
  freqOverlay.innerHTML = '';
  const range = freqHi - freqLo;
  let interval;
  if (range <= 12000) interval = 1000;
  else if (range <= 25000) interval = 2000;
  else interval = 5000;
  const first = Math.ceil(freqLo / interval) * interval;
  for (let f = first; f <= freqHi; f += interval) {{
    const pct = ((f - freqLo) / range) * 100;
    const label = document.createElement('div');
    label.className = 'freq-label';
    label.style.bottom = pct + '%';
    label.textContent = (f/1000).toFixed(f % 1000 === 0 ? 0 : 1) + 'k';
    freqOverlay.appendChild(label);
    const line = document.createElement('div');
    line.className = 'freq-line';
    line.style.bottom = pct + '%';
    freqOverlay.appendChild(line);
  }}
  if (freqLo < 18000 && freqHi > 18000) {{
    const pct = ((18000 - freqLo) / range) * 100;
    const line = document.createElement('div');
    line.className = 'freq-line ultra';
    line.style.bottom = pct + '%';
    freqOverlay.appendChild(line);
  }}
}}

// Mouse hover
spectCanvas.addEventListener('mousemove', (e) => {{
  const rect = spectCanvas.getBoundingClientRect();
  const pct = 1 - ((e.clientY - rect.top) / rect.height);
  const freq = freqLo + pct * (freqHi - freqLo);
  cursorInfo.style.display = 'block';
  cursorInfo.textContent = freq.toFixed(0) + ' Hz (' + (freq/1000).toFixed(1) + ' kHz)';
}});
spectCanvas.addEventListener('mouseleave', () => {{ cursorInfo.style.display = 'none'; }});

// Canvas resize
function resizeCanvases() {{
  const sr = spectCanvas.parentElement.getBoundingClientRect();
  spectCanvas.width = sr.width;
  spectCanvas.height = sr.height;
  const br = specCanvas.parentElement.getBoundingClientRect();
  specCanvas.width = br.width;
  specCanvas.height = br.height;
}}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();
setupFreqLabels();

// Color maps
function heatColor(t) {{
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.15) {{ const s=t/0.15; r=0; g=0; b=20+180*s|0; }}
  else if (t < 0.35) {{ const s=(t-0.15)/0.2; r=0; g=160*s|0; b=180; }}
  else if (t < 0.55) {{ const s=(t-0.35)/0.2; r=220*s|0; g=160+95*s|0; b=180*(1-s)|0; }}
  else if (t < 0.75) {{ const s=(t-0.55)/0.2; r=220+35*s|0; g=255; b=60*s|0; }}
  else {{ const s=(t-0.75)/0.25; r=255; g=255; b=60+195*s|0; }}
  return (r<<16)|(g<<8)|b;
}}

function diffColor(db) {{
  let r,g,b;
  if (db < -3) {{ const t=Math.min(1,Math.abs(db+3)/15); r=8; g=8; b=30+120*t|0; }}
  else if (db < 0) {{ const t=(db+3)/3; r=8+12*t|0; g=8+12*t|0; b=130-100*t|0; }}
  else if (db < 6) {{ const t=db/6; r=220*t|0; g=70*t|0; b=0; }}
  else if (db < 15) {{ const t=(db-6)/9; r=220+35*t|0; g=70+185*t|0; b=20*t|0; }}
  else {{ const t=Math.min(1,(db-15)/15); r=255; g=255; b=20+235*t|0; }}
  return (r<<16)|(g<<8)|b;
}}

function identifySource(freq) {{
  const s = [];
  if (freq > 15000 && freq < 22000) s.push('coil whine');
  if (freq > 18000 && freq < 22500) s.push('SMPS');
  if (freq > 19000 && freq < 26000) s.push('LCD PWM');
  if (freq > 15500 && freq < 16500) s.push('LED driver');
  if (freq > 22000 && freq < 28000) s.push('pest repeller');
  if (freq > 30000 && freq < 45000) s.push('piezo/appliance');
  if (Math.abs(freq - 32768) < 200) s.push('32k crystal');
  if (freq > 25000 && freq < 48000) s.push('structural');
  return s.length ? s.join(' / ') : '';
}}

// Base64 decode
function b64decode(str) {{
  const bin = atob(str);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}}

// ── Data layer: SSE receives frames ──
let latestFrame = null;
let newFrameReady = false;
let frameCount = 0;

function connect() {{
  const es = new EventSource('/stream');
  es.onopen = () => {{
    connStatus.textContent = 'streaming';
    connStatus.className = 'connected';
  }};
  es.onerror = () => {{
    connStatus.textContent = 'reconnecting...';
    connStatus.className = 'disconnected';
  }};
  let dbData = null;

  es.onmessage = (ev) => {{
    const raw = b64decode(ev.data);
    const view = new DataView(raw.buffer);
    const numBins = view.getUint16(0, true);
    if (!dbData || dbData.length !== numBins) dbData = new Float32Array(numBins);
    for (let i = 0; i < numBins; i++) {{
      dbData[i] = view.getFloat32(6 + i * 4, true);
    }}
    if (capturingBaseline) {{
      for (let i = 0; i < numBins; i++) {{
        baselineAccum[i] += Math.pow(10, dbData[i] / 20);
      }}
      baselineFrames++;
      if (baselineFrames >= BASELINE_TARGET) finishBaseline();
    }}
    latestFrame = dbData;
    newFrameReady = true;
  }};
}}

// ── Render loop ──
// Decoupled from SSE: always scroll the waterfall at rAF rate for smooth
// animation. SSE updates latestFrame asynchronously; render uses whatever
// data is current. This prevents jitter from SSE timing mismatches.
function render() {{
  requestAnimationFrame(render);
  if (paused || !latestFrame) return;

  const dbData = latestFrame;
  const numBins = dbData.length;

  const binLo = Math.max(0, Math.floor(freqLo / FREQ_RES));
  const binHi = Math.min(numBins - 1, Math.ceil(freqHi / FREQ_RES));
  const numVisible = binHi - binLo + 1;
  const displayDB = new Float32Array(numVisible);

  const isSub = baseline && displayMode === 'subtract';
  const isRatio = baseline && displayMode === 'ratio';
  const isProcessed = isSub || isRatio;

  for (let j = 0; j < numVisible; j++) {{
    const i = binLo + j;
    let db;
    if (isSub) {{
      const mag = Math.pow(10, dbData[i] / 20);
      const diff = mag - baseline[i];
      db = diff > 1e-15 ? 20 * Math.log10(diff) : -150;
    }} else if (isRatio) {{
      const mag = Math.pow(10, dbData[i] / 20);
      const ratio = baseline[i] > 1e-15 ? mag / baseline[i] : 1;
      db = 20 * Math.log10(Math.max(ratio, 1e-8));
    }} else {{
      db = dbData[i];
    }}
    displayDB[j] = db + gainBoost;
  }}

  // Scrolling spectrogram
  const sw = spectCanvas.width, sh = spectCanvas.height;
  const dbMin = isProcessed ? -15 + gainBoost : -120 + gainBoost;
  const dbMax = isProcessed ? 25 + gainBoost : -25 + gainBoost;
  const dbRange = dbMax - dbMin;

  const shift = 2;
  spectCtx.drawImage(spectCanvas, shift, 0, sw - shift, sh, 0, 0, sw - shift, sh);
  const col = spectCtx.createImageData(shift, sh);
  const px = col.data;
  for (let py = 0; py < sh; py++) {{
    const fPct = 1 - (py / sh);
    const j = Math.round(fPct * (numVisible - 1));
    if (j < 0 || j >= numVisible) continue;
    const db = displayDB[j];
    const t = (db - dbMin) / dbRange;
    let rgb;
    if (isProcessed) rgb = diffColor(db - gainBoost);
    else rgb = heatColor(t);
    const r = (rgb>>16)&0xff, g = (rgb>>8)&0xff, b = rgb&0xff;
    for (let sx = 0; sx < shift; sx++) {{
      const idx = (py * shift + sx) * 4;
      px[idx] = r; px[idx+1] = g; px[idx+2] = b; px[idx+3] = 255;
    }}
  }}
  spectCtx.putImageData(col, sw - shift, 0);

  // Spectrum chart
  const cw = specCanvas.width, ch = specCanvas.height;
  const margin = {{ l: 45, r: 10, t: 8, b: 22 }};
  const pw = cw - margin.l - margin.r, ph = ch - margin.t - margin.b;
  const cDbMin = isProcessed ? -15 : -120;
  const cDbMax = isProcessed ? 30 : -20;

  const gridKey = `${{cw}},${{ch}},${{freqLo}},${{freqHi}},${{isProcessed}}`;
  if (gridKey !== render._gridKey) {{
    render._gridKey = gridKey;
    if (!render._gridCanvas) render._gridCanvas = document.createElement('canvas');
    render._gridCanvas.width = cw;
    render._gridCanvas.height = ch;
    const gc = render._gridCanvas.getContext('2d');
    gc.fillStyle = '#0a0a0a';
    gc.fillRect(0, 0, cw, ch);
    gc.strokeStyle = '#1a1a1a'; gc.lineWidth = 0.5;
    gc.fillStyle = '#444'; gc.font = '9px monospace';
    gc.textAlign = 'right';
    const dbStep = isProcessed ? 5 : 10;
    for (let db = cDbMin; db <= cDbMax; db += dbStep) {{
      const y = margin.t + ph - ((db - cDbMin) / (cDbMax - cDbMin)) * ph;
      gc.beginPath(); gc.moveTo(margin.l, y); gc.lineTo(cw-margin.r, y); gc.stroke();
      gc.fillText(db.toString(), margin.l - 4, y + 3);
    }}
    gc.textAlign = 'center';
    const fRange = freqHi - freqLo;
    let fInt; if (fRange <= 12000) fInt = 1000; else if (fRange <= 25000) fInt = 2000; else fInt = 5000;
    for (let f = Math.ceil(freqLo/fInt)*fInt; f <= freqHi; f += fInt) {{
      const x = margin.l + ((f - freqLo) / fRange) * pw;
      gc.strokeStyle = '#1a1a1a'; gc.beginPath();
      gc.moveTo(x, margin.t); gc.lineTo(x, margin.t + ph); gc.stroke();
      gc.fillStyle = '#555';
      gc.fillText((f/1000).toFixed(f%1000===0?0:1)+'k', x, ch - 4);
    }}
    if (freqLo < 18000 && freqHi > 18000) {{
      const x18 = margin.l + ((18000 - freqLo) / fRange) * pw;
      gc.strokeStyle = '#00ccff33'; gc.lineWidth = 1.5;
      gc.beginPath(); gc.moveTo(x18, margin.t); gc.lineTo(x18, margin.t+ph); gc.stroke();
    }}
    if (isProcessed) {{
      const y0 = margin.t + ph - ((0 - cDbMin) / (cDbMax - cDbMin)) * ph;
      gc.strokeStyle = '#555'; gc.lineWidth = 1;
      gc.setLineDash([4,4]); gc.beginPath();
      gc.moveTo(margin.l, y0); gc.lineTo(cw-margin.r, y0); gc.stroke();
      gc.setLineDash([]);
    }}
  }}
  specCtx.drawImage(render._gridCanvas, 0, 0);
  const fRange = freqHi - freqLo;

  // Frequency bars
  const step = Math.max(1, Math.floor(numVisible / pw));
  const barW = Math.max(1, pw / (numVisible / step) - 0.5);
  const yBase = margin.t + ph;
  for (let j = 0; j < numVisible; j += step) {{
    const freq = (binLo + j) * FREQ_RES;
    const x = margin.l + ((freq - freqLo) / fRange) * pw;
    const db = Math.max(cDbMin, Math.min(cDbMax, displayDB[j]));
    const t = (db - cDbMin) / (cDbMax - cDbMin);
    const barH = t * ph;
    let r, g, b;
    if (isProcessed) {{
      if (db < 0) {{ r = 40; g = 40; b = 80; }}
      else if (db < 10) {{ const s = db / 10; r = 60 + 195*s|0; g = 40 + 30*s|0; b = 0; }}
      else {{ const s = Math.min(1, (db - 10) / 20); r = 255; g = 70 + 185*s|0; b = 50*s|0; }}
    }} else {{
      if (t < 0.3) {{ const s = t / 0.3; r = 0; g = 60*s|0; b = 100 + 155*s|0; }}
      else if (t < 0.6) {{ const s = (t-0.3) / 0.3; r = 0; g = 60 + 140*s|0; b = 255 - 55*s|0; }}
      else {{ const s = (t-0.6) / 0.4; r = 200*s|0; g = 200 + 55*s|0; b = 200 - 200*s|0; }}
    }}
    specCtx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
    specCtx.fillRect(x, yBase - barH, barW, barH);
  }}

  // Peaks
  if (frameCount % 8 === 0) {{
    const thresh = isProcessed ? -2 : -80;
    const peaks = [];
    for (let j = 2; j < numVisible - 2; j++) {{
      if (displayDB[j] > displayDB[j-1] && displayDB[j] > displayDB[j+1] &&
          displayDB[j] > displayDB[j-2] && displayDB[j] > displayDB[j+2] &&
          displayDB[j] > thresh) {{
        peaks.push({{ freq: (binLo + j) * FREQ_RES, db: displayDB[j] }});
      }}
    }}
    peaks.sort((a,b) => b.db - a.db);
    let html = '';
    for (const p of peaks.slice(0, 10)) {{
      const src = identifySource(p.freq);
      const dbVal = p.db - gainBoost;
      const dbStr = isProcessed ? (dbVal>0?'+':'')+dbVal.toFixed(1)+' dB' : dbVal.toFixed(1)+' dB';
      html += `<div class="peak-row"><span class="peak-freq">${{p.freq.toFixed(0)}} Hz</span><span class="peak-db">${{dbStr}}</span></div>`;
      if (src) html += `<div class="peak-source">${{src}}</div>`;
    }}
    if (!peaks.length) html = '<div style="color:#444;padding:10px 0">No peaks above threshold</div>';
    peaksList.innerHTML = html;
  }}
  frameCount++;
}}

connect();
requestAnimationFrame(render);
</script>
</body></html>"##,
        sample_rate = sample_rate,
        nyquist = nyquist,
        nyquist_k = nyquist / 1000,
        num_bins = NUM_BINS,
        freq_res = freq_res,
        device_name_esc = device_name.replace('<', "&lt;").replace('>', "&gt;"),
    )
}
