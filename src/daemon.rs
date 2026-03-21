//! Daemon module: background service that listens for recognition and zone
//! detection requests via TCP.

use crate::audio::AudioSignal;
use crate::database::Database;
use crate::fingerprint::Fingerprinter;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// Default daemon port.
pub const DEFAULT_PORT: u16 = 18923;

/// Default database path.
pub fn default_db_path() -> PathBuf {
    dirs_or_default().join("beacond.db")
}

/// Default socket/config directory.
fn dirs_or_default() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let dir = PathBuf::from(home).join(".beacond");
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Default PID file path.
pub fn pid_file_path() -> PathBuf {
    dirs_or_default().join("beacond.pid")
}

/// Request types the daemon understands.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DaemonRequest {
    /// Recognize audio from a WAV file path.
    #[serde(rename = "recognize")]
    Recognize { path: String },

    /// Ingest a new track into the database.
    #[serde(rename = "ingest")]
    Ingest {
        path: String,
        title: String,
        artist: String,
        album: Option<String>,
    },

    /// Remove a track from the database.
    #[serde(rename = "remove")]
    Remove { track_id: i64 },

    /// List all tracks.
    #[serde(rename = "list")]
    List,

    /// Get database stats.
    #[serde(rename = "stats")]
    Stats,

    /// Shutdown the daemon.
    #[serde(rename = "shutdown")]
    Shutdown,

    /// Ping to check if daemon is alive.
    #[serde(rename = "ping")]
    Ping,

    /// Register a named zone from a beacon WAV or mic recording.
    #[serde(rename = "zone_add")]
    ZoneAdd {
        name: String,
        path: Option<String>,
        frequency_mode: Option<String>,
        record_duration: Option<u64>,
    },

    /// List all registered zones.
    #[serde(rename = "zone_list")]
    ZoneList,

    /// Remove a zone and its beacon fingerprints.
    #[serde(rename = "zone_remove")]
    ZoneRemove { name: String },

    /// Detect which zone a WAV clip belongs to.
    #[serde(rename = "zone_detect")]
    ZoneDetect {
        path: String,
        frequency_mode: Option<String>,
    },

    /// Subscribe to live zone transition events.
    #[serde(rename = "monitor_subscribe")]
    MonitorSubscribe,
}

/// Response from the daemon.
#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonResponse {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl DaemonResponse {
    pub fn ok(message: impl Into<String>) -> Self {
        DaemonResponse {
            success: true,
            message: message.into(),
            data: None,
        }
    }

    pub fn ok_with_data(message: impl Into<String>, data: serde_json::Value) -> Self {
        DaemonResponse {
            success: true,
            message: message.into(),
            data: Some(data),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        DaemonResponse {
            success: false,
            message: message.into(),
            data: None,
        }
    }
}

/// A zone transition event emitted by the monitor loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneEvent {
    pub event: String,
    pub zone: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    pub timestamp: String,
}

/// Shared state for the daemon.
struct DaemonState {
    db: Database,
    fingerprinter: Fingerprinter,
    /// Optional ML embedding model for learned audio signatures.
    #[cfg(feature = "ml-embeddings")]
    embedder: Option<crate::embeddings::AudioEmbedder>,
}

/// Compute a signature from audio, using ML embeddings if available, falling back to spectral.
/// The spectrogram is only computed if needed (no embedder, or embedding fails).
fn compute_signature(
    state: &DaemonState,
    samples: &[f32],
    sample_rate: u32,
) -> (crate::signature::Signature, crate::spectrogram::Spectrogram) {
    #[cfg(feature = "ml-embeddings")]
    if let Some(ref embedder) = state.embedder {
        match embedder.embed(samples, sample_rate) {
            Ok(vec) => {
                // Spectrogram still needed for fingerprinting, compute it.
                let spectrogram = crate::spectrogram::Spectrogram::compute(
                    samples,
                    sample_rate,
                    &state.fingerprinter.config.spectrogram,
                );
                return (
                    crate::signature::Signature::from_embedding(vec),
                    spectrogram,
                );
            }
            Err(e) => {
                log::warn!("Embedding failed, falling back to spectral: {}", e);
            }
        }
    }

    let spectrogram = crate::spectrogram::Spectrogram::compute(
        samples,
        sample_rate,
        &state.fingerprinter.config.spectrogram,
    );
    let sig = crate::signature::Signature::from_spectrogram(&spectrogram);
    (sig, spectrogram)
}

/// Broadcast channel for zone events (separate from mutex-guarded state).
type EventTx = tokio::sync::broadcast::Sender<ZoneEvent>;

/// Run the daemon server.
pub async fn run_daemon(
    db_path: &Path,
    port: u16,
    bind: &str,
    monitor: bool,
    _model_path: Option<&Path>,
) -> Result<()> {
    let db = Database::open(db_path)
        .with_context(|| format!("Failed to open database at {}", db_path.display()))?;

    #[cfg(feature = "ml-embeddings")]
    let embedder = _model_path
        .map(|p| {
            crate::embeddings::AudioEmbedder::load(p)
                .with_context(|| format!("Failed to load embedding model from {}", p.display()))
        })
        .transpose()?;

    #[cfg(feature = "ml-embeddings")]
    if embedder.is_some() {
        log::info!("ML embedding model loaded — using learned signatures");
    }

    let state = Arc::new(Mutex::new(DaemonState {
        db,
        fingerprinter: Fingerprinter::with_defaults(),
        #[cfg(feature = "ml-embeddings")]
        embedder,
    }));

    let (event_tx, _) = tokio::sync::broadcast::channel::<ZoneEvent>(64);

    // Write PID file
    let pid = std::process::id();
    std::fs::write(pid_file_path(), pid.to_string())?;

    let addr = format!("{}:{}", bind, port);
    let listener = TcpListener::bind(&addr).await?;

    log::info!("Beacon daemon listening on {}", addr);
    println!("📡 Beacon daemon started on {}", addr);

    if monitor {
        println!("📡 Monitor mode active — scanning for zones via microphone");
        let monitor_state = Arc::clone(&state);
        let monitor_tx = event_tx.clone();
        tokio::spawn(async move {
            monitor_loop(monitor_state, monitor_tx).await;
        });
    }

    let shutdown = Arc::new(tokio::sync::Notify::new());

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, addr)) => {
                        log::debug!("New connection from {}", addr);
                        let state = Arc::clone(&state);
                        let shutdown = Arc::clone(&shutdown);
                        let event_tx = event_tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, state, shutdown, event_tx).await {
                                log::error!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        log::error!("Accept error: {}", e);
                    }
                }
            }
            _ = shutdown.notified() => {
                log::info!("Daemon shutting down...");
                println!("🛑 Daemon shutting down");
                // Clean up PID file
                std::fs::remove_file(pid_file_path()).ok();
                break;
            }
        }
    }

    Ok(())
}

/// Handle a single client connection.
async fn handle_connection(
    stream: TcpStream,
    state: Arc<Mutex<DaemonState>>,
    shutdown: Arc<tokio::sync::Notify>,
    event_tx: EventTx,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    while reader.read_line(&mut line).await? > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        let response = match serde_json::from_str::<DaemonRequest>(trimmed) {
            Ok(request) => {
                let is_monitor = matches!(request, DaemonRequest::MonitorSubscribe);
                let should_shutdown = matches!(request, DaemonRequest::Shutdown);
                let resp = process_request(request, &state);

                if is_monitor {
                    // Send initial ack, then switch to streaming mode
                    let json = serde_json::to_string(&resp)? + "\n";
                    writer.write_all(json.as_bytes()).await?;
                    writer.flush().await?;

                    // Stream events until client disconnects
                    let mut rx = event_tx.subscribe();
                    loop {
                        match rx.recv().await {
                            Ok(event) => {
                                let json = serde_json::to_string(&event).unwrap_or_default() + "\n";
                                if writer.write_all(json.as_bytes()).await.is_err() {
                                    break;
                                }
                                if writer.flush().await.is_err() {
                                    break;
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                log::warn!("Monitor client lagged, skipped {} events", n);
                            }
                            Err(_) => break,
                        }
                    }
                    return Ok(());
                }

                if should_shutdown {
                    let json = serde_json::to_string(&resp)? + "\n";
                    writer.write_all(json.as_bytes()).await?;
                    writer.flush().await?;
                    shutdown.notify_one();
                    return Ok(());
                }
                resp
            }
            Err(e) => DaemonResponse::error(format!("Invalid request: {}", e)),
        };

        let json = serde_json::to_string(&response)? + "\n";
        writer.write_all(json.as_bytes()).await?;
        writer.flush().await?;

        line.clear();
    }

    Ok(())
}

/// Process a single daemon request (runs synchronously with database lock).
fn process_request(request: DaemonRequest, state: &Arc<Mutex<DaemonState>>) -> DaemonResponse {
    match request {
        DaemonRequest::Ping => DaemonResponse::ok("pong"),

        DaemonRequest::Recognize { path } => match AudioSignal::from_wav(&path) {
            Ok(signal) => {
                let state = state.lock().unwrap();
                let fingerprints = state
                    .fingerprinter
                    .fingerprint(&signal.samples, signal.sample_rate);

                if fingerprints.is_empty() {
                    return DaemonResponse::error("No fingerprints could be extracted from audio");
                }

                match state
                    .db
                    .search(&fingerprints, 5, crate::audio::TARGET_SAMPLE_RATE)
                {
                    Ok(results) => {
                        if results.is_empty() {
                            DaemonResponse::ok_with_data(
                                "No matches found",
                                serde_json::json!({ "matches": [] }),
                            )
                        } else {
                            DaemonResponse::ok_with_data(
                                format!("Found {} match(es)", results.len()),
                                serde_json::to_value(&results).unwrap(),
                            )
                        }
                    }
                    Err(e) => DaemonResponse::error(format!("Search failed: {}", e)),
                }
            }
            Err(e) => DaemonResponse::error(format!("Failed to read audio: {}", e)),
        },

        DaemonRequest::Ingest {
            path,
            title,
            artist,
            album,
        } => match AudioSignal::from_wav(&path) {
            Ok(signal) => {
                let mut state = state.lock().unwrap();
                let fingerprints = state
                    .fingerprinter
                    .fingerprint(&signal.samples, signal.sample_rate);

                if fingerprints.is_empty() {
                    return DaemonResponse::error("No fingerprints could be extracted from audio");
                }

                let fp_count = fingerprints.len();
                match state.db.insert_track(
                    &title,
                    &artist,
                    album.as_deref(),
                    signal.duration_secs,
                    Some(&path),
                    &fingerprints,
                ) {
                    Ok(id) => DaemonResponse::ok_with_data(
                        format!(
                            "Ingested '{}' by '{}' ({} fingerprints)",
                            title, artist, fp_count
                        ),
                        serde_json::json!({
                            "track_id": id,
                            "fingerprint_count": fp_count,
                            "duration_secs": signal.duration_secs
                        }),
                    ),
                    Err(e) => DaemonResponse::error(format!("Ingest failed: {}", e)),
                }
            }
            Err(e) => DaemonResponse::error(format!("Failed to read audio: {}", e)),
        },

        DaemonRequest::Remove { track_id } => {
            let mut state = state.lock().unwrap();
            match state.db.remove_track(track_id) {
                Ok(true) => DaemonResponse::ok(format!("Removed track {}", track_id)),
                Ok(false) => DaemonResponse::error(format!("Track {} not found", track_id)),
                Err(e) => DaemonResponse::error(format!("Remove failed: {}", e)),
            }
        }

        DaemonRequest::List => {
            let state = state.lock().unwrap();
            match state.db.list_tracks() {
                Ok(tracks) => DaemonResponse::ok_with_data(
                    format!("{} track(s) in database", tracks.len()),
                    serde_json::to_value(&tracks).unwrap(),
                ),
                Err(e) => DaemonResponse::error(format!("List failed: {}", e)),
            }
        }

        DaemonRequest::Stats => {
            let state = state.lock().unwrap();
            match state.db.stats() {
                Ok(stats) => DaemonResponse::ok_with_data(
                    "Database statistics",
                    serde_json::to_value(&stats).unwrap(),
                ),
                Err(e) => DaemonResponse::error(format!("Stats failed: {}", e)),
            }
        }

        DaemonRequest::Shutdown => {
            // Handled above in handle_connection
            DaemonResponse::ok("Shutting down...")
        }

        DaemonRequest::ZoneAdd {
            name,
            path,
            frequency_mode,
            record_duration,
        } => {
            let mode = frequency_mode
                .as_deref()
                .and_then(crate::audio::FrequencyMode::parse)
                .unwrap_or(crate::audio::FrequencyMode::Ultrasonic);

            let signal = if let Some(wav_path) = &path {
                AudioSignal::from_wav_with_mode(wav_path, mode)
            } else {
                let dur_secs = record_duration.unwrap_or(10);
                crate::microphone::record_from_mic(std::time::Duration::from_secs(dur_secs), mode)
            };

            match signal {
                Ok(signal) => {
                    let mut state = state.lock().unwrap();
                    let (sig, spectrogram) =
                        compute_signature(&state, &signal.samples, signal.sample_rate);

                    let sig_bytes = sig.to_bytes();
                    let fingerprints = state.fingerprinter.fingerprint_spectrogram(&spectrogram);

                    if fingerprints.is_empty() {
                        return DaemonResponse::error(
                            "No fingerprints could be extracted from audio",
                        );
                    }

                    let fp_count = fingerprints.len();
                    let sig_dim = sig.vector.len();
                    match state.db.insert_track(
                        &name,
                        "zone",
                        None,
                        signal.duration_secs,
                        path.as_deref(),
                        &fingerprints,
                    ) {
                        Ok(track_id) => {
                            match state.db.insert_zone(&name, track_id, mode.as_str(), Some(&sig_bytes)) {
                                Ok(zone_id) => DaemonResponse::ok_with_data(
                                    format!(
                                        "Zone '{}' registered ({} fingerprints, {}-dim signature, {})",
                                        name,
                                        fp_count,
                                        sig_dim,
                                        mode.as_str()
                                    ),
                                    serde_json::json!({
                                        "zone_id": zone_id,
                                        "track_id": track_id,
                                        "fingerprint_count": fp_count,
                                        "signature_dim": sig_dim,
                                        "frequency_mode": mode.as_str(),
                                    }),
                                ),
                                Err(e) => {
                                    DaemonResponse::error(format!("Failed to create zone: {}", e))
                                }
                            }
                        }
                        Err(e) => DaemonResponse::error(format!("Ingest failed: {}", e)),
                    }
                }
                Err(e) => DaemonResponse::error(format!("Failed to process audio: {}", e)),
            }
        }

        DaemonRequest::ZoneList => {
            let state = state.lock().unwrap();
            match state.db.list_zones() {
                Ok(zones) => DaemonResponse::ok_with_data(
                    format!("{} zone(s)", zones.len()),
                    serde_json::to_value(&zones).unwrap(),
                ),
                Err(e) => DaemonResponse::error(format!("Failed to list zones: {}", e)),
            }
        }

        DaemonRequest::ZoneRemove { name } => {
            let mut state = state.lock().unwrap();
            match state.db.remove_zone(&name) {
                Ok(true) => DaemonResponse::ok(format!("Zone '{}' removed", name)),
                Ok(false) => DaemonResponse::error(format!("Zone '{}' not found", name)),
                Err(e) => DaemonResponse::error(format!("Remove failed: {}", e)),
            }
        }

        DaemonRequest::ZoneDetect {
            path,
            frequency_mode,
        } => {
            let mode = frequency_mode
                .as_deref()
                .and_then(crate::audio::FrequencyMode::parse)
                .unwrap_or(crate::audio::FrequencyMode::Ultrasonic);

            match AudioSignal::from_wav_with_mode(&path, mode) {
                Ok(signal) => {
                    let state = state.lock().unwrap();

                    let mut best_detection: Option<(String, f64, &str)> = None;

                    let (query_sig, _spectrogram) =
                        compute_signature(&state, &signal.samples, signal.sample_rate);

                    // Vector similarity matching.
                    if let Ok(zone_sigs) = state.db.get_zone_signatures() {
                        let threshold = query_sig.similarity_threshold();
                        for (zone_name, sig_bytes) in &zone_sigs {
                            if let Some(stored_sig) =
                                crate::signature::Signature::from_bytes(sig_bytes)
                            {
                                if stored_sig.dim() != query_sig.dim() {
                                    continue;
                                }
                                let sim = query_sig.similarity(&stored_sig) as f64;
                                if sim > threshold
                                    && best_detection.as_ref().is_none_or(|(_, c, _)| sim > *c)
                                {
                                    best_detection = Some((zone_name.clone(), sim, "signature"));
                                }
                            }
                        }
                    }

                    // Hash-based fingerprint matching.
                    let fingerprints = state
                        .fingerprinter
                        .fingerprint(&signal.samples, signal.sample_rate);

                    if !fingerprints.is_empty() {
                        if let Ok(results) = state.db.search(&fingerprints, 3, mode.sample_rate()) {
                            if let Some(top) = results.first() {
                                if top.confidence >= 0.05 {
                                    if let Ok(Some(zone)) =
                                        state.db.get_zone_by_track_id(top.track.id)
                                    {
                                        if best_detection
                                            .as_ref()
                                            .is_none_or(|(_, c, _)| top.confidence > *c)
                                        {
                                            best_detection = Some((
                                                zone.name.clone(),
                                                top.confidence,
                                                "fingerprint",
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    match best_detection {
                        Some((zone_name, confidence, method)) => DaemonResponse::ok_with_data(
                            format!("Zone detected: {}", zone_name),
                            serde_json::json!({
                                "zone": zone_name,
                                "confidence": confidence,
                                "method": method,
                            }),
                        ),
                        None => DaemonResponse::ok_with_data(
                            "No zone detected",
                            serde_json::json!({ "zone": null }),
                        ),
                    }
                }
                Err(e) => DaemonResponse::error(format!("Failed to read audio: {}", e)),
            }
        }

        DaemonRequest::MonitorSubscribe => {
            // This is handled specially in handle_connection
            DaemonResponse::ok("Monitor mode — events will stream on this connection")
        }
    }
}

/// Send a request to the daemon and receive a response.
pub async fn send_request(port: u16, request: &DaemonRequest) -> Result<DaemonResponse> {
    let addr = format!("127.0.0.1:{}", port);
    let stream = TcpStream::connect(&addr).await.with_context(|| {
        format!(
            "Cannot connect to daemon at {}. Is it running? Start with: beacond daemon",
            addr
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    let json = serde_json::to_string(request)? + "\n";
    writer.write_all(json.as_bytes()).await?;
    writer.flush().await?;

    let mut response_line = String::new();
    reader.read_line(&mut response_line).await?;

    let response: DaemonResponse =
        serde_json::from_str(response_line.trim()).context("Failed to parse daemon response")?;

    Ok(response)
}

/// Check if daemon is running by trying to connect.
pub async fn is_daemon_running(port: u16) -> bool {
    match send_request(port, &DaemonRequest::Ping).await {
        Ok(resp) => resp.success,
        Err(_) => false,
    }
}

/// Background loop that continuously records from the microphone and emits zone events.
/// Scans each distinct frequency mode registered across zones.
async fn monitor_loop(state: Arc<Mutex<DaemonState>>, event_tx: EventTx) {
    use crate::audio::FrequencyMode;
    use crate::microphone;
    use std::time::Duration;

    let record_duration = Duration::from_secs(3);
    let pause_between = Duration::from_secs(1);
    let mut current_zone: Option<String> = None;

    loop {
        // Determine which frequency modes we need to scan.
        let modes: Vec<FrequencyMode> = {
            let state = state.lock().unwrap();
            state
                .db
                .get_distinct_zone_modes()
                .unwrap_or_default()
                .iter()
                .filter_map(|s| FrequencyMode::parse(s))
                .collect()
        };

        if modes.is_empty() {
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        // Record and search in each mode; take the best match across all modes.
        let mut best_detection: Option<(String, f64)> = None;

        for mode in modes {
            let dur = record_duration;
            let signal =
                tokio::task::spawn_blocking(move || microphone::record_from_mic(dur, mode)).await;

            match signal {
                Ok(Ok(signal)) => {
                    let (query_sig, fingerprints) = {
                        let state = state.lock().unwrap();
                        let (sig, spectrogram) =
                            compute_signature(&state, &signal.samples, signal.sample_rate);
                        let fps = state.fingerprinter.fingerprint_spectrogram(&spectrogram);
                        (sig, fps)
                    };

                    // Vector similarity matching.
                    {
                        let state = state.lock().unwrap();
                        if let Ok(zone_sigs) = state.db.get_zone_signatures() {
                            let threshold = query_sig.similarity_threshold();
                            for (zone_name, sig_bytes) in &zone_sigs {
                                if let Some(stored_sig) =
                                    crate::signature::Signature::from_bytes(sig_bytes)
                                {
                                    if stored_sig.dim() != query_sig.dim() {
                                        continue;
                                    }
                                    let sim = query_sig.similarity(&stored_sig) as f64;
                                    if sim > threshold
                                        && best_detection.as_ref().is_none_or(|(_, c)| sim > *c)
                                    {
                                        best_detection = Some((zone_name.clone(), sim));
                                    }
                                }
                            }
                        }
                    }

                    // Hash-based matching.
                    if !fingerprints.is_empty() {
                        let state = state.lock().unwrap();
                        if let Ok(results) = state.db.search(&fingerprints, 1, mode.sample_rate()) {
                            if let Some(top) = results.first() {
                                if top.confidence >= 0.05 {
                                    if let Ok(Some(zone)) =
                                        state.db.get_zone_by_track_id(top.track.id)
                                    {
                                        if best_detection
                                            .as_ref()
                                            .is_none_or(|(_, c)| top.confidence > *c)
                                        {
                                            best_detection =
                                                Some((zone.name.clone(), top.confidence));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Err(e)) => {
                    log::error!("Monitor recording error ({}): {}", mode.as_str(), e);
                }
                Err(e) => {
                    log::error!("Monitor task panic: {}", e);
                }
            }
        }

        let now = chrono::Utc::now().to_rfc3339();

        match (&current_zone, &best_detection) {
            (None, Some((zone_name, confidence))) => {
                current_zone = Some(zone_name.clone());
                let _ = event_tx.send(ZoneEvent {
                    event: "zone_enter".to_string(),
                    zone: zone_name.clone(),
                    confidence: Some(*confidence),
                    timestamp: now,
                });
                log::info!("Zone enter: {}", zone_name);
            }
            (Some(prev), Some((zone_name, confidence))) if prev != zone_name => {
                let prev_name = prev.clone();
                let exit_now = chrono::Utc::now().to_rfc3339();
                let _ = event_tx.send(ZoneEvent {
                    event: "zone_exit".to_string(),
                    zone: prev_name.clone(),
                    confidence: None,
                    timestamp: exit_now,
                });
                current_zone = Some(zone_name.clone());
                let _ = event_tx.send(ZoneEvent {
                    event: "zone_enter".to_string(),
                    zone: zone_name.clone(),
                    confidence: Some(*confidence),
                    timestamp: now,
                });
                log::info!("Zone transition: {} → {}", prev_name, zone_name);
            }
            (Some(prev), None) => {
                let _ = event_tx.send(ZoneEvent {
                    event: "zone_exit".to_string(),
                    zone: prev.clone(),
                    confidence: None,
                    timestamp: now,
                });
                log::info!("Zone exit: {}", prev);
                current_zone = None;
            }
            _ => {
                // Same zone or still no zone — no event
            }
        }

        tokio::time::sleep(pause_between).await;
    }
}
