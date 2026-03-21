//! CLI module: command-line interface for the Beacon daemon.

use crate::audio::{AudioSignal, FrequencyMode};
use crate::daemon::{self, DaemonRequest, DaemonResponse, DEFAULT_PORT};
use crate::database::{Database, MatchResult};
use crate::fingerprint::Fingerprinter;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

/// Resolve frequency mode from CLI flags.
/// Default is ultrasonic. Use --audible for standard audio range.
fn resolve_mode(audible: bool, full_spectrum: bool) -> FrequencyMode {
    if audible {
        FrequencyMode::Audible
    } else if full_spectrum {
        FrequencyMode::Full
    } else {
        FrequencyMode::Ultrasonic
    }
}

#[derive(Parser)]
#[command(
    name = "beacond",
    version,
    about = "📡 Beacon — Ultrasonic spatial awareness toolkit",
    long_about = "Ultrasonic spatial awareness toolkit for operators who control their own infrastructure.\n\n\
                  Define zones with ultrasonic beacon WAV files, detect which zone you're in, \
                  and stream real-time location events.\n\n\
                  Run 'beacond daemon' to start the background service, \
                  or use commands directly."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Daemon port
    #[arg(long, default_value_t = DEFAULT_PORT, global = true)]
    pub port: u16,

    /// Bind address (use 0.0.0.0 to listen on all interfaces)
    #[arg(long, default_value = "127.0.0.1", global = true)]
    pub bind: String,

    /// Database path (for direct mode)
    #[arg(long, global = true)]
    pub db: Option<PathBuf>,

    /// Verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Path to an ONNX audio embedding model for ML-powered zone signatures.
    /// When provided (and built with --features ml-embeddings), replaces the
    /// default spectral signatures with learned embeddings.
    #[arg(long, global = true)]
    pub model_path: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the beacon daemon
    Daemon {
        /// Run in foreground (don't daemonize)
        #[arg(long)]
        foreground: bool,

        /// Enable monitor mode: continuously scan for zones via microphone
        #[arg(long)]
        monitor: bool,
    },

    /// Ingest a WAV file into the fingerprint database
    Ingest {
        /// Path to WAV file
        path: PathBuf,

        /// Track title
        #[arg(short, long)]
        title: Option<String>,

        /// Artist name
        #[arg(short, long, default_value = "Unknown")]
        artist: String,

        /// Album name
        #[arg(long)]
        album: Option<String>,

        /// Audible mode: use standard audio range (20 Hz - 8 kHz)
        #[arg(long)]
        audible: bool,

        /// Full spectrum mode: fingerprint all frequencies up to 48 kHz
        #[arg(long)]
        full_spectrum: bool,
    },

    /// Ingest all WAV files in a directory
    IngestDir {
        /// Directory containing WAV files
        path: PathBuf,

        /// Artist name for all tracks
        #[arg(short, long, default_value = "Unknown")]
        artist: String,

        /// Audible mode: use standard audio range (20 Hz - 8 kHz)
        #[arg(long)]
        audible: bool,

        /// Full spectrum mode: fingerprint all frequencies up to 48 kHz
        #[arg(long)]
        full_spectrum: bool,
    },

    /// Recognize a WAV audio clip
    Recognize {
        /// Path to WAV file to recognize
        path: PathBuf,

        /// Audible mode: use standard audio range (20 Hz - 8 kHz)
        #[arg(long)]
        audible: bool,

        /// Full spectrum mode: recognize using all frequencies up to 48 kHz
        #[arg(long)]
        full_spectrum: bool,
    },

    /// Listen via microphone and recognize what's playing
    Listen {
        /// Recording duration in seconds
        #[arg(short, long, default_value_t = 5)]
        duration: u64,

        /// Save the recorded audio to a WAV file
        #[arg(short, long)]
        save: Option<PathBuf>,

        /// Audible mode: use standard audio range (20 Hz - 8 kHz)
        #[arg(long)]
        audible: bool,

        /// Full spectrum mode: capture and recognize all frequencies up to 48 kHz
        #[arg(long)]
        full_spectrum: bool,
    },

    /// List available audio input devices
    Devices,

    /// List all tracks in the database
    List,

    /// Remove a track from the database
    Remove {
        /// Track ID to remove
        track_id: i64,
    },

    /// Show database statistics
    Stats,

    /// Check if daemon is running
    Status,

    /// Stop the daemon
    Stop,

    /// Run a self-test / demo
    Demo,

    /// Manage named zones for spatial awareness
    Zone {
        #[command(subcommand)]
        command: ZoneCommand,
    },

    /// Subscribe to live zone transition events from the daemon
    Monitor,

    /// Launch real-time spectrogram visualizer in the browser (native 96kHz capture)
    Spectrogram {
        /// HTTP server port for the visualizer
        #[arg(long, default_value_t = 18924)]
        viz_port: u16,
    },
}

#[derive(Subcommand)]
pub enum ZoneCommand {
    /// Register a zone from a WAV file or by recording ambient audio
    Add {
        /// Zone name (e.g., "kitchen", "office", "lobby")
        name: String,

        /// Path to WAV file (omit when using --record)
        path: Option<PathBuf>,

        /// Record ambient audio from the microphone instead of using a WAV file
        #[arg(long)]
        record: bool,

        /// Recording duration in seconds (used with --record)
        #[arg(long, default_value_t = 10)]
        duration: u64,

        /// Audible mode: fingerprint standard audio range (20 Hz – 8 kHz)
        #[arg(long)]
        audible: bool,

        /// Full spectrum mode: fingerprint all frequencies up to 48 kHz
        #[arg(long)]
        full_spectrum: bool,
    },

    /// List all registered zones
    List,

    /// Remove a zone and its fingerprints
    Remove {
        /// Zone name to remove
        name: String,
    },

    /// Listen via microphone and detect which zone you're in
    Detect {
        /// Recording duration in seconds
        #[arg(short, long, default_value_t = 3)]
        duration: u64,

        /// Audible mode
        #[arg(long)]
        audible: bool,

        /// Full spectrum mode
        #[arg(long)]
        full_spectrum: bool,
    },
}

impl Cli {
    fn db_path(&self) -> PathBuf {
        self.db.clone().unwrap_or_else(daemon::default_db_path)
    }
}

/// Execute the CLI.
pub async fn execute() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

    let db_path = cli.db_path();
    let port = cli.port;
    let bind = cli.bind.clone();

    let model_path = cli.model_path.clone();

    match cli.command {
        Command::Daemon {
            foreground: _,
            monitor,
        } => {
            // Always run in foreground for now
            println!("{}", "━".repeat(50).dimmed());
            println!(
                "  {} Beacon Daemon v{}",
                "📡".bold(),
                env!("CARGO_PKG_VERSION")
            );
            println!("  {} {}", "Database:".dimmed(), db_path.display());
            println!("  {} {}:{}", "Listening:".dimmed(), bind, port);
            if monitor {
                println!("  {} enabled", "Monitor:".dimmed());
            }
            if let Some(ref mp) = model_path {
                println!("  {} {}", "Model:".dimmed(), mp.display());
            }
            println!("{}", "━".repeat(50).dimmed());
            daemon::run_daemon(&db_path, port, &bind, monitor, model_path.as_deref()).await
        }

        Command::Ingest {
            path,
            title,
            artist,
            album,
            audible,
            full_spectrum,
        } => {
            let mode = resolve_mode(audible, full_spectrum);
            let title = title.unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Unknown".to_string())
            });

            if mode != FrequencyMode::Audible {
                println!(
                    "{} Frequency mode: {}",
                    "🔊".bold(),
                    mode.description().cyan()
                );
            }

            // Try daemon first, fall back to direct mode
            if daemon::is_daemon_running(port).await {
                let request = DaemonRequest::Ingest {
                    path: path.to_string_lossy().to_string(),
                    title,
                    artist,
                    album,
                };
                let resp = daemon::send_request(port, &request).await?;
                print_response(&resp);
            } else {
                ingest_direct(&db_path, &path, &title, &artist, album.as_deref(), mode)?;
            }
            Ok(())
        }

        Command::IngestDir {
            path,
            artist,
            audible,
            full_spectrum,
        } => {
            let mode = resolve_mode(audible, full_spectrum);
            let entries: Vec<_> = std::fs::read_dir(&path)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext.eq_ignore_ascii_case("wav"))
                        .unwrap_or(false)
                })
                .collect();

            if entries.is_empty() {
                println!("{} No WAV files found in {}", "⚠".yellow(), path.display());
                return Ok(());
            }

            println!(
                "{} Found {} WAV file(s) in {}",
                "📂".bold(),
                entries.len(),
                path.display()
            );

            let pb = ProgressBar::new(entries.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("  {spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("█▓░"),
            );

            let use_daemon = daemon::is_daemon_running(port).await;

            for entry in &entries {
                let file_path = entry.path();
                let title = file_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Unknown".to_string());

                pb.set_message(title.clone());

                if use_daemon {
                    let request = DaemonRequest::Ingest {
                        path: file_path.to_string_lossy().to_string(),
                        title,
                        artist: artist.clone(),
                        album: None,
                    };
                    let _ = daemon::send_request(port, &request).await;
                } else {
                    let _ = ingest_direct(&db_path, &file_path, &title, &artist, None, mode);
                }

                pb.inc(1);
            }

            pb.finish_with_message("Done!");
            println!("\n{} Ingested {} track(s)", "✅".green(), entries.len());
            Ok(())
        }

        Command::Recognize {
            path,
            audible,
            full_spectrum,
        } => {
            let mode = resolve_mode(audible, full_spectrum);
            println!(
                "{} Analyzing: {}{}",
                "🔍".bold(),
                path.display().to_string().cyan(),
                if mode != FrequencyMode::Audible {
                    format!(" ({})", mode.description())
                } else {
                    String::new()
                }
            );

            let start = Instant::now();

            if daemon::is_daemon_running(port).await {
                let request = DaemonRequest::Recognize {
                    path: path.to_string_lossy().to_string(),
                };
                let resp = daemon::send_request(port, &request).await?;
                let elapsed = start.elapsed();

                if resp.success {
                    if let Some(data) = &resp.data {
                        if let Ok(results) =
                            serde_json::from_value::<Vec<MatchResult>>(data.clone())
                        {
                            print_match_results(&results, elapsed.as_secs_f64());
                            return Ok(());
                        }
                        if let Some(matches) = data.get("matches") {
                            if let Ok(results) =
                                serde_json::from_value::<Vec<MatchResult>>(matches.clone())
                            {
                                print_match_results(&results, elapsed.as_secs_f64());
                                return Ok(());
                            }
                        }
                    }
                    println!("{} {}", "ℹ".blue(), resp.message);
                } else {
                    println!("{} {}", "✗".red(), resp.message);
                }
            } else {
                recognize_direct(&db_path, &path, mode)?;
            }
            Ok(())
        }

        Command::Listen {
            duration,
            save,
            audible,
            full_spectrum,
        } => {
            let mode = resolve_mode(audible, full_spectrum);
            listen_and_recognize(&db_path, port, duration, save.as_deref(), mode).await
        }

        Command::Devices => {
            print_audio_devices();
            Ok(())
        }

        Command::List => {
            if daemon::is_daemon_running(port).await {
                let resp = daemon::send_request(port, &DaemonRequest::List).await?;
                if resp.success {
                    if let Some(data) = &resp.data {
                        if let Ok(tracks) =
                            serde_json::from_value::<Vec<crate::database::TrackInfo>>(data.clone())
                        {
                            print_track_list(&tracks);
                            return Ok(());
                        }
                    }
                }
                print_response(&resp);
            } else {
                let db = Database::open(db_path.clone())?;
                let tracks = db.list_tracks()?;
                print_track_list(&tracks);
            }
            Ok(())
        }

        Command::Remove { track_id } => {
            if daemon::is_daemon_running(port).await {
                let request = DaemonRequest::Remove { track_id };
                let resp = daemon::send_request(port, &request).await?;
                print_response(&resp);
            } else {
                let mut db = Database::open(db_path.clone())?;
                if db.remove_track(track_id)? {
                    println!("{} Removed track {}", "✅".green(), track_id);
                } else {
                    println!("{} Track {} not found", "✗".red(), track_id);
                }
            }
            Ok(())
        }

        Command::Stats => {
            if daemon::is_daemon_running(port).await {
                let resp = daemon::send_request(port, &DaemonRequest::Stats).await?;
                if resp.success {
                    if let Some(data) = &resp.data {
                        if let Ok(stats) =
                            serde_json::from_value::<crate::database::DbStats>(data.clone())
                        {
                            print_stats(&stats);
                            return Ok(());
                        }
                    }
                }
                print_response(&resp);
            } else {
                let db = Database::open(db_path.clone())?;
                let stats = db.stats()?;
                print_stats(&stats);
            }
            Ok(())
        }

        Command::Status => {
            if daemon::is_daemon_running(port).await {
                println!("{} Daemon is running on port {}", "✅".green(), port);
            } else {
                println!("{} Daemon is not running", "✗".red());
            }
            Ok(())
        }

        Command::Stop => {
            if daemon::is_daemon_running(port).await {
                let resp = daemon::send_request(port, &DaemonRequest::Shutdown).await?;
                print_response(&resp);
            } else {
                println!("{} Daemon is not running", "ℹ".blue());
            }
            Ok(())
        }

        Command::Demo => run_demo(&db_path).await,

        Command::Zone { command } => handle_zone_command(command, &db_path, port).await,

        Command::Monitor => monitor_events(port).await,

        Command::Spectrogram { viz_port } => {
            println!("{}", "━".repeat(50).dimmed());
            println!("  {} Beacon Realtime Spectrogram", "📡".bold(),);
            println!("{}", "━".repeat(50).dimmed());
            println!();
            crate::spectrogram_server::run(viz_port, &bind).await
        }
    }
}

// ── Zone commands ───────────────────────────────────────────────────

async fn handle_zone_command(cmd: ZoneCommand, db_path: &PathBuf, port: u16) -> Result<()> {
    match cmd {
        ZoneCommand::Add {
            name,
            path,
            record,
            duration,
            audible,
            full_spectrum,
        } => {
            let mode = resolve_mode(audible, full_spectrum);
            let start = Instant::now();

            let signal = if record {
                println!(
                    "{} Recording {}s of ambient audio ({})",
                    "🎤".bold(),
                    duration.to_string().yellow(),
                    mode.description().cyan()
                );

                let dur = std::time::Duration::from_secs(duration);
                let pb = ProgressBar::new(duration);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template(
                            "  {spinner:.magenta} [{bar:30.magenta/blue}] {pos}s / {len}s  {msg}",
                        )
                        .unwrap()
                        .progress_chars("▓▒░"),
                );
                pb.set_message("recording...");

                let record_handle = tokio::task::spawn_blocking(move || {
                    crate::microphone::record_from_mic(dur, mode)
                });

                let tick_handle = {
                    let pb = pb.clone();
                    let secs = duration;
                    tokio::spawn(async move {
                        for i in 1..=secs {
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            pb.set_position(i);
                        }
                    })
                };

                let signal = record_handle.await??;
                tick_handle.abort();
                pb.finish_and_clear();
                signal
            } else if let Some(wav_path) = &path {
                AudioSignal::from_wav_with_mode(wav_path, mode)?
            } else {
                anyhow::bail!("Provide a WAV file path or use --record to capture ambient audio");
            };

            // Compute spectrogram once, share between fingerprinting and signature.
            let fp = Fingerprinter::with_defaults();
            let spectrogram = crate::spectrogram::Spectrogram::compute(
                &signal.samples,
                signal.sample_rate,
                &fp.config.spectrogram,
            );
            let fingerprints = fp.fingerprint_spectrogram(&spectrogram);

            if fingerprints.is_empty() {
                println!(
                    "{} No fingerprints could be extracted from audio",
                    "✗".red()
                );
                return Ok(());
            }

            let sig = crate::signature::Signature::from_spectrogram(&spectrogram);
            let sig_bytes = sig.to_bytes();

            let mut db = Database::open(db_path)?;
            let file_path_str = path.as_ref().map(|p| p.to_string_lossy().to_string());
            let track_id = db.insert_track(
                &name,
                "zone",
                None,
                signal.duration_secs,
                file_path_str.as_deref(),
                &fingerprints,
            )?;
            db.insert_zone(&name, track_id, mode.as_str(), Some(&sig_bytes))?;

            let elapsed = start.elapsed();
            println!(
                "{} Zone '{}' registered ({} fingerprints, {}-dim signature, {:.1}s audio, {}, took {:.2}s)",
                "✅".green(),
                name.bold().cyan(),
                fingerprints.len(),
                sig.vector.len(),
                signal.duration_secs,
                mode.as_str().dimmed(),
                elapsed.as_secs_f64()
            );
            Ok(())
        }

        ZoneCommand::List => {
            let db = Database::open(db_path)?;
            let zones = db.list_zones()?;

            if zones.is_empty() {
                println!("{} No zones registered", "ℹ".blue());
                return Ok(());
            }

            println!();
            println!("  {} {} zone(s) registered", "📡".bold(), zones.len());
            println!("{}", "  ─".repeat(30).dimmed());
            println!(
                "  {:<20}  {:>8}  {:>10}  {:>12}",
                "ZONE".dimmed(),
                "DURATION".dimmed(),
                "FPs".dimmed(),
                "MODE".dimmed()
            );
            println!("{}", "  ─".repeat(30).dimmed());

            for zone in &zones {
                println!(
                    "  {:<20}  {:>7.1}s  {:>10}  {:>12}",
                    zone.name.bold().cyan(),
                    zone.duration_secs,
                    zone.fingerprint_count.to_string().green(),
                    zone.frequency_mode.dimmed()
                );
            }
            println!();
            Ok(())
        }

        ZoneCommand::Remove { name } => {
            let mut db = Database::open(db_path)?;
            if db.remove_zone(&name)? {
                println!("{} Zone '{}' removed", "✅".green(), name.bold());
            } else {
                println!("{} Zone '{}' not found", "✗".red(), name);
            }
            Ok(())
        }

        ZoneCommand::Detect {
            duration,
            audible,
            full_spectrum,
        } => {
            let mode = resolve_mode(audible, full_spectrum);
            zone_detect(db_path, port, duration, mode).await
        }
    }
}

async fn zone_detect(
    db_path: &PathBuf,
    _port: u16,
    duration_secs: u64,
    mode: FrequencyMode,
) -> Result<()> {
    use crate::microphone;
    use std::time::Duration;

    match microphone::default_input_device_info() {
        Ok(info) => {
            println!(
                "{} Using: {} ({}Hz, {}ch)",
                "🎤".bold(),
                info.name.cyan(),
                info.sample_rate,
                info.channels
            );
        }
        Err(e) => {
            println!("{} {}", "✗".red(), e);
            return Err(e);
        }
    }

    println!(
        "{} Listening for {}s ({})...",
        "👂".bold(),
        duration_secs.to_string().yellow(),
        mode.as_str()
    );

    let duration = Duration::from_secs(duration_secs);

    let pb = ProgressBar::new(duration_secs);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {spinner:.magenta} [{bar:30.magenta/blue}] {pos}s / {len}s  {msg}")
            .unwrap()
            .progress_chars("▓▒░"),
    );
    pb.set_message("📡 scanning...");

    let record_handle =
        tokio::task::spawn_blocking(move || microphone::record_from_mic(duration, mode));

    let tick_handle = {
        let pb = pb.clone();
        let dur = duration_secs;
        tokio::spawn(async move {
            for i in 1..=dur {
                tokio::time::sleep(Duration::from_secs(1)).await;
                pb.set_position(i);
            }
        })
    };

    let signal = record_handle.await??;
    tick_handle.abort();
    pb.finish_and_clear();

    println!(
        "  {} Captured {:.1}s of audio",
        "✅".green(),
        signal.duration_secs
    );

    let db = Database::open(db_path)?;

    // Compute spectrogram once, share between signature and fingerprinting.
    let fp = Fingerprinter::with_defaults();
    let spectrogram = crate::spectrogram::Spectrogram::compute(
        &signal.samples,
        signal.sample_rate,
        &fp.config.spectrogram,
    );

    // --- Vector similarity matching (ambient signatures) ---
    let query_sig = crate::signature::Signature::from_spectrogram(&spectrogram);

    let zone_sigs = db.get_zone_signatures()?;
    let mut best_sig_match: Option<(String, f32)> = None;
    for (zone_name, sig_bytes) in &zone_sigs {
        if let Some(stored_sig) = crate::signature::Signature::from_bytes(sig_bytes) {
            let sim = query_sig.similarity(&stored_sig);
            if best_sig_match.as_ref().is_none_or(|(_, s)| sim > *s) {
                best_sig_match = Some((zone_name.clone(), sim));
            }
        }
    }

    // --- Hash-based matching (Shazam-style, reuses same spectrogram) ---
    let fingerprints = fp.fingerprint_spectrogram(&spectrogram);

    let mut hash_zone: Option<(String, f64)> = None;
    if !fingerprints.is_empty() {
        println!(
            "  {} Extracted {} fingerprints",
            "🔑".dimmed(),
            fingerprints.len().to_string().cyan()
        );

        let results = db.search(&fingerprints, 3, signal.sample_rate)?;
        if let Some(top) = results.first() {
            if let Ok(Some(zone)) = db.get_zone_by_track_id(top.track.id) {
                hash_zone = Some((zone.name, top.confidence));
            }
        }
    }

    // --- Pick the best result ---
    // Normalize: signature similarity is [-1,1], hash confidence is [0,1].
    // Use signature match if it's strong enough (>0.6), otherwise fall back to hash match.
    let sig_threshold = 0.6;

    let detected = match (&best_sig_match, &hash_zone) {
        (Some((sig_name, sim)), Some((hash_name, conf))) => {
            if sig_name == hash_name {
                // Both agree — report with combined confidence.
                Some((
                    sig_name.clone(),
                    format!("{:.0}% hash + {:.0}% signature", conf * 100.0, sim * 100.0),
                ))
            } else if *sim > sig_threshold && *sim as f64 > *conf {
                Some((sig_name.clone(), format!("{:.0}% signature", sim * 100.0)))
            } else {
                Some((hash_name.clone(), format!("{:.0}% hash", conf * 100.0)))
            }
        }
        (Some((name, sim)), None) if *sim > sig_threshold => {
            Some((name.clone(), format!("{:.0}% signature", sim * 100.0)))
        }
        (None, Some((name, conf))) => Some((name.clone(), format!("{:.0}% hash", conf * 100.0))),
        _ => None,
    };

    match detected {
        Some((zone_name, confidence_str)) => {
            println!();
            println!(
                "  {} Zone detected: {} ({})",
                "📍".bold(),
                zone_name.bold().cyan(),
                confidence_str.green()
            );
            println!();
        }
        None => {
            println!();
            println!("  {} No zone detected", "📡".dimmed());
            println!();
        }
    }

    Ok(())
}

// ── Monitor ─────────────────────────────────────────────────────────

async fn monitor_events(port: u16) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::net::TcpStream;

    let addr = format!("127.0.0.1:{}", port);
    let stream = TcpStream::connect(&addr).await.with_context(|| {
        format!(
            "Cannot connect to daemon at {}. Start with: beacond daemon --monitor",
            addr
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    // Send subscribe request
    let request = r#"{"type":"monitor_subscribe"}"#.to_string() + "\n";
    tokio::io::AsyncWriteExt::write_all(&mut writer, request.as_bytes()).await?;
    tokio::io::AsyncWriteExt::flush(&mut writer).await?;

    println!(
        "{} Connected to daemon — streaming zone events (Ctrl+C to stop)",
        "📡".bold()
    );
    println!();

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            println!("{} Daemon disconnected", "ℹ".blue());
            break;
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            println!("{}", trimmed);
        }
    }

    Ok(())
}

// ── Microphone capture ───────────────────────────────────────────────

fn print_audio_devices() {
    use crate::microphone;

    println!();
    println!("  {} Audio Input Devices", "🎤".bold());
    println!("{}", "  ─".repeat(25).dimmed());

    match microphone::list_input_devices() {
        Ok(devices) if devices.is_empty() => {
            println!("  {} No audio input devices found", "⚠".yellow());
        }
        Ok(devices) => {
            for (i, dev) in devices.iter().enumerate() {
                let marker = match microphone::default_input_device_info() {
                    Ok(default) if default.name == dev.name => " (default)".green().to_string(),
                    _ => String::new(),
                };
                println!(
                    "  {} {}{}",
                    format!("[{}]", i).dimmed(),
                    dev.name.bold(),
                    marker
                );
                println!(
                    "      {}Hz, {}ch, {}",
                    dev.sample_rate.to_string().cyan(),
                    dev.channels,
                    dev.sample_format.dimmed()
                );
            }
        }
        Err(e) => {
            println!("  {} Failed to list devices: {}", "✗".red(), e);
        }
    }
    println!();
}

async fn listen_and_recognize(
    db_path: &PathBuf,
    _port: u16,
    duration_secs: u64,
    save_path: Option<&std::path::Path>,
    mode: FrequencyMode,
) -> Result<()> {
    use crate::microphone;
    use std::time::Duration;

    // Show device info
    match microphone::default_input_device_info() {
        Ok(info) => {
            println!(
                "{} Using: {} ({}Hz, {}ch)",
                "🎤".bold(),
                info.name.cyan(),
                info.sample_rate,
                info.channels
            );
        }
        Err(e) => {
            println!("{} {}", "✗".red(), e);
            return Err(e);
        }
    }

    if mode != FrequencyMode::Audible {
        println!(
            "{} Frequency mode: {}",
            "🔊".bold(),
            mode.description().cyan()
        );
    }

    let duration = Duration::from_secs(duration_secs);

    println!(
        "{} Listening for {}s...",
        "👂".bold(),
        duration_secs.to_string().yellow()
    );
    println!();

    // Animated listening indicator
    let pb = ProgressBar::new(duration_secs);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {spinner:.magenta} [{bar:30.magenta/blue}] {pos}s / {len}s  {msg}")
            .unwrap()
            .progress_chars("▓▒░"),
    );
    pb.set_message("🎵 listening...");

    // Record in a blocking thread (CPAL uses callbacks on its own thread)
    let record_handle = {
        let save = save_path.map(|p| p.to_path_buf());
        tokio::task::spawn_blocking(move || {
            if let Some(ref wav_path) = save {
                microphone::record_to_wav(duration, wav_path, mode)
            } else {
                microphone::record_from_mic(duration, mode)
            }
        })
    };

    // Tick the progress bar while recording
    let tick_handle = {
        let pb = pb.clone();
        let dur = duration_secs;
        tokio::spawn(async move {
            for i in 1..=dur {
                tokio::time::sleep(Duration::from_secs(1)).await;
                pb.set_position(i);
            }
        })
    };

    // Wait for recording to complete
    let signal = record_handle.await??;
    tick_handle.abort();
    pb.finish_and_clear();

    if let Some(p) = save_path {
        println!(
            "  {} Saved recording to {}",
            "💾".bold(),
            p.display().to_string().cyan()
        );
    }

    println!(
        "  {} Captured {:.1}s of audio ({} samples)",
        "✅".green(),
        signal.duration_secs,
        signal.samples.len()
    );
    println!();

    // Now fingerprint and search
    println!("{} Analyzing audio...", "🔍".bold());

    let start = Instant::now();
    let fp = Fingerprinter::with_defaults();
    let fingerprints = fp.fingerprint(&signal.samples, signal.sample_rate);

    if fingerprints.is_empty() {
        println!(
            "{} Could not extract fingerprints from the recording",
            "⚠".yellow()
        );
        return Ok(());
    }

    println!(
        "  {} Extracted {} fingerprints",
        "🔑".dimmed(),
        fingerprints.len().to_string().cyan()
    );

    // Search
    let db = Database::open(db_path)?;
    let results = db.search(&fingerprints, 5, signal.sample_rate)?;
    let elapsed = start.elapsed();

    print_match_results(&results, elapsed.as_secs_f64());

    Ok(())
}

// ── Direct mode (no daemon) ──────────────────────────────────────────

fn ingest_direct(
    db_path: &PathBuf,
    wav_path: &PathBuf,
    title: &str,
    artist: &str,
    album: Option<&str>,
    mode: FrequencyMode,
) -> Result<()> {
    let start = Instant::now();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("Reading {}", wav_path.display()));

    let signal = AudioSignal::from_wav_with_mode(wav_path, mode)?;
    pb.set_message("Computing fingerprints...");

    let fp = Fingerprinter::with_defaults();
    let fingerprints = fp.fingerprint(&signal.samples, signal.sample_rate);

    if fingerprints.is_empty() {
        pb.finish_and_clear();
        println!("{} No fingerprints could be extracted", "⚠".yellow());
        return Ok(());
    }

    pb.set_message("Storing in database...");
    let mut db = Database::open(db_path)?;
    let track_id = db.insert_track(
        title,
        artist,
        album,
        signal.duration_secs,
        Some(&wav_path.to_string_lossy()),
        &fingerprints,
    )?;

    pb.finish_and_clear();

    let elapsed = start.elapsed();
    println!(
        "{} Ingested: {} — {} (id: {}, {} fps, {:.1}s audio, took {:.2}s)",
        "✅".green(),
        title.bold(),
        artist.dimmed(),
        track_id,
        fingerprints.len(),
        signal.duration_secs,
        elapsed.as_secs_f64()
    );

    Ok(())
}

fn recognize_direct(db_path: &PathBuf, wav_path: &PathBuf, mode: FrequencyMode) -> Result<()> {
    let start = Instant::now();

    let signal = AudioSignal::from_wav_with_mode(wav_path, mode)?;
    let fp = Fingerprinter::with_defaults();
    let fingerprints = fp.fingerprint(&signal.samples, signal.sample_rate);

    if fingerprints.is_empty() {
        println!(
            "{} No fingerprints could be extracted from audio",
            "✗".red()
        );
        return Ok(());
    }

    let db = Database::open(db_path)?;
    let results = db.search(&fingerprints, 5, signal.sample_rate)?;
    let elapsed = start.elapsed();

    print_match_results(&results, elapsed.as_secs_f64());

    Ok(())
}

// ── Pretty printing ──────────────────────────────────────────────────

fn print_response(resp: &DaemonResponse) {
    let icon = if resp.success {
        "✅".green()
    } else {
        "✗".red()
    };
    println!("{} {}", icon, resp.message);
    if let Some(data) = &resp.data {
        if let Ok(pretty) = serde_json::to_string_pretty(data) {
            println!("{}", pretty.dimmed());
        }
    }
}

fn print_match_results(results: &[MatchResult], elapsed_secs: f64) {
    println!();
    if results.is_empty() {
        println!(
            "  {} No matches found (searched in {:.2}s)",
            "🔇".dimmed(),
            elapsed_secs
        );
        println!();
        return;
    }

    println!(
        "  🎯 {} match(es) found in {:.2}s",
        results.len(),
        elapsed_secs
    );
    println!("{}", "  ─".repeat(20).dimmed());

    for (i, result) in results.iter().enumerate() {
        let confidence_pct = (result.confidence * 100.0) as u32;
        let confidence_bar = confidence_bar(result.confidence);
        let rank = format!("#{}", i + 1);

        println!();
        println!(
            "  {} {} — {}",
            rank.bold().cyan(),
            result.track.title.bold().white(),
            result.track.artist.yellow()
        );

        if let Some(ref album) = result.track.album {
            println!("     {} {}", "Album:".dimmed(), album);
        }

        println!(
            "     {} {} ({}%)",
            "Confidence:".dimmed(),
            confidence_bar,
            confidence_pct
        );
        println!(
            "     {} {} matching hashes",
            "Matches:".dimmed(),
            result.match_count.to_string().green()
        );
        println!(
            "     {} ~{:.1}s into track",
            "Position:".dimmed(),
            result.estimated_position_secs
        );
    }
    println!();
}

fn confidence_bar(confidence: f64) -> String {
    let filled = (confidence * 20.0) as usize;
    let empty = 20 - filled;
    let bar_color = if confidence > 0.5 {
        "green"
    } else if confidence > 0.2 {
        "yellow"
    } else {
        "red"
    };

    let filled_str: String = "█".repeat(filled);
    let empty_str: String = "░".repeat(empty);

    match bar_color {
        "green" => format!("{}{}", filled_str.green(), empty_str.dimmed()),
        "yellow" => format!("{}{}", filled_str.yellow(), empty_str.dimmed()),
        _ => format!("{}{}", filled_str.red(), empty_str.dimmed()),
    }
}

fn print_track_list(tracks: &[crate::database::TrackInfo]) {
    if tracks.is_empty() {
        println!("{} No tracks in database", "ℹ".blue());
        return;
    }

    println!();
    println!("  {} {} track(s) in database", "🎶".bold(), tracks.len());
    println!("{}", "  ─".repeat(25).dimmed());
    println!(
        "  {:>4}  {:<30}  {:<20}  {:>8}  {:>8}",
        "ID".dimmed(),
        "TITLE".dimmed(),
        "ARTIST".dimmed(),
        "DURATION".dimmed(),
        "FPs".dimmed()
    );
    println!("{}", "  ─".repeat(25).dimmed());

    for track in tracks {
        let duration = format!("{:.1}s", track.duration_secs);
        let title = if track.title.len() > 28 {
            format!("{}…", &track.title[..27])
        } else {
            track.title.clone()
        };
        let artist = if track.artist.len() > 18 {
            format!("{}…", &track.artist[..17])
        } else {
            track.artist.clone()
        };

        println!(
            "  {:>4}  {:<30}  {:<20}  {:>8}  {:>8}",
            track.id.to_string().cyan(),
            title.bold(),
            artist.yellow(),
            duration.dimmed(),
            track.fingerprint_count.to_string().green()
        );
    }
    println!();
}

fn print_stats(stats: &crate::database::DbStats) {
    println!();
    println!("  {} Database Statistics", "📊".bold());
    println!("{}", "  ─".repeat(20).dimmed());
    println!(
        "  {} {}",
        "Tracks:".dimmed(),
        stats.track_count.to_string().bold()
    );
    println!(
        "  {} {}",
        "Fingerprints:".dimmed(),
        stats.fingerprint_count.to_string().bold()
    );
    println!(
        "  {} {}",
        "Database size:".dimmed(),
        stats.db_size_human().bold()
    );
    if stats.track_count > 0 {
        let avg = stats.fingerprint_count / stats.track_count;
        println!(
            "  {} {} per track",
            "Avg fingerprints:".dimmed(),
            avg.to_string().bold()
        );
    }
    println!();
}

// ── Demo ─────────────────────────────────────────────────────────────

async fn run_demo(db_path: &PathBuf) -> Result<()> {
    use crate::audio::{generate_composite, write_wav};

    println!();
    println!("{}", "━".repeat(50).cyan());
    println!("  {} Beacon — Self-Test Demo", "📡".bold());
    println!("{}", "━".repeat(50).cyan());
    println!();

    let tmp_dir = std::env::temp_dir().join("beacond_demo");
    std::fs::create_dir_all(&tmp_dir)?;

    // Generate synthetic "songs"
    type Song<'a> = (&'a str, &'a str, Vec<(f32, f32)>);
    let songs: Vec<Song> = vec![
        (
            "Midnight Synth",
            "Neon Dreams",
            vec![
                (261.63, 0.4),  // C4
                (329.63, 0.3),  // E4
                (392.0, 0.25),  // G4
                (523.25, 0.15), // C5
                (150.0, 0.2),   // Bass
            ],
        ),
        (
            "Ocean Waves",
            "Blue Horizon",
            vec![
                (196.0, 0.35),  // G3
                (246.94, 0.3),  // B3
                (293.66, 0.25), // D4
                (440.0, 0.2),   // A4
                (110.0, 0.15),  // Bass
            ],
        ),
        (
            "Electric Storm",
            "Thunderbolt",
            vec![
                (349.23, 0.4), // F4
                (415.3, 0.3),  // Ab4
                (523.25, 0.2), // C5
                (698.46, 0.2), // F5
                (174.61, 0.3), // F3 bass
            ],
        ),
        (
            "Starlight Waltz",
            "Celestial",
            vec![
                (277.18, 0.35), // C#4
                (369.99, 0.3),  // F#4
                (554.37, 0.2),  // C#5
                (415.30, 0.25), // G#4
                (138.59, 0.2),  // C#3 bass
            ],
        ),
    ];

    let mut db = Database::open(db_path)?;

    println!("  {} Generating synthetic songs...", "🎹".bold());
    println!();

    let fp = Fingerprinter::with_defaults();

    for (title, artist, freqs) in &songs {
        let samples = generate_composite(freqs, 8.0, 16000);
        let wav_path = tmp_dir.join(format!("{}.wav", title.replace(' ', "_")));
        write_wav(&wav_path, &samples, 16000)?;

        let start = Instant::now();
        let fingerprints = fp.fingerprint(&samples, 16000);

        db.insert_track(
            title,
            artist,
            None,
            8.0,
            Some(&wav_path.to_string_lossy()),
            &fingerprints,
        )?;

        println!(
            "  {} Ingested: {} — {} ({} fps in {:.2}s)",
            "✅".green(),
            title.bold(),
            artist.dimmed(),
            fingerprints.len().to_string().cyan(),
            start.elapsed().as_secs_f64()
        );
    }

    println!();
    println!("  {} Testing recognition...", "🔍".bold());
    println!();

    // Test 1: Recognize a full-length original
    {
        let (title, artist, freqs) = &songs[0];
        let samples = generate_composite(freqs, 8.0, 16000);
        let start = Instant::now();
        let query_fps = fp.fingerprint(&samples, 16000);
        let results = db.search(&query_fps, 3, 16000)?;
        let elapsed = start.elapsed();

        println!(
            "  {} Full signal test ({} — {})",
            "Test 1:".bold().cyan(),
            title,
            artist
        );
        if let Some(top) = results.first() {
            let pass = top.track.title == *title;
            let icon = if pass { "✅" } else { "❌" };
            println!(
                "  {} Found: {} (confidence: {:.0}%, {} hits, {:.2}s)",
                icon,
                top.track.title.bold(),
                top.confidence * 100.0,
                top.match_count,
                elapsed.as_secs_f64()
            );
        } else {
            println!("  ❌ No match found");
        }
    }

    // Test 2: Recognize a short snippet (middle 3 seconds)
    {
        let (title, artist, freqs) = &songs[2];
        let full = generate_composite(freqs, 8.0, 16000);
        // Take 3 seconds from the middle
        let start_sample = 2 * 16000; // 2s in
        let end_sample = 5 * 16000; // 5s in
        let snippet = &full[start_sample..end_sample.min(full.len())];

        let start = Instant::now();
        let query_fps = fp.fingerprint(snippet, 16000);
        let results = db.search(&query_fps, 3, 16000)?;
        let elapsed = start.elapsed();

        println!(
            "  {} 3-second snippet test ({} — {})",
            "Test 2:".bold().cyan(),
            title,
            artist
        );
        if let Some(top) = results.first() {
            let pass = top.track.title == *title;
            let icon = if pass { "✅" } else { "❌" };
            println!(
                "  {} Found: {} (confidence: {:.0}%, {} hits, {:.2}s)",
                icon,
                top.track.title.bold(),
                top.confidence * 100.0,
                top.match_count,
                elapsed.as_secs_f64()
            );
        } else {
            println!("  ❌ No match found");
        }
    }

    // Test 3: Recognize with added noise
    {
        let (title, artist, freqs) = &songs[1];
        let mut samples = generate_composite(freqs, 8.0, 16000);
        // Add noise
        let mut rng_state: u32 = 42;
        for s in samples.iter_mut() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng_state >> 16) as f32 / 32768.0 - 1.0) * 0.1;
            *s += noise;
        }

        let start = Instant::now();
        let query_fps = fp.fingerprint(&samples, 16000);
        let results = db.search(&query_fps, 3, 16000)?;
        let elapsed = start.elapsed();

        println!(
            "  {} Noisy signal test ({} — {})",
            "Test 3:".bold().cyan(),
            title,
            artist
        );
        if let Some(top) = results.first() {
            let pass = top.track.title == *title;
            let icon = if pass { "✅" } else { "❌" };
            println!(
                "  {} Found: {} (confidence: {:.0}%, {} hits, {:.2}s)",
                icon,
                top.track.title.bold(),
                top.confidence * 100.0,
                top.match_count,
                elapsed.as_secs_f64()
            );
        } else {
            println!("  ❌ No match found");
        }
    }

    // Test 4: Non-matching signal
    {
        let unknown = generate_composite(
            &[
                (500.0, 0.3),
                (750.0, 0.3),
                (1000.0, 0.2),
                (1250.0, 0.1),
                (125.0, 0.3),
            ],
            5.0,
            16000,
        );

        let start = Instant::now();
        let query_fps = fp.fingerprint(&unknown, 16000);
        let results = db.search(&query_fps, 3, 16000)?;
        let elapsed = start.elapsed();

        println!(
            "  {} Unknown signal (should not match)",
            "Test 4:".bold().cyan()
        );
        if results.is_empty() {
            println!(
                "  ✅ Correctly returned no matches ({:.2}s)",
                elapsed.as_secs_f64()
            );
        } else {
            let top = &results[0];
            if top.confidence < 0.15 {
                println!(
                    "  ⚠️ Low-confidence match: {} ({:.0}% - acceptable)",
                    top.track.title,
                    top.confidence * 100.0
                );
            } else {
                println!(
                    "  ❌ False positive: {} ({:.0}%)",
                    top.track.title,
                    top.confidence * 100.0
                );
            }
        }
    }

    // Print stats
    let stats = db.stats()?;
    println!();
    print_stats(&stats);

    // Cleanup temp files
    std::fs::remove_dir_all(&tmp_dir).ok();

    println!("  {} Demo complete!", "🎉".bold());
    println!();

    Ok(())
}
