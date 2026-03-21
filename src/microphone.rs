//! Microphone capture module using CPAL for cross-platform audio input.
//!
//! Provides live audio recording from the system's default input device,
//! with real-time sample buffering and conversion to our internal format.

use crate::audio::AudioSignal;
use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Information about the selected audio input device.
#[derive(Debug, Clone)]
pub struct InputDeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: String,
}

/// List available audio input devices.
pub fn list_input_devices() -> Result<Vec<InputDeviceInfo>> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    for device in host
        .input_devices()
        .context("Failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());

        if let Ok(config) = device.default_input_config() {
            devices.push(InputDeviceInfo {
                name,
                sample_rate: config.sample_rate().0,
                channels: config.channels(),
                sample_format: format!("{:?}", config.sample_format()),
            });
        }
    }

    Ok(devices)
}

/// Get info about the default input device.
pub fn default_input_device_info() -> Result<InputDeviceInfo> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No default audio input device found")?;

    let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
    let config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    Ok(InputDeviceInfo {
        name,
        sample_rate: config.sample_rate().0,
        channels: config.channels(),
        sample_format: format!("{:?}", config.sample_format()),
    })
}

/// Record audio from the default microphone for a specified duration.
///
/// Returns an `AudioSignal` already resampled and filtered for the given mode.
pub fn record_from_mic(
    duration: Duration,
    mode: crate::audio::FrequencyMode,
) -> Result<AudioSignal> {
    use crate::audio::FrequencyMode;

    let host = cpal::default_host();

    let device = host.default_input_device().context(
        "No default audio input device available.\n\
                  Make sure a microphone is connected and accessible.",
    )?;

    let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
    log::info!("Using input device: {}", device_name);

    let default_config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    // For ultrasonic/full modes, try to get a higher sample rate from the device
    let (stream_config, sample_rate, channels, sample_format) = if mode != FrequencyMode::Audible {
        // Try to find a config with >= 44100 Hz, preferably 96000 Hz
        match find_high_rate_config(&device) {
            Some((cfg, fmt)) => {
                let sr = cfg.sample_rate.0;
                let ch = cfg.channels as usize;
                log::info!("Using high-rate config for {:?}: {}Hz, {}ch", mode, sr, ch);
                if sr < 44100 {
                    log::warn!(
                        "Device max sample rate is {}Hz — ultrasonic content above {}Hz \
                             will not be captured.",
                        sr,
                        sr / 2
                    );
                }
                (cfg, sr, ch, fmt)
            }
            None => {
                let sr = default_config.sample_rate().0;
                let ch = default_config.channels() as usize;
                let fmt = default_config.sample_format();
                log::warn!(
                    "Could not find high sample rate config; falling back to {}Hz. \
                         Ultrasonic capture may be limited.",
                    sr
                );
                (default_config.into(), sr, ch, fmt)
            }
        }
    } else {
        let sr = default_config.sample_rate().0;
        let ch = default_config.channels() as usize;
        let fmt = default_config.sample_format();
        (default_config.into(), sr, ch, fmt)
    };

    log::info!(
        "Recording config: {}Hz, {}ch, {:?} (mode: {:?})",
        sample_rate,
        channels,
        sample_format,
        mode
    );

    // Shared buffer for collecting samples from the audio callback
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = Arc::clone(&buffer);
    let err_flag: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let err_flag_clone = Arc::clone(&err_flag);

    // Build the input stream based on the sample format
    let stream = match sample_format {
        cpal::SampleFormat::I8 => build_input_stream::<i8>(
            &device,
            &stream_config,
            channels,
            buffer_clone,
            err_flag_clone,
        )?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(
            &device,
            &stream_config,
            channels,
            buffer_clone,
            err_flag_clone,
        )?,
        cpal::SampleFormat::F32 => build_input_stream::<f32>(
            &device,
            &stream_config,
            channels,
            buffer_clone,
            err_flag_clone,
        )?,
        // For I32 and other formats, use a manual conversion stream
        _ => build_input_stream_i32(
            &device,
            &stream_config,
            channels,
            buffer_clone,
            err_flag_clone,
        )?,
    };

    // Start recording
    stream.play().context("Failed to start audio stream")?;

    let start = Instant::now();

    // Wait for the recording duration, checking periodically
    while start.elapsed() < duration {
        std::thread::sleep(Duration::from_millis(50));

        // Check for stream errors
        if let Some(err) = err_flag.lock().unwrap().as_ref() {
            bail!("Audio stream error: {}", err);
        }
    }

    // Stop recording
    drop(stream);

    // Check for any final errors
    if let Some(err) = err_flag.lock().unwrap().as_ref() {
        bail!("Audio stream error: {}", err);
    }

    // Get the recorded samples
    let raw_samples = std::mem::take(&mut *buffer.lock().unwrap());

    if raw_samples.is_empty() {
        bail!(
            "No audio samples were captured. \
             Check that your microphone is working and permissions are granted."
        );
    }

    let actual_duration = raw_samples.len() as f32 / (sample_rate as f32 * channels as f32);
    log::info!(
        "Captured {} samples ({:.2}s of audio)",
        raw_samples.len(),
        actual_duration
    );

    // Mix to mono if multi-channel
    let mono_samples = if channels > 1 {
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    // Create AudioSignal (handles resampling and filtering for the mode)
    Ok(AudioSignal::from_samples_with_mode(
        mono_samples,
        sample_rate,
        mode,
    ))
}

/// Build a CPAL input stream for a given sample type.
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    buffer: Arc<Mutex<Vec<f32>>>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + Into<f32> + Send + 'static,
{
    let _ = channels; // channels info is already in the config

    let stream = device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let float_data: Vec<f32> = data.iter().map(|&s| s.into()).collect();
                if let Ok(mut buf) = buffer.lock() {
                    buf.extend_from_slice(&float_data);
                }
            },
            move |err| {
                log::error!("Audio stream error: {}", err);
                if let Ok(mut flag) = err_flag.lock() {
                    *flag = Some(err.to_string());
                }
            },
            None, // no timeout
        )
        .context("Failed to build input audio stream")?;

    Ok(stream)
}

/// Build a CPAL input stream specifically for i32 samples (no lossless Into<f32>).
fn build_input_stream_i32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    _channels: usize,
    buffer: Arc<Mutex<Vec<f32>>>,
    err_flag: Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream> {
    let stream = device
        .build_input_stream(
            config,
            move |data: &[i32], _: &cpal::InputCallbackInfo| {
                let float_data: Vec<f32> =
                    data.iter().map(|&s| s as f32 / i32::MAX as f32).collect();
                if let Ok(mut buf) = buffer.lock() {
                    buf.extend_from_slice(&float_data);
                }
            },
            move |err| {
                log::error!("Audio stream error: {}", err);
                if let Ok(mut flag) = err_flag.lock() {
                    *flag = Some(err.to_string());
                }
            },
            None,
        )
        .context("Failed to build input audio stream (i32)")?;

    Ok(stream)
}

/// Try to find an input config with a high sample rate (for ultrasonic capture).
pub fn find_high_rate_config(
    device: &cpal::Device,
) -> Option<(cpal::StreamConfig, cpal::SampleFormat)> {
    use cpal::traits::DeviceTrait;

    // Preferred rates in descending order
    let preferred_rates = [96000u32, 48000, 44100];

    let supported = device.supported_input_configs().ok()?;
    let configs: Vec<_> = supported.collect();

    for &target_rate in &preferred_rates {
        for cfg in &configs {
            let min = cfg.min_sample_rate().0;
            let max = cfg.max_sample_rate().0;
            if target_rate >= min && target_rate <= max {
                let concrete = cfg.with_sample_rate(cpal::SampleRate(target_rate));
                return Some((concrete.into(), cfg.sample_format()));
            }
        }
    }

    // Fall back to the highest rate available
    let mut best: Option<(cpal::StreamConfig, cpal::SampleFormat, u32)> = None;
    for cfg in &configs {
        let max_rate = cfg.max_sample_rate().0;
        if best.as_ref().is_none_or(|b| max_rate > b.2) {
            let concrete = cfg.with_sample_rate(cpal::SampleRate(max_rate));
            best = Some((concrete.into(), cfg.sample_format(), max_rate));
        }
    }

    best.map(|(cfg, fmt, _)| (cfg, fmt))
}

/// Record from the microphone and save to a WAV file.
pub fn record_to_wav(
    duration: Duration,
    output_path: &std::path::Path,
    mode: crate::audio::FrequencyMode,
) -> Result<AudioSignal> {
    let signal = record_from_mic(duration, mode)?;

    // Write the captured audio to WAV
    crate::audio::write_wav(output_path, &signal.samples, signal.sample_rate)?;
    log::info!("Saved recording to {}", output_path.display());

    Ok(signal)
}
