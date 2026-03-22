//! Audio processing module: WAV reading, resampling, mono conversion,
//! and frequency-band filtering for ultrasonic operation.

use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default target sample rate for normal (audible) fingerprinting.
pub const TARGET_SAMPLE_RATE: u32 = 16000;

/// Target sample rate for ultrasonic mode (captures up to ~48 kHz).
pub const ULTRASONIC_SAMPLE_RATE: u32 = 96000;

/// Frequency modes for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrequencyMode {
    /// Standard audible range: 20 Hz – 8 kHz (16 kHz sample rate).
    Audible,
    /// Ultrasonic range: 18 kHz – 48 kHz (96 kHz sample rate).
    /// Filters out audible frequencies so only inaudible content is fingerprinted.
    Ultrasonic,
    /// Full spectrum: preserves everything the source provides (up to 48 kHz at 96 kHz rate).
    /// Useful when you want to fingerprint both audible and ultrasonic content.
    Full,
}

impl FrequencyMode {
    /// The sample rate to use for this mode.
    pub fn sample_rate(&self) -> u32 {
        match self {
            FrequencyMode::Audible => TARGET_SAMPLE_RATE,
            FrequencyMode::Ultrasonic => ULTRASONIC_SAMPLE_RATE,
            FrequencyMode::Full => ULTRASONIC_SAMPLE_RATE,
        }
    }

    /// High-pass cutoff in Hz (signals below this are filtered out).
    /// Returns `None` if no high-pass is needed.
    pub fn highpass_cutoff(&self) -> Option<f32> {
        match self {
            FrequencyMode::Ultrasonic => Some(18000.0),
            _ => None,
        }
    }

    /// Low-pass cutoff in Hz (signals above this are filtered out).
    /// Returns `None` if no low-pass is needed.
    pub fn lowpass_cutoff(&self) -> Option<f32> {
        match self {
            FrequencyMode::Audible => Some(8000.0),
            _ => None,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            FrequencyMode::Audible => "audible (20 Hz – 8 kHz)",
            FrequencyMode::Ultrasonic => "ultrasonic (18 kHz – 48 kHz)",
            FrequencyMode::Full => "full spectrum (20 Hz – 48 kHz)",
        }
    }

    /// Short string identifier for database storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            FrequencyMode::Audible => "audible",
            FrequencyMode::Ultrasonic => "ultrasonic",
            FrequencyMode::Full => "full",
        }
    }

    /// Parse from a stored string. Returns `None` for unknown values.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "audible" => Some(FrequencyMode::Audible),
            "ultrasonic" => Some(FrequencyMode::Ultrasonic),
            "full" => Some(FrequencyMode::Full),
            _ => None,
        }
    }
}

/// Represents a decoded audio signal, always mono at the configured sample rate.
#[derive(Debug, Clone)]
pub struct AudioSignal {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration_secs: f32,
    pub mode: FrequencyMode,
}

impl AudioSignal {
    /// Load a WAV file from disk and normalize to mono at the target rate.
    pub fn from_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_wav_with_mode(path, FrequencyMode::Audible)
    }

    /// Load a WAV file with a specific frequency mode.
    pub fn from_wav_with_mode<P: AsRef<Path>>(path: P, mode: FrequencyMode) -> Result<Self> {
        let path = path.as_ref();
        let reader = WavReader::open(path)
            .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_rate = spec.sample_rate;

        log::debug!(
            "Reading WAV: {}ch, {}Hz, {:?}, {}bit (mode: {:?})",
            channels,
            source_rate,
            spec.sample_format,
            spec.bits_per_sample,
            mode
        );

        // For ultrasonic mode, warn if source rate is too low
        if mode == FrequencyMode::Ultrasonic && source_rate < 44100 {
            log::warn!(
                "Source sample rate {}Hz is too low for ultrasonic mode. \
                 Need at least 44100Hz to capture frequencies above 18kHz.",
                source_rate
            );
        }

        // Read all samples as f32
        let raw_samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1u64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
        };

        // Mix down to mono
        let mono_samples = mix_to_mono(&raw_samples, channels);

        // Resample to target rate for this mode
        let target_rate = mode.sample_rate();
        let mut samples = resample(&mono_samples, source_rate, target_rate);

        // Apply frequency-band filtering in-place
        apply_mode_filter(&mut samples, target_rate, mode);

        let duration_secs = samples.len() as f32 / target_rate as f32;

        Ok(AudioSignal {
            samples,
            sample_rate: target_rate,
            duration_secs,
            mode,
        })
    }

    /// Create an AudioSignal from raw f32 samples (assumed mono at given rate).
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::from_samples_with_mode(samples, sample_rate, FrequencyMode::Audible)
    }

    /// Create an AudioSignal from raw samples with a specific frequency mode.
    pub fn from_samples_with_mode(
        samples: Vec<f32>,
        sample_rate: u32,
        mode: FrequencyMode,
    ) -> Self {
        let target_rate = mode.sample_rate();
        let mut resampled = if sample_rate != target_rate {
            resample(&samples, sample_rate, target_rate)
        } else {
            samples
        };
        apply_mode_filter(&mut resampled, target_rate, mode);
        let duration_secs = resampled.len() as f32 / target_rate as f32;
        AudioSignal {
            samples: resampled,
            sample_rate: target_rate,
            duration_secs,
            mode,
        }
    }
}

/// Apply the appropriate band-pass filter for the frequency mode.
/// Filters in-place to avoid redundant allocations.
fn apply_mode_filter(samples: &mut Vec<f32>, sample_rate: u32, mode: FrequencyMode) {
    // Apply high-pass filter if needed (removes low frequencies)
    if let Some(cutoff) = mode.highpass_cutoff() {
        biquad_highpass_inplace(samples, sample_rate, cutoff);
        // Apply twice for steeper rolloff (4th-order)
        biquad_highpass_inplace(samples, sample_rate, cutoff);
    }

    // Apply low-pass filter if needed (removes high frequencies)
    if let Some(cutoff) = mode.lowpass_cutoff() {
        biquad_lowpass_inplace(samples, sample_rate, cutoff);
        biquad_lowpass_inplace(samples, sample_rate, cutoff);
    }
}

/// 2nd-order Butterworth high-pass filter (biquad, in-place).
fn biquad_highpass_inplace(samples: &mut [f32], sample_rate: u32, cutoff: f32) {
    let omega = 2.0 * std::f64::consts::PI * cutoff as f64 / sample_rate as f64;
    let cos_omega = omega.cos();
    let alpha = omega.sin() / (2.0_f64.sqrt()); // Q = 1/√2 for Butterworth

    let b0 = (1.0 + cos_omega) / 2.0;
    let b1 = -(1.0 + cos_omega);
    let b2 = (1.0 + cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    apply_biquad_inplace(samples, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

/// 2nd-order Butterworth low-pass filter (biquad, in-place).
fn biquad_lowpass_inplace(samples: &mut [f32], sample_rate: u32, cutoff: f32) {
    let omega = 2.0 * std::f64::consts::PI * cutoff as f64 / sample_rate as f64;
    let cos_omega = omega.cos();
    let alpha = omega.sin() / (2.0_f64.sqrt());

    let b0 = (1.0 - cos_omega) / 2.0;
    let b1 = 1.0 - cos_omega;
    let b2 = (1.0 - cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    apply_biquad_inplace(samples, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

/// Apply a biquad filter with the given coefficients, mutating samples in-place.
fn apply_biquad_inplace(samples: &mut [f32], b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) {
    let mut x1: f64 = 0.0;
    let mut x2: f64 = 0.0;
    let mut y1: f64 = 0.0;
    let mut y2: f64 = 0.0;

    for sample in samples.iter_mut() {
        let x0 = *sample as f64;
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

        *sample = y0 as f32;

        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
}

/// Mix multi-channel audio to mono by averaging channels.
fn mix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Simple linear interpolation resampler.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        if idx + 1 < samples.len() {
            let val = samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac;
            output.push(val as f32);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }

    output
}

/// Generate a WAV file from samples (for testing).
pub fn write_wav<P: AsRef<Path>>(path: P, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(val)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Generate a sine wave tone (for testing / demo).
pub fn generate_tone(freq: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect()
}

/// Generate a composite signal with multiple frequencies (for testing).
pub fn generate_composite(
    freqs: &[(f32, f32)], // (frequency, amplitude)
    duration_secs: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            freqs
                .iter()
                .map(|(freq, amp)| amp * (2.0 * std::f32::consts::PI * freq * t).sin())
                .sum()
        })
        .collect()
}
