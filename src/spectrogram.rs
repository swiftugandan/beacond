//! Spectrogram computation using FFT.

use rustfft::{num_complex::Complex, FftPlanner};

/// Parameters for spectrogram generation.
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size between successive windows.
    pub hop_size: usize,
    /// Apply Hann window before FFT.
    pub apply_window: bool,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        SpectrogramConfig {
            window_size: 1024,
            hop_size: 512,
            apply_window: true,
        }
    }
}

/// A computed spectrogram stored as a flat, contiguous buffer.
///
/// Magnitudes are stored in row-major order: `[frame * num_bins + bin]`.
/// This gives better cache locality than `Vec<Vec<f32>>` and eliminates
/// per-frame heap allocations.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Flat magnitude data: `num_frames * num_bins` elements, row-major.
    pub magnitudes: Vec<f32>,
    /// Number of time frames.
    pub num_frames: usize,
    /// Number of frequency bins per frame (window_size / 2 + 1).
    pub num_bins: usize,
    /// Hop size used.
    pub hop_size: usize,
    /// Sample rate of the source audio.
    pub sample_rate: u32,
}

/// Compute a Hann window of the given size.
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
}

impl Spectrogram {
    /// Access magnitude at `[frame][bin]`.
    #[inline(always)]
    pub fn mag(&self, frame: usize, bin: usize) -> f32 {
        self.magnitudes[frame * self.num_bins + bin]
    }

    /// Get a slice of magnitudes for an entire frame.
    #[inline(always)]
    pub fn frame(&self, frame: usize) -> &[f32] {
        let start = frame * self.num_bins;
        &self.magnitudes[start..start + self.num_bins]
    }

    /// Compute a spectrogram from audio samples.
    pub fn compute(samples: &[f32], sample_rate: u32, config: &SpectrogramConfig) -> Self {
        let window_size = config.window_size;
        let hop_size = config.hop_size;
        let num_bins = window_size / 2 + 1;

        // Pre-compute Hann window
        let hann_win: Vec<f32> = if config.apply_window {
            hann_window(window_size)
        } else {
            vec![1.0; window_size]
        };

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window_size);

        let num_frames = if samples.len() >= window_size {
            (samples.len() - window_size) / hop_size + 1
        } else {
            0
        };

        // Single contiguous allocation.
        let mut magnitudes = vec![0.0f32; num_frames * num_bins];
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); window_size];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + window_size).min(samples.len());

            // Fill buffer with windowed samples.
            for (i, &w) in hann_win[..end - start].iter().enumerate() {
                buffer[i] = Complex::new(samples[start + i] * w, 0.0);
            }
            // Zero-pad if needed.
            for b in &mut buffer[end - start..] {
                *b = Complex::new(0.0, 0.0);
            }

            // FFT in-place.
            fft.process(&mut buffer);

            // Write magnitudes directly into flat buffer.
            let out = &mut magnitudes[frame_idx * num_bins..(frame_idx + 1) * num_bins];
            for (o, c) in out.iter_mut().zip(buffer[..num_bins].iter()) {
                *o = (c.re * c.re + c.im * c.im).sqrt();
            }
        }

        Spectrogram {
            magnitudes,
            num_frames,
            num_bins,
            hop_size,
            sample_rate,
        }
    }

    /// Convert a frequency bin index to Hz.
    pub fn bin_to_freq(&self, bin: usize) -> f32 {
        let window_size = (self.num_bins - 1) * 2;
        bin as f32 * self.sample_rate as f32 / window_size as f32
    }

    /// Convert a frame index to seconds.
    pub fn frame_to_time(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_size as f32 / self.sample_rate as f32
    }

    /// Compute spectral flatness for a frame.
    ///
    /// Spectral flatness = geometric_mean / arithmetic_mean of the magnitude spectrum.
    /// Returns a value in [0, 1]: 0 = tonal (peaked), 1 = white noise (flat).
    /// Frames with flatness > threshold can be skipped during peak extraction
    /// to avoid extracting noise-derived peaks.
    pub fn spectral_flatness(&self, frame: usize) -> f32 {
        let mags = self.frame(frame);
        let n = mags.len() as f32;
        if n == 0.0 {
            return 1.0;
        }
        let arith_mean = mags.iter().sum::<f32>() / n;
        if arith_mean <= 0.0 {
            return 1.0; // silence → treat as noise
        }
        // Geometric mean via exp(mean(ln(x))). Add epsilon to avoid ln(0).
        let log_sum: f32 = mags.iter().map(|&m| (m + 1e-10).ln()).sum();
        let geo_mean = (log_sum / n).exp();
        (geo_mean / arith_mean).clamp(0.0, 1.0)
    }
}

/// Compute a single frame's magnitude spectrum into an output buffer.
/// This is exposed for sparse peak extraction to avoid storing the full spectrogram.
pub fn compute_frame_magnitudes(
    samples: &[f32],
    frame_idx: usize,
    hop_size: usize,
    hann_win: &[f32],
    fft: &dyn rustfft::Fft<f32>,
    fft_buffer: &mut [Complex<f32>],
    out: &mut [f32],
) {
    compute_frame_impl(samples, frame_idx, hop_size, hann_win, fft, fft_buffer, out, false);
}

/// Like `compute_frame_magnitudes` but stores squared magnitudes (skips sqrt).
/// Use this when only relative ordering matters (e.g. peak detection).
pub fn compute_frame_magnitudes_squared(
    samples: &[f32],
    frame_idx: usize,
    hop_size: usize,
    hann_win: &[f32],
    fft: &dyn rustfft::Fft<f32>,
    fft_buffer: &mut [Complex<f32>],
    out: &mut [f32],
) {
    compute_frame_impl(samples, frame_idx, hop_size, hann_win, fft, fft_buffer, out, true);
}

fn compute_frame_impl(
    samples: &[f32],
    frame_idx: usize,
    hop_size: usize,
    hann_win: &[f32],
    fft: &dyn rustfft::Fft<f32>,
    fft_buffer: &mut [Complex<f32>],
    out: &mut [f32],
    squared: bool,
) {
    let window_size = hann_win.len();
    let start = frame_idx * hop_size;
    let end = (start + window_size).min(samples.len());

    // Fill buffer with windowed samples.
    for (i, &w) in hann_win[..end - start].iter().enumerate() {
        fft_buffer[i] = Complex::new(samples[start + i] * w, 0.0);
    }
    // Zero-pad if needed.
    for b in &mut fft_buffer[end - start..] {
        *b = Complex::new(0.0, 0.0);
    }

    fft.process(fft_buffer);

    let num_bins = out.len();
    if squared {
        for (o, c) in out.iter_mut().zip(fft_buffer[..num_bins].iter()) {
            *o = c.re * c.re + c.im * c.im;
        }
    } else {
        for (o, c) in out.iter_mut().zip(fft_buffer[..num_bins].iter()) {
            *o = (c.re * c.re + c.im * c.im).sqrt();
        }
    }
}

/// Count how many frames fit in the given number of samples.
pub fn frame_count(num_samples: usize, window_size: usize, hop_size: usize) -> usize {
    if num_samples >= window_size {
        (num_samples - window_size) / hop_size + 1
    } else {
        0
    }
}
