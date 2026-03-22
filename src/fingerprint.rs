//! Audio fingerprinting using constellation map + combinatorial hashing.
//!
//! This implements the core Shazam algorithm:
//! 1. Compute spectrogram
//! 2. Find spectral peaks (constellation points)
//! 3. Form fingerprint hashes by pairing peaks in a target zone
//! 4. Each hash encodes: (freq1, freq2, time_delta) → anchor_time

use crate::spectrogram::{
    compute_frame_magnitudes_squared, frame_count, hann_window, Spectrogram, SpectrogramConfig,
};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use xxhash_rust::xxh3::xxh3_64;

// ── Peak-finding parameters ──────────────────────────────────────────

/// Number of frequency bands for peak extraction.
const NUM_BANDS: usize = 6;

/// Maximum number of peaks per band per frame.
const MAX_PEAKS_PER_BAND: usize = 3;

/// Minimum magnitude threshold for a peak (in linear scale).
const PEAK_THRESHOLD: f32 = 0.01;

/// Size of the local neighbourhood for peak detection (frames on each side).
const PEAK_NEIGHBOURHOOD_TIME: usize = 5;
/// Size of the local neighbourhood for peak detection (bins on each side).
const PEAK_NEIGHBOURHOOD_FREQ: usize = 5;

// ── Combinatorial hashing parameters ─────────────────────────────────

/// Target zone: how far ahead in time to look for pair peaks.
const TARGET_ZONE_T_MIN: usize = 1;
const TARGET_ZONE_T_MAX: usize = 60;

/// Target zone: frequency range around anchor peak.
const TARGET_ZONE_F_RANGE: usize = 100;

/// Maximum number of pairs per anchor point.
const MAX_PAIRS_PER_ANCHOR: usize = 5;

// ── Types ────────────────────────────────────────────────────────────

/// A spectral peak in the constellation map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Time frame index.
    pub frame: usize,
    /// Frequency bin index.
    pub bin: usize,
    /// Magnitude at this peak.
    pub magnitude: f32,
}

/// A fingerprint hash with its time offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    /// The hash value encoding (freq1, freq2, dt).
    pub hash: u64,
    /// The absolute time offset (frame index of anchor peak).
    pub offset: u32,
}

/// Configuration for the fingerprinting engine.
#[derive(Debug, Clone)]
pub struct FingerprintConfig {
    pub spectrogram: SpectrogramConfig,
    pub num_bands: usize,
    pub max_peaks_per_band: usize,
    pub peak_threshold: f32,
    /// Use adaptive per-band threshold instead of fixed peak_threshold.
    pub adaptive_threshold: bool,
    pub target_zone_t_min: usize,
    pub target_zone_t_max: usize,
    pub target_zone_f_range: usize,
    pub max_pairs_per_anchor: usize,
    /// Use log-spaced frequency bands instead of linear.
    pub log_frequency_bands: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        FingerprintConfig {
            spectrogram: SpectrogramConfig::default(),
            num_bands: NUM_BANDS,
            max_peaks_per_band: MAX_PEAKS_PER_BAND,
            peak_threshold: PEAK_THRESHOLD,
            adaptive_threshold: true,
            target_zone_t_min: TARGET_ZONE_T_MIN,
            target_zone_t_max: TARGET_ZONE_T_MAX,
            target_zone_f_range: TARGET_ZONE_F_RANGE,
            max_pairs_per_anchor: MAX_PAIRS_PER_ANCHOR,
            log_frequency_bands: true,
        }
    }
}

impl FingerprintConfig {
    /// A compact configuration that produces ~60% fewer fingerprints.
    /// Useful for resource-constrained devices or when storage is limited.
    pub fn compact() -> Self {
        FingerprintConfig {
            spectrogram: SpectrogramConfig::default(),
            num_bands: 4,
            max_peaks_per_band: 2,
            peak_threshold: PEAK_THRESHOLD,
            adaptive_threshold: true,
            target_zone_t_min: TARGET_ZONE_T_MIN,
            target_zone_t_max: 30,
            target_zone_f_range: TARGET_ZONE_F_RANGE,
            max_pairs_per_anchor: 3,
            log_frequency_bands: true,
        }
    }
}

/// Compute frequency band boundaries.
///
/// With `log_scale = true`, bands are log-spaced so lower frequencies get narrower
/// bands (higher resolution) and higher frequencies get wider bands, matching
/// how acoustic content is distributed. With `log_scale = false`, bands are
/// equal-width (linear).
fn band_boundaries(num_bins: usize, num_bands: usize, log_scale: bool) -> Vec<(usize, usize)> {
    if !log_scale || num_bins <= 1 {
        // Linear: equal-width bands
        let band_size = num_bins / num_bands;
        return (0..num_bands)
            .map(|b| {
                let start = b * band_size;
                let end = if b == num_bands - 1 {
                    num_bins
                } else {
                    (b + 1) * band_size
                };
                (start, end)
            })
            .collect();
    }

    // Log-spaced: boundaries grow exponentially.
    // Start from bin 1 (skip DC bin 0) to avoid log(0).
    // Chain boundaries so each band starts where the previous ended (no overlaps or gaps).
    let min_bin = 1.0f32;
    let max_bin = num_bins as f32;
    let log_min = min_bin.ln();
    let log_max = max_bin.ln();

    let mut bounds = Vec::with_capacity(num_bands);
    let mut prev_end = 0usize; // first band starts at bin 0 (includes DC)
    for b in 0..num_bands {
        let end_f = (log_min + (log_max - log_min) * (b + 1) as f32 / num_bands as f32).exp();
        let end = if b == num_bands - 1 {
            num_bins // last band always reaches the end
        } else {
            (end_f as usize).min(num_bins)
        };
        // Only add non-empty bands; if empty, next band will cover this range
        if prev_end < end {
            bounds.push((prev_end, end));
        }
        prev_end = end;
    }
    bounds
}

/// The fingerprinting engine.
#[derive(Clone)]
pub struct Fingerprinter {
    pub config: FingerprintConfig,
}

impl Fingerprinter {
    pub fn new(config: FingerprintConfig) -> Self {
        Fingerprinter { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(FingerprintConfig::default())
    }

    /// Create a fingerprinter with compact settings (fewer, sparser fingerprints).
    pub fn with_compact() -> Self {
        Self::new(FingerprintConfig::compact())
    }

    /// Generate fingerprints from audio samples.
    pub fn fingerprint(&self, samples: &[f32], sample_rate: u32) -> Vec<Fingerprint> {
        let spectrogram = Spectrogram::compute(samples, sample_rate, &self.config.spectrogram);
        self.fingerprint_spectrogram(&spectrogram)
    }

    /// Generate fingerprints from a pre-computed spectrogram.
    /// Use this when you already have a spectrogram to avoid recomputing it.
    pub fn fingerprint_spectrogram(&self, spectrogram: &Spectrogram) -> Vec<Fingerprint> {
        if spectrogram.num_frames == 0 {
            log::warn!("Empty spectrogram, no fingerprints generated");
            return Vec::new();
        }

        let peaks = self.find_peaks(spectrogram);

        log::debug!(
            "Found {} peaks in {} frames",
            peaks.len(),
            spectrogram.num_frames
        );

        let fingerprints = self.generate_hashes(&peaks);

        log::debug!("Generated {} fingerprint hashes", fingerprints.len());

        fingerprints
    }

    /// Generate fingerprints using a memory-efficient sparse pipeline.
    ///
    /// Instead of building the full spectrogram in memory, this processes frames
    /// through a rolling buffer just large enough for local-maximum validation.
    /// Memory usage is O(neighbourhood_frames * num_bins) instead of
    /// O(total_frames * num_bins).
    pub fn fingerprint_sparse(&self, samples: &[f32], _sample_rate: u32) -> Vec<Fingerprint> {
        let window_size = self.config.spectrogram.window_size;
        let hop_size = self.config.spectrogram.hop_size;
        let num_bins = window_size / 2 + 1;
        let total_frames = frame_count(samples.len(), window_size, hop_size);

        if total_frames == 0 {
            log::warn!("Not enough samples for sparse fingerprinting");
            return Vec::new();
        }

        let hann_win = if self.config.spectrogram.apply_window {
            hann_window(window_size)
        } else {
            vec![1.0; window_size]
        };

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window_size);
        let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); window_size];

        // Use squared magnitudes throughout — avoids sqrt per bin.
        // The threshold must also be squared for consistent comparison.
        let threshold_sq = self.config.peak_threshold * self.config.peak_threshold;

        // Rolling buffer: stores up to (2*PEAK_NEIGHBOURHOOD_TIME + 1) frames.
        // ring_start tracks which absolute frame index ring[0] corresponds to.
        let ring_cap = PEAK_NEIGHBOURHOOD_TIME * 2 + 1;
        let mut ring: VecDeque<Vec<f32>> = VecDeque::with_capacity(ring_cap);
        let mut ring_start: usize = 0; // absolute frame of ring[0]
        let mut peaks: Vec<Peak> = Vec::new();

        let bands =
            band_boundaries(num_bins, self.config.num_bands, self.config.log_frequency_bands);

        // Pre-fill: compute frames [0, min(ring_cap, total_frames))
        let prefill = ring_cap.min(total_frames);
        for f in 0..prefill {
            let mut mags = vec![0.0f32; num_bins];
            compute_frame_magnitudes_squared(
                samples, f, hop_size, &hann_win, fft.as_ref(), &mut fft_buffer, &mut mags,
            );
            ring.push_back(mags);
        }
        // ring now holds frames [0..prefill), ring_start = 0

        // Pre-compute per-band adaptive thresholds from the pre-fill frames (squared).
        // Uses mean energy across pre-fill frames, scaled by 2× (in squared domain: 4×).
        let band_thresholds_sq: Vec<f32> = if self.config.adaptive_threshold && !ring.is_empty() {
            bands
                .iter()
                .map(|&(bin_start, bin_end)| {
                    let band_width = (bin_end - bin_start).max(1) as f32;
                    let mean_energy: f32 = ring
                        .iter()
                        .map(|frame| {
                            frame[bin_start..bin_end].iter().sum::<f32>() / band_width
                        })
                        .sum::<f32>()
                        / ring.len() as f32;
                    // 2× mean energy in squared domain = 4× in linear domain
                    (mean_energy * 2.0).max(threshold_sq)
                })
                .collect()
        } else {
            vec![threshold_sq; bands.len()]
        };

        for center_frame in 0..total_frames {
            // The neighbourhood in absolute frame indices
            let t_start = center_frame.saturating_sub(PEAK_NEIGHBOURHOOD_TIME);
            let t_end = (center_frame + PEAK_NEIGHBOURHOOD_TIME + 1).min(total_frames);

            // Evict frames that are before the neighbourhood
            while ring_start < t_start {
                ring.pop_front();
                ring_start += 1;
            }

            // Compute any frames we need but haven't computed yet
            let ring_end = ring_start + ring.len(); // exclusive absolute frame
            for f in ring_end..t_end {
                let mut mags = vec![0.0f32; num_bins];
                compute_frame_magnitudes_squared(
                    samples, f, hop_size, &hann_win, fft.as_ref(), &mut fft_buffer, &mut mags,
                );
                ring.push_back(mags);
            }

            // center_frame's data is at ring index (center_frame - ring_start)
            let center_idx = center_frame - ring_start;

            // Spectral flatness gating: skip noise-dominated frames (consistent with find_peaks).
            // Flatness = geo_mean / arith_mean, computed from squared magnitudes.
            {
                let frame_mags = &ring[center_idx];
                let n = frame_mags.len() as f32;
                let arith_mean = frame_mags.iter().sum::<f32>() / n;
                if arith_mean > 0.0 {
                    let log_sum: f32 = frame_mags.iter().map(|&m| (m + 1e-20).ln()).sum();
                    let geo_mean = (log_sum / n).exp();
                    let flatness = (geo_mean / arith_mean).clamp(0.0, 1.0);
                    if flatness > 0.8 {
                        continue; // skip noise-dominated frame
                    }
                }
            }

            for (band_idx, &(bin_start, bin_end)) in bands.iter().enumerate() {
                let band_threshold = band_thresholds_sq[band_idx];
                let mut band_peaks: Vec<Peak> = Vec::new();

                for bin in bin_start..bin_end {
                    let mag = ring[center_idx][bin];
                    if mag < band_threshold {
                        continue;
                    }

                    // Check local maximum against all frames in [t_start, t_end)
                    let f_start = bin.saturating_sub(PEAK_NEIGHBOURHOOD_FREQ);
                    let f_end = (bin + PEAK_NEIGHBOURHOOD_FREQ + 1).min(num_bins);
                    let mut is_max = true;

                    'outer: for t in t_start..t_end {
                        let ri = t - ring_start;
                        for f in f_start..f_end {
                            if t == center_frame && f == bin {
                                continue;
                            }
                            if ring[ri][f] > mag {
                                is_max = false;
                                break 'outer;
                            }
                        }
                    }

                    if is_max {
                        band_peaks.push(Peak {
                            frame: center_frame,
                            bin,
                            magnitude: mag,
                        });
                    }
                }

                let max = self.config.max_peaks_per_band;
                if band_peaks.len() > max {
                    band_peaks.select_nth_unstable_by(max - 1, |a, b| {
                        b.magnitude.partial_cmp(&a.magnitude).unwrap()
                    });
                    band_peaks.truncate(max);
                }
                peaks.extend(band_peaks);
            }
        }

        peaks.sort_by_key(|p| p.frame);

        log::debug!(
            "Sparse: found {} peaks in {} frames",
            peaks.len(),
            total_frames
        );

        let fingerprints = self.generate_hashes(&peaks);
        log::debug!("Sparse: generated {} fingerprint hashes", fingerprints.len());
        fingerprints
    }

    /// Find spectral peaks using band-based extraction with local maximum detection.
    fn find_peaks(&self, spectrogram: &Spectrogram) -> Vec<Peak> {
        let num_bins = spectrogram.num_bins;
        let num_frames = spectrogram.num_frames;
        let bands =
            band_boundaries(num_bins, self.config.num_bands, self.config.log_frequency_bands);

        // Pre-compute per-band adaptive thresholds if enabled.
        // Uses the mean energy across all frames in the band, then scales up.
        // This adapts to recording volume while keeping the threshold below peaks.
        let band_thresholds: Vec<f32> = if self.config.adaptive_threshold {
            bands
                .iter()
                .map(|&(bin_start, bin_end)| {
                    let band_width = (bin_end - bin_start).max(1) as f32;
                    let mean_energy: f32 = (0..num_frames)
                        .map(|f| {
                            let frame = spectrogram.frame(f);
                            frame[bin_start..bin_end].iter().sum::<f32>() / band_width
                        })
                        .sum::<f32>()
                        / num_frames as f32;
                    // Threshold at 2× mean energy, floored at absolute minimum
                    (mean_energy * 2.0).max(self.config.peak_threshold)
                })
                .collect()
        } else {
            vec![self.config.peak_threshold; bands.len()]
        };

        let mut all_peaks = Vec::new();

        for frame in 0..num_frames {
            // Skip noise-dominated frames (spectral flatness > 0.8).
            if spectrogram.spectral_flatness(frame) > 0.8 {
                continue;
            }

            let mags = spectrogram.frame(frame);

            for (band_idx, &(bin_start, bin_end)) in bands.iter().enumerate() {
                let threshold = band_thresholds[band_idx];
                let mut band_peaks: Vec<Peak> = Vec::new();

                #[allow(clippy::needless_range_loop)]
                for bin in bin_start..bin_end {
                    let mag = mags[bin];
                    if mag < threshold {
                        continue;
                    }

                    // Check if this is a local maximum
                    if self.is_local_maximum(spectrogram, frame, bin, mag) {
                        band_peaks.push(Peak {
                            frame,
                            bin,
                            magnitude: mag,
                        });
                    }
                }

                // Keep only top N peaks per band using O(n) partial sort
                let max = self.config.max_peaks_per_band;
                if band_peaks.len() > max {
                    band_peaks.select_nth_unstable_by(max - 1, |a, b| {
                        b.magnitude.partial_cmp(&a.magnitude).unwrap()
                    });
                    band_peaks.truncate(max);
                }
                all_peaks.extend(band_peaks);
            }
        }

        // Sort by time for hash generation
        all_peaks.sort_by_key(|p| p.frame);
        all_peaks
    }

    /// Check if a point is a local maximum in its neighbourhood.
    fn is_local_maximum(
        &self,
        spectrogram: &Spectrogram,
        frame: usize,
        bin: usize,
        mag: f32,
    ) -> bool {
        let t_start = frame.saturating_sub(PEAK_NEIGHBOURHOOD_TIME);
        let t_end = (frame + PEAK_NEIGHBOURHOOD_TIME + 1).min(spectrogram.num_frames);
        let f_start = bin.saturating_sub(PEAK_NEIGHBOURHOOD_FREQ);
        let f_end = (bin + PEAK_NEIGHBOURHOOD_FREQ + 1).min(spectrogram.num_bins);

        for t in t_start..t_end {
            for f in f_start..f_end {
                if t == frame && f == bin {
                    continue;
                }
                if spectrogram.mag(t, f) > mag {
                    return false;
                }
            }
        }
        true
    }

    /// Generate combinatorial hashes from constellation peaks.
    fn generate_hashes(&self, peaks: &[Peak]) -> Vec<Fingerprint> {
        let mut fingerprints = Vec::new();

        for (i, anchor) in peaks.iter().enumerate() {
            let mut pairs = 0;

            for target in &peaks[i + 1..] {
                // Check time delta is within target zone
                let dt = target.frame - anchor.frame;
                if dt < self.config.target_zone_t_min {
                    continue;
                }
                if dt > self.config.target_zone_t_max {
                    break; // Peaks are sorted by time
                }

                // Check frequency range
                let df = (target.bin as isize - anchor.bin as isize).unsigned_abs();
                if df > self.config.target_zone_f_range {
                    continue;
                }

                // Create hash: encode (freq1, freq2, dt)
                let hash = compute_hash(anchor.bin as u32, target.bin as u32, dt as u32);

                fingerprints.push(Fingerprint {
                    hash,
                    offset: anchor.frame as u32,
                });

                pairs += 1;
                if pairs >= self.config.max_pairs_per_anchor {
                    break;
                }
            }
        }

        fingerprints
    }
}

/// Compute a hash from anchor frequency, target frequency, and time delta.
fn compute_hash(freq1: u32, freq2: u32, dt: u32) -> u64 {
    let mut data = [0u8; 12];
    data[0..4].copy_from_slice(&freq1.to_le_bytes());
    data[4..8].copy_from_slice(&freq2.to_le_bytes());
    data[8..12].copy_from_slice(&dt.to_le_bytes());
    xxh3_64(&data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{generate_composite, generate_tone};

    #[test]
    fn test_fingerprint_deterministic() {
        let samples = generate_tone(440.0, 3.0, 16000);
        let fp = Fingerprinter::with_defaults();
        let a = fp.fingerprint(&samples, 16000);
        let b = fp.fingerprint(&samples, 16000);
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa.hash, fb.hash);
            assert_eq!(fa.offset, fb.offset);
        }
    }

    #[test]
    fn test_sparse_matches_full_pipeline() {
        let samples = generate_composite(&[(440.0, 0.5), (880.0, 0.3), (1320.0, 0.2)], 3.0, 16000);
        let fp = Fingerprinter::with_defaults();

        let full = fp.fingerprint(&samples, 16000);
        let sparse = fp.fingerprint_sparse(&samples, 16000);

        // Both should produce the same fingerprints
        assert_eq!(full.len(), sparse.len(), "Full ({}) and sparse ({}) fingerprint counts differ", full.len(), sparse.len());
        for (f, s) in full.iter().zip(sparse.iter()) {
            assert_eq!(f.hash, s.hash, "Hash mismatch at offset {}", f.offset);
            assert_eq!(f.offset, s.offset);
        }
    }

    #[test]
    fn test_compact_produces_fewer_fingerprints() {
        let samples = generate_composite(
            &[(440.0, 0.5), (880.0, 0.3), (1320.0, 0.2), (220.0, 0.4)],
            5.0,
            16000,
        );
        let default_fp = Fingerprinter::with_defaults();
        let compact_fp = Fingerprinter::with_compact();

        let default_count = default_fp.fingerprint(&samples, 16000).len();
        let compact_count = compact_fp.fingerprint(&samples, 16000).len();

        assert!(
            compact_count < default_count,
            "Compact ({}) should produce fewer fingerprints than default ({})",
            compact_count,
            default_count
        );
    }

    #[test]
    fn test_different_signals_different_fingerprints() {
        let fp = Fingerprinter::with_defaults();

        let sig_a = generate_composite(&[(440.0, 0.5), (880.0, 0.3), (1320.0, 0.2)], 3.0, 16000);
        let sig_b = generate_composite(&[(300.0, 0.5), (600.0, 0.3), (900.0, 0.2)], 3.0, 16000);

        let fps_a = fp.fingerprint(&sig_a, 16000);
        let fps_b = fp.fingerprint(&sig_b, 16000);

        // The hash sets should be mostly different
        let set_a: std::collections::HashSet<u64> = fps_a.iter().map(|f| f.hash).collect();
        let set_b: std::collections::HashSet<u64> = fps_b.iter().map(|f| f.hash).collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        // Jaccard similarity should be low
        let similarity = intersection as f64 / union as f64;
        assert!(
            similarity < 0.5,
            "Fingerprints too similar: {:.2}",
            similarity
        );
    }
}
