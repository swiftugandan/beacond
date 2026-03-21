//! Ambient sound signature computation and matching via vector similarity.
//!
//! Instead of exact hash matching (Shazam-style), this module computes a
//! fixed-length spectral envelope vector that captures the "character" of a
//! room's ambient sound — HVAC hum, electrical noise, resonance profile.
//!
//! Matching uses cosine similarity between signature vectors.

use crate::spectrogram::{Spectrogram, SpectrogramConfig};

/// Number of frequency bands in the signature vector.
/// Each band stores: mean energy, variance, spectral centroid weight.
const NUM_BANDS: usize = 64;

/// Features per band: mean, variance, peak ratio.
const FEATURES_PER_BAND: usize = 3;

/// Total signature vector length for the spectral backend.
pub const SIGNATURE_LEN: usize = NUM_BANDS * FEATURES_PER_BAND;

/// A sound signature vector (spectral or learned embedding).
#[derive(Debug, Clone)]
pub struct Signature {
    /// The feature vector — L2-normalized.
    /// Length is `SIGNATURE_LEN` (192) for spectral signatures or model-dependent
    /// for learned embeddings.
    pub vector: Vec<f32>,
}

impl Signature {
    /// Compute an ambient signature from raw audio samples.
    pub fn from_samples(samples: &[f32], sample_rate: u32) -> Self {
        let config = SpectrogramConfig::default();
        let spectrogram = Spectrogram::compute(samples, sample_rate, &config);
        Self::from_spectrogram(&spectrogram)
    }

    /// Compute a signature from a pre-computed spectrogram.
    /// Use this when you already have a spectrogram to avoid recomputing it.
    pub fn from_spectrogram(spectrogram: &Spectrogram) -> Self {
        if spectrogram.num_frames == 0 {
            return Signature {
                vector: vec![0.0; SIGNATURE_LEN],
            };
        }

        let num_bins = spectrogram.num_bins;
        let bins_per_band = num_bins / NUM_BANDS;

        let mut vector = Vec::with_capacity(SIGNATURE_LEN);

        for band in 0..NUM_BANDS {
            let bin_start = band * bins_per_band;
            let bin_end = if band == NUM_BANDS - 1 {
                num_bins
            } else {
                (band + 1) * bins_per_band
            };
            let band_width = bin_end - bin_start;

            let mut energy_sum = 0.0f32;
            let mut energy_sq_sum = 0.0f32;
            let mut peak_count = 0usize;
            let n = spectrogram.num_frames as f32;

            for f in 0..spectrogram.num_frames {
                let frame = spectrogram.frame(f);
                let band_slice = &frame[bin_start..bin_end];

                let band_energy: f32 =
                    band_slice.iter().map(|&m| m * m).sum::<f32>() / band_width as f32;
                energy_sum += band_energy;
                energy_sq_sum += band_energy * band_energy;

                // Count frames where this band has a dominant peak.
                let mut frame_max = 0.0f32;
                let mut frame_sum = 0.0f32;
                for &m in band_slice {
                    frame_sum += m;
                    if m > frame_max {
                        frame_max = m;
                    }
                }
                let frame_mean = frame_sum / band_width as f32;
                if frame_mean > 0.0 && frame_max / frame_mean > 3.0 {
                    peak_count += 1;
                }
            }

            // Feature 1: Mean energy (log scale).
            let mean = energy_sum / n;
            let log_mean = (mean + 1e-10).ln();

            // Feature 2: Variance (online formula avoids second pass).
            let variance = (energy_sq_sum / n) - (mean * mean);
            let log_variance = (variance.max(0.0) + 1e-10).ln();

            // Feature 3: Peak ratio.
            let peak_ratio = peak_count as f32 / n;

            vector.push(log_mean);
            vector.push(log_variance);
            vector.push(peak_ratio);
        }

        // L2-normalize so cosine similarity = dot product.
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        Signature { vector }
    }

    /// Cosine similarity between two signatures. Returns value in [-1, 1].
    pub fn similarity(&self, other: &Signature) -> f32 {
        // Vectors are already L2-normalized, so cosine similarity = dot product.
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Create a signature from a pre-computed embedding vector (e.g., from an ONNX model).
    /// The vector must already be L2-normalized.
    pub fn from_embedding(vector: Vec<f32>) -> Self {
        Signature { vector }
    }

    /// Serialize the vector to bytes for SQLite blob storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.vector.iter().flat_map(|&f| f.to_le_bytes()).collect()
    }

    /// Deserialize from bytes (SQLite blob).
    /// Accepts any length that is a multiple of 4 bytes (variable-dim embeddings).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() % 4 != 0 || bytes.is_empty() {
            return None;
        }
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Some(Signature { vector })
    }

    /// Returns the dimensionality of this signature.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{generate_composite, generate_tone};

    #[test]
    fn test_same_signal_high_similarity() {
        let signal = generate_composite(&[(100.0, 0.5), (200.0, 0.3), (400.0, 0.2)], 5.0, 16000);
        let sig_a = Signature::from_samples(&signal, 16000);
        let sig_b = Signature::from_samples(&signal, 16000);
        let sim = sig_a.similarity(&sig_b);
        assert!(
            sim > 0.99,
            "Same signal should have near-perfect similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_different_signals_lower_similarity() {
        // Use a higher sample rate so 64 bands spread across more distinct regions.
        let sr = 44100;
        // Low-frequency hum signal.
        let signal_a = generate_composite(&[(60.0, 1.0), (120.0, 0.5)], 5.0, sr);
        // High-frequency signal with many harmonics.
        let signal_b = generate_composite(
            &[
                (5000.0, 0.5),
                (10000.0, 0.4),
                (15000.0, 0.3),
                (18000.0, 0.2),
            ],
            5.0,
            sr,
        );
        let sig_a = Signature::from_samples(&signal_a, sr);
        let sig_b = Signature::from_samples(&signal_b, sr);
        let sim = sig_a.similarity(&sig_b);
        // With synthetic tones, many bands are near-silent and contribute shared structure.
        // Real ambient recordings diverge much more. Here we just verify they're less similar
        // than identical signals (which score >0.99).
        assert!(
            sim < 0.95,
            "Spectrally distant signals should have lower similarity than identical ones, got {}",
            sim
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let signal = generate_tone(440.0, 3.0, 16000);
        let sig = Signature::from_samples(&signal, 16000);
        let bytes = sig.to_bytes();
        let recovered = Signature::from_bytes(&bytes).unwrap();
        assert_eq!(sig.vector.len(), recovered.vector.len());
        for (a, b) in sig.vector.iter().zip(recovered.vector.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_signature_length() {
        let signal = generate_tone(440.0, 3.0, 16000);
        let sig = Signature::from_samples(&signal, 16000);
        assert_eq!(sig.vector.len(), SIGNATURE_LEN);
    }
}
