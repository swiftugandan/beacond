//! Learned audio embeddings via ONNX Runtime.
//!
//! Replaces or augments the handcrafted 192-dim spectral signatures with
//! embeddings from a pre-trained audio model (e.g., YAMNet, OpenL3).
//! The model runs locally via ONNX Runtime — no cloud dependencies.
//!
//! # Model requirements
//!
//! The ONNX model must accept a 2D input tensor of shape `[1, num_samples]`
//! (mono 16 kHz float32 audio) and produce a 2D output tensor of shape
//! `[1, embedding_dim]` or `[1, num_frames, embedding_dim]` (per-frame
//! embeddings get average-pooled). Most audio embedding models (YAMNet,
//! OpenL3, VGGish) conform to this after export.
//!
//! # Runtime linking
//!
//! Built with `load-dynamic` — requires the ONNX Runtime shared library
//! to be available at runtime (via `ORT_DYLIB_PATH` env var or system path).

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Mutex;

/// An audio embedding model backed by ONNX Runtime.
///
/// The session is wrapped in a Mutex because `ort::Session::run` requires
/// `&mut self` but we share the embedder from `DaemonState` behind `Arc<Mutex>`.
pub struct AudioEmbedder {
    session: Mutex<ort::session::Session>,
    /// Expected sample rate for the model (typically 16000).
    pub model_sample_rate: u32,
    /// Dimensionality of the output embedding.
    pub embedding_dim: usize,
}

impl AudioEmbedder {
    /// Load an ONNX model from disk.
    ///
    /// The model is probed to determine its output embedding dimension.
    /// Returns an error if the model file is missing or incompatible.
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create ONNX session builder: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("Failed to set thread count: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load ONNX model from {}: {e}",
                    model_path.display()
                )
            })?;

        // Probe output shape to determine embedding dimension.
        let embedding_dim = session
            .outputs()
            .first()
            .and_then(|outlet| match outlet.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    // Take the last dimension as embedding dim.
                    shape.last().and_then(|&d| {
                        if d > 0 {
                            Some(d as usize)
                        } else {
                            None
                        }
                    })
                }
                _ => None,
            })
            .unwrap_or(512); // fallback if dynamic or unknown

        log::info!(
            "Loaded audio embedding model: {} ({}D embeddings)",
            model_path.display(),
            embedding_dim,
        );

        Ok(AudioEmbedder {
            session: Mutex::new(session),
            model_sample_rate: 16000,
            embedding_dim,
        })
    }

    /// Produce an L2-normalized embedding from raw audio samples.
    ///
    /// The input is resampled to the model's expected sample rate if needed,
    /// then fed through the ONNX model. The output embedding is L2-normalized
    /// so that cosine similarity == dot product.
    pub fn embed(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Resample to model sample rate if needed.
        let resampled;
        let input_samples = if sample_rate != self.model_sample_rate {
            resampled = crate::audio::resample(samples, sample_rate, self.model_sample_rate);
            &resampled
        } else {
            samples
        };

        // Zero-pad to a multiple of 160 samples (10ms frames at 16kHz).
        // Many audio models internally reshape into fixed-size frames.
        const FRAME_ALIGN: usize = 160;
        let mut padded = input_samples.to_vec();
        let remainder = padded.len() % FRAME_ALIGN;
        if remainder != 0 {
            padded.resize(padded.len() + FRAME_ALIGN - remainder, 0.0);
        }

        // Build input tensor: [1, num_samples].
        let input_len = padded.len();
        let input_array =
            ndarray::Array2::from_shape_vec((1, input_len), padded)
                .context("Failed to create input tensor")?;

        // Create a Tensor from the ndarray, then run inference.
        let input_tensor = ort::value::Tensor::from_array(input_array)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| anyhow::anyhow!("ONNX inference failed: {e}"))?;

        // Extract embedding from first output.
        let output_value = &outputs[0];
        let (shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract f32 tensor from model output: {e}"))?;

        // Average-pool if the model returns per-frame embeddings [1, frames, dim].
        let dims: Vec<i64> = shape.iter().copied().collect();
        let mut embedding = if dims.len() == 3 {
            // [1, num_frames, embedding_dim] — average across frames.
            let num_frames = dims[1] as usize;
            let dim = dims[2] as usize;
            let mut pooled = vec![0.0f32; dim];
            for frame in 0..num_frames {
                let offset = frame * dim;
                for d in 0..dim {
                    pooled[d] += data[offset + d];
                }
            }
            let scale = 1.0 / num_frames as f32;
            for v in &mut pooled {
                *v *= scale;
            }
            pooled
        } else {
            // [1, embedding_dim] or flat — use directly.
            data.to_vec()
        };

        // L2-normalize.
        let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_requires_model_file() {
        let result = AudioEmbedder::load(Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err(), "Should fail with missing model file");
    }
}
