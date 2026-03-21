//! Beacon — Ultrasonic spatial awareness toolkit.
//!
//! Audio fingerprinting engine for zone detection and spatial awareness:
//! - Spectrogram computation via FFT
//! - Constellation map peak extraction
//! - Combinatorial hashing (anchor + target zone pairing)
//! - Offset histogram alignment for robust matching

pub mod audio;
pub mod cli;
pub mod daemon;
pub mod database;
pub mod fingerprint;
pub mod microphone;
pub mod signature;
pub mod spectrogram;
pub mod spectrogram_server;
