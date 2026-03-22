//! Bloom filter for fast hash pre-screening in fingerprint lookups.
//!
//! A bloom filter provides O(1) set-membership queries with no false negatives
//! and a configurable false-positive rate. By checking the bloom filter before
//! hitting SQLite, we skip the vast majority of queries for hashes that don't
//! exist in the database.

use xxhash_rust::xxh3::xxh3_64;

/// A simple bloom filter backed by a bit vector.
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
}

impl BloomFilter {
    /// Create a bloom filter sized for `expected_items` with the given
    /// false-positive rate (e.g. 0.01 for 1%).
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let expected_items = expected_items.max(1);
        // Optimal number of bits: m = -n * ln(p) / (ln(2))^2
        let num_bits = (-(expected_items as f64) * fp_rate.ln() / (2.0_f64.ln().powi(2)))
            .ceil() as usize;
        let num_bits = num_bits.max(64);
        // Optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            ((num_bits as f64 / expected_items as f64) * 2.0_f64.ln()).ceil() as u32;
        let num_hashes = num_hashes.clamp(1, 16);

        let words = (num_bits + 63) / 64;
        BloomFilter {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
        }
    }

    /// Insert a hash into the filter.
    pub fn insert(&mut self, hash: u64) {
        for i in 0..self.num_hashes {
            let bit = self.bit_index(hash, i);
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    /// Check if a hash might be in the set. Returns `false` only if the hash
    /// is definitely absent; `true` means it may or may not be present.
    pub fn may_contain(&self, hash: u64) -> bool {
        for i in 0..self.num_hashes {
            let bit = self.bit_index(hash, i);
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Build a bloom filter from an iterator of hashes.
    pub fn from_hashes(iter: impl Iterator<Item = u64>, expected_items: usize) -> Self {
        let mut bf = Self::new(expected_items, 0.01);
        for h in iter {
            bf.insert(h);
        }
        bf
    }

    /// Derive a bit index from the original hash and a hash-function index.
    /// Uses xxh3 with different seeds to produce independent bit positions.
    fn bit_index(&self, hash: u64, k: u32) -> usize {
        let derived = xxh3_64(&[
            hash.to_le_bytes().as_slice(),
            &k.to_le_bytes(),
        ].concat());
        (derived as usize) % self.num_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(42);
        bf.insert(123);
        bf.insert(999);

        assert!(bf.may_contain(42));
        assert!(bf.may_contain(123));
        assert!(bf.may_contain(999));
    }

    #[test]
    fn test_false_positive_rate() {
        let n = 10_000;
        let mut bf = BloomFilter::new(n, 0.01);
        for i in 0..n as u64 {
            bf.insert(i);
        }

        // All inserted items must be found (no false negatives)
        for i in 0..n as u64 {
            assert!(bf.may_contain(i), "false negative for {}", i);
        }

        // Check false positive rate on non-inserted items
        let test_count = 100_000;
        let false_positives = (n as u64..n as u64 + test_count)
            .filter(|&i| bf.may_contain(i))
            .count();
        let fp_rate = false_positives as f64 / test_count as f64;

        // Allow up to 3% (we target 1%, but allow margin)
        assert!(
            fp_rate < 0.03,
            "False positive rate too high: {:.4}",
            fp_rate
        );
    }

    #[test]
    fn test_empty_filter() {
        let bf = BloomFilter::new(100, 0.01);
        // Empty filter should (almost certainly) return false
        let hits = (0..1000u64).filter(|&i| bf.may_contain(i)).count();
        assert!(hits < 10, "Empty filter has too many false positives: {}", hits);
    }

    #[test]
    fn test_from_hashes() {
        let hashes: Vec<u64> = (0..50).collect();
        let bf = BloomFilter::from_hashes(hashes.iter().copied(), 50);
        for &h in &hashes {
            assert!(bf.may_contain(h));
        }
    }
}
