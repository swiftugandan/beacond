//! SQLite-backed fingerprint database for storage and lookup.

use crate::fingerprint::Fingerprint;
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Metadata for a stored track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackInfo {
    pub id: i64,
    pub title: String,
    pub artist: String,
    pub album: Option<String>,
    pub duration_secs: f32,
    pub file_path: Option<String>,
    pub fingerprint_count: usize,
    pub added_at: String,
}

/// A match result from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub track: TrackInfo,
    /// Number of matching fingerprint hashes.
    pub match_count: usize,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Best offset alignment (in frames).
    pub best_offset: i64,
    /// Estimated position in the track (seconds).
    pub estimated_position_secs: f32,
}

/// The fingerprint database.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("Failed to open database: {}", path.as_ref().display()))?;

        let db = Database { conn };
        db.initialize()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Database { conn };
        db.initialize()?;
        Ok(db)
    }

    /// Initialize database schema.
    fn initialize(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;

            CREATE TABLE IF NOT EXISTS tracks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                artist      TEXT NOT NULL DEFAULT 'Unknown',
                album       TEXT,
                duration    REAL NOT NULL DEFAULT 0.0,
                file_path   TEXT,
                fp_count    INTEGER NOT NULL DEFAULT 0,
                added_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS fingerprints (
                hash        INTEGER NOT NULL,
                track_id    INTEGER NOT NULL,
                offset      INTEGER NOT NULL,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_fp_hash ON fingerprints(hash);
            CREATE INDEX IF NOT EXISTS idx_fp_track ON fingerprints(track_id);

            CREATE TABLE IF NOT EXISTS zones (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL UNIQUE,
                track_id        INTEGER NOT NULL,
                frequency_mode  TEXT NOT NULL DEFAULT 'ultrasonic',
                signature       BLOB,
                added_at        TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_zone_name ON zones(name);
            ",
        )?;

        // Migrations: add columns to existing databases (silently ignored if already present).
        let _ = self.conn.execute_batch(
            "ALTER TABLE zones ADD COLUMN frequency_mode TEXT NOT NULL DEFAULT 'ultrasonic';",
        );
        let _ = self
            .conn
            .execute_batch("ALTER TABLE zones ADD COLUMN signature BLOB;");

        Ok(())
    }

    /// Insert a new track with its fingerprints.
    pub fn insert_track(
        &mut self,
        title: &str,
        artist: &str,
        album: Option<&str>,
        duration_secs: f32,
        file_path: Option<&str>,
        fingerprints: &[Fingerprint],
    ) -> Result<i64> {
        let tx = self.conn.transaction()?;

        tx.execute(
            "INSERT INTO tracks (title, artist, album, duration, file_path, fp_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                title,
                artist,
                album,
                duration_secs,
                file_path,
                fingerprints.len()
            ],
        )?;

        let track_id = tx.last_insert_rowid();

        {
            let mut stmt = tx
                .prepare("INSERT INTO fingerprints (hash, track_id, offset) VALUES (?1, ?2, ?3)")?;

            for fp in fingerprints {
                stmt.execute(params![fp.hash as i64, track_id, fp.offset])?;
            }
        }

        tx.commit()?;

        log::info!(
            "Inserted track '{}' by '{}' (id={}, {} fingerprints)",
            title,
            artist,
            track_id,
            fingerprints.len()
        );

        Ok(track_id)
    }

    /// Remove a track and its fingerprints.
    pub fn remove_track(&mut self, track_id: i64) -> Result<bool> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM fingerprints WHERE track_id = ?1",
            params![track_id],
        )?;
        let rows = tx.execute("DELETE FROM tracks WHERE id = ?1", params![track_id])?;
        tx.commit()?;
        Ok(rows > 0)
    }

    /// Look up a track by ID.
    pub fn get_track(&self, track_id: i64) -> Result<Option<TrackInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, artist, album, duration, file_path, fp_count, added_at
             FROM tracks WHERE id = ?1",
        )?;

        let result = stmt
            .query_row(params![track_id], track_from_row)
            .optional()?;

        Ok(result)
    }

    /// List all tracks.
    pub fn list_tracks(&self) -> Result<Vec<TrackInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, artist, album, duration, file_path, fp_count, added_at
             FROM tracks ORDER BY added_at DESC",
        )?;

        let tracks = stmt
            .query_map([], track_from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(tracks)
    }

    /// Search the database for matches to a set of query fingerprints.
    /// Uses the offset-histogram alignment technique from the Shazam paper.
    /// `sample_rate` is used to convert frame offsets to seconds.
    pub fn search(
        &self,
        query_fingerprints: &[Fingerprint],
        max_results: usize,
        sample_rate: u32,
    ) -> Result<Vec<MatchResult>> {
        if query_fingerprints.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Collect all hash matches grouped by track_id.
        // For each match, store (db_offset - query_offset) as the time alignment.
        let mut track_offsets: HashMap<i64, Vec<i64>> = HashMap::new();

        let mut stmt = self
            .conn
            .prepare("SELECT track_id, offset FROM fingerprints WHERE hash = ?1")?;

        for qfp in query_fingerprints {
            let matches = stmt.query_map(params![qfp.hash as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
            })?;

            for (track_id, db_offset) in matches.flatten() {
                let alignment = db_offset - qfp.offset as i64;
                track_offsets.entry(track_id).or_default().push(alignment);
            }
        }

        // Step 2: For each track, find the offset with the most hits (histogram peak).
        let mut results: Vec<MatchResult> = Vec::new();

        for (track_id, offsets) in &track_offsets {
            // Build histogram of offsets
            let mut histogram: HashMap<i64, usize> = HashMap::new();
            for &off in offsets {
                *histogram.entry(off).or_insert(0) += 1;
            }

            // Find the peak
            let (best_offset, best_count) = histogram
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .unwrap_or((0, 0));

            // Minimum threshold for a valid match
            if best_count < 5 {
                continue;
            }

            if let Ok(Some(track)) = self.get_track(*track_id) {
                // Confidence: ratio of aligned matches to total query fingerprints
                let confidence = (best_count as f64 / query_fingerprints.len() as f64).min(1.0);

                // Estimate position in track from offset
                // offset is in spectrogram frames
                let hop_size = 512; // default
                let estimated_position =
                    (best_offset as f32 * hop_size as f32) / sample_rate as f32;

                results.push(MatchResult {
                    track,
                    match_count: best_count,
                    confidence,
                    best_offset,
                    estimated_position_secs: estimated_position.max(0.0),
                });
            }
        }

        // Sort by match count descending
        results.sort_by(|a, b| b.match_count.cmp(&a.match_count));
        results.truncate(max_results);

        Ok(results)
    }

    /// Get database statistics.
    pub fn stats(&self) -> Result<DbStats> {
        let track_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM tracks", [], |r| r.get(0))?;
        let fingerprint_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM fingerprints", [], |r| r.get(0))?;

        let db_size = self
            .conn
            .query_row(
                "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0);

        Ok(DbStats {
            track_count: track_count as usize,
            fingerprint_count: fingerprint_count as usize,
            db_size_bytes: db_size as u64,
        })
    }

    // ── Zone operations ─────────────────────────────────────────────

    /// Insert a named zone linked to an existing track, with an optional signature vector.
    pub fn insert_zone(
        &self,
        name: &str,
        track_id: i64,
        frequency_mode: &str,
        signature: Option<&[u8]>,
    ) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO zones (name, track_id, frequency_mode, signature) VALUES (?1, ?2, ?3, ?4)",
                params![name, track_id, frequency_mode, signature],
            )
            .with_context(|| format!("Failed to insert zone '{}' (duplicate name?)", name))?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Remove a zone and its associated track + fingerprints.
    pub fn remove_zone(&mut self, name: &str) -> Result<bool> {
        // Look up the track_id first
        let track_id: Option<i64> = self
            .conn
            .query_row(
                "SELECT track_id FROM zones WHERE name = ?1",
                params![name],
                |row| row.get(0),
            )
            .optional()?;

        match track_id {
            Some(tid) => {
                let tx = self.conn.transaction()?;
                tx.execute("DELETE FROM zones WHERE name = ?1", params![name])?;
                tx.execute("DELETE FROM fingerprints WHERE track_id = ?1", params![tid])?;
                tx.execute("DELETE FROM tracks WHERE id = ?1", params![tid])?;
                tx.commit()?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// List all zones with their track metadata.
    pub fn list_zones(&self) -> Result<Vec<ZoneInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT z.id, z.name, z.track_id, t.fp_count, t.duration, z.frequency_mode, z.added_at
             FROM zones z JOIN tracks t ON z.track_id = t.id
             ORDER BY z.name",
        )?;

        let zones = stmt
            .query_map([], zone_from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(zones)
    }

    /// Look up a zone by its associated track ID.
    pub fn get_zone_by_track_id(&self, track_id: i64) -> Result<Option<ZoneInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT z.id, z.name, z.track_id, t.fp_count, t.duration, z.frequency_mode, z.added_at
             FROM zones z JOIN tracks t ON z.track_id = t.id
             WHERE z.track_id = ?1",
        )?;

        let result = stmt
            .query_row(params![track_id], zone_from_row)
            .optional()?;

        Ok(result)
    }

    /// Load all zone signatures for vector similarity matching.
    /// Returns (zone_name, signature_bytes) for zones that have a signature.
    pub fn get_zone_signatures(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, signature FROM zones WHERE signature IS NOT NULL")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Return the distinct frequency modes across all registered zones.
    pub fn get_distinct_zone_modes(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT frequency_mode FROM zones")?;
        let modes = stmt
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<Vec<String>, _>>()?;
        Ok(modes)
    }
}

/// Map a row from the tracks table to a TrackInfo.
fn track_from_row(row: &rusqlite::Row) -> rusqlite::Result<TrackInfo> {
    Ok(TrackInfo {
        id: row.get(0)?,
        title: row.get(1)?,
        artist: row.get(2)?,
        album: row.get(3)?,
        duration_secs: row.get(4)?,
        file_path: row.get(5)?,
        fingerprint_count: row.get::<_, i64>(6)? as usize,
        added_at: row.get(7)?,
    })
}

/// Map a row from the zones+tracks join to a ZoneInfo.
fn zone_from_row(row: &rusqlite::Row) -> rusqlite::Result<ZoneInfo> {
    Ok(ZoneInfo {
        id: row.get(0)?,
        name: row.get(1)?,
        track_id: row.get(2)?,
        fingerprint_count: row.get::<_, i64>(3)? as usize,
        duration_secs: row.get(4)?,
        frequency_mode: row.get(5)?,
        added_at: row.get(6)?,
    })
}

/// Metadata for a named zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneInfo {
    pub id: i64,
    pub name: String,
    pub track_id: i64,
    pub fingerprint_count: usize,
    pub duration_secs: f32,
    pub frequency_mode: String,
    pub added_at: String,
}

/// Database statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbStats {
    pub track_count: usize,
    pub fingerprint_count: usize,
    pub db_size_bytes: u64,
}

impl DbStats {
    pub fn db_size_human(&self) -> String {
        let bytes = self.db_size_bytes;
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::generate_composite;
    use crate::fingerprint::Fingerprinter;

    #[test]
    fn test_insert_and_search() {
        let mut db = Database::in_memory().unwrap();
        let fp = Fingerprinter::with_defaults();

        // Create a "song"
        let song = generate_composite(
            &[(440.0, 0.5), (880.0, 0.3), (1320.0, 0.2), (220.0, 0.4)],
            5.0,
            16000,
        );
        let fingerprints = fp.fingerprint(&song, 16000);

        db.insert_track(
            "Test Song",
            "Test Artist",
            Some("Test Album"),
            5.0,
            None,
            &fingerprints,
        )
        .unwrap();

        // Search with the same signal
        let query_fps = fp.fingerprint(&song, 16000);
        let results = db.search(&query_fps, 5, 16000).unwrap();

        assert!(!results.is_empty(), "Should find at least one match");
        assert_eq!(results[0].track.title, "Test Song");
        assert!(results[0].confidence > 0.5);
    }

    #[test]
    fn test_zone_crud() {
        let mut db = Database::in_memory().unwrap();
        let fp = Fingerprinter::with_defaults();

        let song = generate_composite(&[(440.0, 0.5), (880.0, 0.3)], 3.0, 16000);
        let fingerprints = fp.fingerprint(&song, 16000);

        let track_id = db
            .insert_track("kitchen-beacon", "beacon", None, 3.0, None, &fingerprints)
            .unwrap();

        // Insert zone
        let zone_id = db
            .insert_zone("kitchen", track_id, "audible", None)
            .unwrap();
        assert!(zone_id > 0);

        // List zones
        let zones = db.list_zones().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].name, "kitchen");
        assert_eq!(zones[0].track_id, track_id);

        // Look up by track_id
        let zone = db.get_zone_by_track_id(track_id).unwrap().unwrap();
        assert_eq!(zone.name, "kitchen");

        // Remove zone (cascades to track + fingerprints)
        assert!(db.remove_zone("kitchen").unwrap());
        assert!(db.list_zones().unwrap().is_empty());
        assert!(db.get_track(track_id).unwrap().is_none());

        // Remove non-existent zone
        assert!(!db.remove_zone("nonexistent").unwrap());
    }
}
