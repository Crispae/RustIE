//! Debug and performance logging helpers.

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Helper function for debug logging (formats JSON properly)
#[inline]
pub(crate) fn write_debug_log(entry: &str) {
    // Debug logging disabled - no-op to avoid file I/O overhead
    let _ = entry;
    // if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(".cursor/debug.log") {
    //     use std::io::Write;
    //     let _ = writeln!(file, "{}", entry);
    // }
}

/// Performance instrumentation helper
#[inline]
pub(crate) fn perf_log(_session_id: &str, _run_id: &str, _hypothesis_id: &str, _location: &str, _message: &str, _data: serde_json::Value) {
    // Performance logging disabled to avoid file I/O overhead
}
