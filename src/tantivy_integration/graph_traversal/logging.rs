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
pub(crate) fn perf_log(session_id: &str, run_id: &str, hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let log_entry = serde_json::json!({
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": timestamp
    });
    // Use absolute path to ensure logs are written regardless of working directory
    let log_path = r"c:\Users\saurav\OneDrive - URV\Escritorio\PARC\tantivy\RustIE\ruste_push\.cursor\debug.log";
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(file, "{}", log_entry);
    }
}
