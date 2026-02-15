use rustie::{Document, ExtractorEngine};
use std::fs;
use std::path::Path;

/// Create an ExtractorEngine backed by a temporary directory, with sample documents indexed.
///
/// The caller must keep the returned `TempDir` alive for the engine to remain valid.
/// When the `TempDir` is dropped, the index directory is cleaned up.
pub fn test_engine() -> (ExtractorEngine, tempfile::TempDir) {
    let tmp_dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let schema_path = Path::new("configs/schema.yaml");

    let mut engine = ExtractorEngine::new(tmp_dir.path(), schema_path)
        .expect("failed to create ExtractorEngine");

    // Index all sample documents
    let sample_dir = Path::new("sample_documents");
    if sample_dir.exists() {
        for entry in fs::read_dir(sample_dir).expect("failed to read sample_documents") {
            let entry = entry.expect("bad dir entry");
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));

                // Try single document, then array
                if let Ok(doc) = serde_json::from_str::<Document>(&content) {
                    engine.add_document(&doc).unwrap();
                } else if let Ok(docs) = serde_json::from_str::<Vec<Document>>(&content) {
                    for doc in &docs {
                        engine.add_document(doc).unwrap();
                    }
                }
            }
        }
    }

    engine.commit().expect("failed to commit");
    (engine, tmp_dir)
}
