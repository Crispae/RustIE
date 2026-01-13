use tantivy::schema::{Schema, TantivyDocument, Value};

/// Extract multiple field values from a Tantivy document
/// For token fields (word, lemma, pos, etc.), decodes the position-aware format
pub fn extract_field_values(schema: &Schema, doc: &TantivyDocument, field_name: &str) -> Vec<String> {
    if let Ok(field) = schema.get_field(field_name) {
        let raw_values: Vec<String> = doc.get_all(field).filter_map(|value| {
            if let Some(text) = value.as_str() {
                Some(text.to_string())
            } else if let Some(u64_val) = value.as_u64() {
                Some(u64_val.to_string())
            } else {
                None
            }
        }).collect();

        // For token fields stored in position-aware format (e.g., "John|eats|pizza"),
        // decode by splitting on | to get individual tokens
        // These fields use the position-aware encoding
        let token_fields = ["word", "lemma", "pos", "tag", "chunk", "entity", "norm", "raw"];
        if token_fields.contains(&field_name) && raw_values.len() == 1 {
            // Single encoded string - decode it
            raw_values[0].split('|').map(|s| s.to_string()).collect()
        } else {
            raw_values
        }
    } else {
        Vec::new()
    }
}
