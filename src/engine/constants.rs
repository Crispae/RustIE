//! Field name constants for consistency across the codebase

pub const FIELD_WORD: &str = "word";
pub const FIELD_LEMMA: &str = "lemma";
pub const FIELD_POS: &str = "pos";
pub const FIELD_ENTITY: &str = "entity";
pub const FIELD_SENTENCE_LENGTH: &str = "sentence_length";
pub const FIELD_DEPENDENCIES_BINARY: &str = "dependencies_binary";
pub const FIELD_DOC_ID: &str = "doc_id";
pub const FIELD_SENTENCE_ID: &str = "sentence_id";
pub const FIELD_INCOMING_EDGES: &str = "incoming_edges";
pub const FIELD_OUTGOING_EDGES: &str = "outgoing_edges";

/// Tokenizer names for position-aware indexing
pub const TOKENIZER_EDGE_POSITION: &str = "edge_position_tokenizer";
pub const TOKENIZER_TOKEN_POSITION: &str = "token_position_tokenizer";

/// Token fields that use position-aware encoding
pub const TOKEN_FIELDS: [&str; 8] = [
    "word", "lemma", "pos", "tag", "chunk", "entity", "norm", "raw"
];

/// Look up a field in the schema, returning a descriptive error on failure.
pub fn get_required_field(schema: &tantivy::schema::Schema, name: &str) -> anyhow::Result<tantivy::schema::Field> {
    schema.get_field(name).map_err(|_| anyhow::anyhow!("Field '{}' not found in schema", name))
}