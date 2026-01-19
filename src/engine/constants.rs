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

/// Token fields that use position-aware encoding
pub const TOKEN_FIELDS: [&str; 8] = [
    "word", "lemma", "pos", "tag", "chunk", "entity", "norm", "raw"
];