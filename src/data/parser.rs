use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use anyhow::{Result, anyhow};
use flate2::read::GzDecoder;
use serde_json;
use tantivy::schema::{TantivyDocument};
use tantivy::schema::Schema;

use crate::data::document::{Document as RustDoc, Field as DocField};
use crate::digraph::graph::DirectedGraph;

/// Required fields that must exist in the schema for core functionality
const REQUIRED_FIELDS: &[&str] = &["doc_id", "sentence_id", "sentence_length", "word"];

/// Parser for JSON and gzipped JSON documents
pub struct DocumentParser {
    schema: Schema,
}

impl DocumentParser {
    /// Create a new DocumentParser with the given schema
    pub fn new(schema: Schema) -> Self {
        Self { schema }
    }

    /// Validate that all required fields exist in the schema
    /// Call this before processing documents to catch configuration errors early
    pub fn validate_schema(&self) -> Result<()> {
        let mut missing_fields = Vec::new();
        for &field_name in REQUIRED_FIELDS {
            if self.schema.get_field(field_name).is_err() {
                missing_fields.push(field_name);
            }
        }
        if !missing_fields.is_empty() {
            return Err(anyhow!(
                "Schema validation failed: missing required fields: {:?}",
                missing_fields
            ));
        }
        Ok(())
    }

    /// Check if a field exists in the schema
    pub fn has_field(&self, field_name: &str) -> bool {
        self.schema.get_field(field_name).is_ok()
    }

    /// Get a field from schema with a descriptive error
    fn get_field_checked(&self, field_name: &str) -> Result<tantivy::schema::Field> {
        self.schema.get_field(field_name).map_err(|_| {
            anyhow!("Field '{}' not found in schema", field_name)
        })
    }

    /// Validate document structure before indexing
    /// Checks for valid edge indices and consistent token counts
    pub fn validate_document(&self, doc: &RustDoc) -> Result<()> {
        for (sentence_idx, sentence) in doc.sentences.iter().enumerate() {
            let token_count = sentence.numTokens as usize;

            for field in &sentence.fields {
                match field {
                    DocField::TokensField { name, tokens, .. } => {
                        // Warn if token count doesn't match declared numTokens
                        if tokens.len() != token_count {
                            log::warn!(
                                "Document '{}' sentence {}: field '{}' has {} tokens but numTokens is {}",
                                doc.id, sentence_idx, name, tokens.len(), token_count
                            );
                        }
                    }
                    DocField::GraphField { name, edges, .. } => {
                        // Validate edge indices are within bounds
                        for (from, to, rel) in edges {
                            let from_idx = *from as usize;
                            let to_idx = *to as usize;
                            if from_idx >= token_count {
                                return Err(anyhow!(
                                    "Document '{}' sentence {}: edge {}->{}:{} has invalid 'from' index {} (token count: {})",
                                    doc.id, sentence_idx, from, to, rel, from_idx, token_count
                                ));
                            }
                            if to_idx >= token_count {
                                return Err(anyhow!(
                                    "Document '{}' sentence {}: edge {}->{}:{} has invalid 'to' index {} (token count: {})",
                                    doc.id, sentence_idx, from, to, rel, to_idx, token_count
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse a JSON file (regular or gzipped)
    pub fn parse_file<P: AsRef<Path>>(&self, file_path: P) -> Result<Vec<RustDoc>> {
        let path = file_path.as_ref();
        let file = File::open(path)?;
        
        // Check if file is gzipped by looking at magic bytes
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 2];
        reader.read_exact(&mut magic)?;
        
        let documents = if magic == [0x1f, 0x8b] {
            // Gzipped file
            let file = File::open(path)?;
            let gz = GzDecoder::new(file);
            // Docding to normal JSON files
            let reader = BufReader::new(gz);
            self.parse_reader(reader)?
        } else {
            // Regular JSON file
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            self.parse_reader(reader)?
        };

        Ok(documents)
    }

    /// Parse from a reader (handles both single document and array of documents)
    pub fn parse_reader<R: Read>(&self, mut reader: BufReader<R>) -> Result<Vec<RustDoc>> {
        let mut documents = Vec::new();
        
        // Read the entire content first
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        
        // Try to parse as a single document first
        if let Ok(doc) = serde_json::from_str::<RustDoc>(&content) {
            documents.push(doc);
            return Ok(documents);
        }

        // Try to parse as an array of documents
        if let Ok(docs) = serde_json::from_str::<Vec<RustDoc>>(&content) {
            return Ok(docs);
        }

        // Try line-by-line JSON (JSONL format)
        for line in content.lines() {
            if !line.trim().is_empty() {
                let doc: RustDoc = serde_json::from_str(line)?;
                documents.push(doc);
            }
        }

        Ok(documents)
    }

    /// Convert a Document to a Tantivy document for indexing
    pub fn to_tantivy_document(&mut self, doc: &RustDoc) -> Result<Vec<TantivyDocument>> {
        // Creating a mutable vectore to store the tantivy documents
        let mut tantivy_docs = Vec::new();
        
        for (sentence_idx, sentence) in doc.sentences.iter().enumerate() {
            let mut tantivy_doc = TantivyDocument::default();
            
            // Add document-level fields
            // TODO: These fields are not required for indexing, but they are required for searching
            tantivy_doc.add_text(self.schema.get_field("doc_id")?, &doc.id);
            tantivy_doc.add_text(self.schema.get_field("sentence_id")?, &format!("{}_{}", doc.id, sentence_idx));
            tantivy_doc.add_u64(self.schema.get_field("sentence_length")?, sentence.numTokens as u64);
            
            // Process each field in the sentence
            for field in &sentence.fields {
                match field {
                    DocField::TokensField { name, tokens, .. } => {
                        // Use position-aware encoding for token fields
                        // This ensures each token gets the correct position (0, 1, 2, ...)
                        // instead of all tokens being at position 0
                        let encoded = crate::tantivy_integration::position_tokenizer::encode_position_aware_tokens(tokens);

                        match name.as_str() {
                            "raw" | "word" | "lemma" | "pos" | "tag" | "chunk" | "entity" | "norm" => {
                                if let Ok(field) = self.schema.get_field(name) {
                                    tantivy_doc.add_text(field, &encoded);
                                }
                            }
                            _ => {
                                // For unknown fields, also use position-aware encoding
                                if let Ok(field) = self.schema.get_field(name) {
                                    tantivy_doc.add_text(field, &encoded);
                                }
                            }
                        }
                    }
                    DocField::GraphField { name, edges, .. } => {
                        match name.as_str() {
                            "dependencies" => {
                                // Create and serialize DirectedGraph for binary storage
                                let mut graph = DirectedGraph::new();
                                
                                // Add nodes (tokens)
                                if let Some(tokens) = doc.get_tokens(sentence_idx, "word") {
                                    for (i, _token) in tokens.iter().enumerate() {
                                        graph.add_node(i);
                                    }
                                }
                                
                                // Add edges
                                for (from, to, rel) in edges {
                                    graph.add_edge(*from as usize, *to as usize, &rel);
                                }
                                
                                // Serialize to binary and store
                                if let Ok(binary_field) = self.schema.get_field("dependencies_binary") {
                                    match graph.to_bytes() {
                                        Ok(bytes) => {
                                            tantivy_doc.add_bytes(binary_field, &bytes);
                                        }
                                        Err(e) => {
                                    log::warn!("Graph serialization failed: {}", e);
                                        }
                                    }
                                }
                                
                                // Populate incoming and outgoing edge fields for filtering
                                // ODINSON-STYLE: Position-aware edge indexing
                                // Each edge label is indexed at its token position
                                let mut outgoing_edges: Vec<Vec<String>> = Vec::new(); // outgoing[i] = list of labels at position i
                                let mut incoming_edges: Vec<Vec<String>> = Vec::new(); // incoming[i] = list of labels at position i

                                // Initialize vectors based on sentence length
                                if let Some(tokens) = doc.get_tokens(sentence_idx, "word") {
                                    outgoing_edges.resize(tokens.len(), Vec::new());
                                    incoming_edges.resize(tokens.len(), Vec::new());
                                }

                                // Populate edges at their token positions
                                for (from, to, rel) in edges {
                                    if (*from as usize) < outgoing_edges.len() {
                                        outgoing_edges[*from as usize].push(rel.clone());
                                    }
                                    if (*to as usize) < incoming_edges.len() {
                                        incoming_edges[*to as usize].push(rel.clone());
                                    }
                                }

                                // Encode as position-aware format and add to document
                                // Format: "|label1,label2||label3|" where | separates positions
                                if let Ok(field) = self.schema.get_field("outgoing_edges") {
                                    let encoded = crate::tantivy_integration::position_tokenizer::encode_position_aware_edges(&outgoing_edges);
                                    tantivy_doc.add_text(field, &encoded);
                                }

                                if let Ok(field) = self.schema.get_field("incoming_edges") {
                                    let encoded = crate::tantivy_integration::position_tokenizer::encode_position_aware_edges(&incoming_edges);
                                    tantivy_doc.add_text(field, &encoded);
                                }
                            }
                            _ => {
                                // Try to add as a generic text field if it exists in schema
                                let edges_str = edges.iter()
                                    .map(|(from, to, rel)| format!("{}:{}:{}", from, to, rel))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                if let Ok(field) = self.schema.get_field(name) {
                                    tantivy_doc.add_text(field, &edges_str);
                                }
                            }
                        }
                    }
                }
            }
            
            tantivy_docs.push(tantivy_doc);
        }
        
        Ok(tantivy_docs)
    }

    /// Parse a JSON string
    pub fn parse_json(&self, json_str: &str) -> Result<Vec<RustDoc>> {
        // Try to parse as a single document
        if let Ok(doc) = serde_json::from_str::<RustDoc>(json_str) {
            return Ok(vec![doc]);
        }

        // Try to parse as an array of documents
        if let Ok(docs) = serde_json::from_str::<Vec<RustDoc>>(json_str) {
            return Ok(docs);
        }

        Err(anyhow!("Failed to parse JSON as single document or array of documents"))
    }

    /// Parse a gzipped JSON string
    pub fn parse_gzipped_json(&self, gzipped_data: &[u8]) -> Result<Vec<RustDoc>> {
        let mut decoder = GzDecoder::new(gzipped_data);
        let mut json_str = String::new();
        decoder.read_to_string(&mut json_str)?;
        self.parse_json(&json_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::document::Sentence;
    use tantivy::schema::SchemaBuilder;

    /// Create a test schema with all required fields
    fn create_complete_schema() -> Schema {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("doc_id", tantivy::schema::TEXT | tantivy::schema::STORED);
        builder.add_text_field("sentence_id", tantivy::schema::TEXT | tantivy::schema::STORED);
        builder.add_u64_field("sentence_length", tantivy::schema::STORED);
        builder.add_text_field("word", tantivy::schema::TEXT | tantivy::schema::STORED);
        builder.add_text_field("lemma", tantivy::schema::TEXT | tantivy::schema::STORED);
        builder.add_bytes_field("dependencies_binary", tantivy::schema::STORED);
        builder.build()
    }

    /// Create a schema missing some required fields
    fn create_incomplete_schema() -> Schema {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("doc_id", tantivy::schema::TEXT | tantivy::schema::STORED);
        // Missing: sentence_id, sentence_length, word
        builder.build()
    }

    /// Create a test document with valid structure
    fn create_valid_document() -> RustDoc {
        RustDoc {
            id: "test_doc".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 3,
                    fields: vec![
                        DocField::TokensField {
                            name: "word".to_string(),
                            tokens: vec!["John".to_string(), "eats".to_string(), "pizza".to_string()],
                        },
                        DocField::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),  // eats -> John
                                (1, 2, "dobj".to_string()),   // eats -> pizza
                            ],
                        },
                    ],
                },
            ],
        }
    }

    /// Create a document with invalid edge indices
    fn create_document_with_invalid_edges() -> RustDoc {
        RustDoc {
            id: "invalid_doc".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 3,
                    fields: vec![
                        DocField::TokensField {
                            name: "word".to_string(),
                            tokens: vec!["John".to_string(), "eats".to_string(), "pizza".to_string()],
                        },
                        DocField::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![1],
                            edges: vec![
                                (1, 0, "nsubj".to_string()),
                                (10, 2, "dobj".to_string()),  // Invalid: from_idx 10 >= token_count 3
                            ],
                        },
                    ],
                },
            ],
        }
    }

    // ==================== Schema Validation Tests ====================

    #[test]
    fn test_validate_schema_success() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        assert!(parser.validate_schema().is_ok());
    }

    #[test]
    fn test_validate_schema_missing_fields() {
        let schema = create_incomplete_schema();
        let parser = DocumentParser::new(schema);
        let result = parser.validate_schema();
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("missing required fields"));
        assert!(error_msg.contains("sentence_id"));
        assert!(error_msg.contains("sentence_length"));
        assert!(error_msg.contains("word"));
    }

    #[test]
    fn test_has_field_exists() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        assert!(parser.has_field("doc_id"));
        assert!(parser.has_field("word"));
        assert!(parser.has_field("sentence_length"));
    }

    #[test]
    fn test_has_field_not_exists() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        assert!(!parser.has_field("nonexistent_field"));
        assert!(!parser.has_field("fake_field"));
    }

    // ==================== Document Validation Tests ====================

    #[test]
    fn test_validate_document_valid() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let doc = create_valid_document();
        assert!(parser.validate_document(&doc).is_ok());
    }

    #[test]
    fn test_validate_document_invalid_from_index() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let doc = create_document_with_invalid_edges();
        let result = parser.validate_document(&doc);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("invalid 'from' index"));
        assert!(error_msg.contains("10"));
        assert!(error_msg.contains("token count: 3"));
    }

    #[test]
    fn test_validate_document_invalid_to_index() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let doc = RustDoc {
            id: "invalid_to".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 3,
                    fields: vec![
                        DocField::TokensField {
                            name: "word".to_string(),
                            tokens: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                        },
                        DocField::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![0],
                            edges: vec![
                                (0, 99, "edge".to_string()),  // Invalid: to_idx 99 >= token_count 3
                            ],
                        },
                    ],
                },
            ],
        };
        let result = parser.validate_document(&doc);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("invalid 'to' index"));
        assert!(error_msg.contains("99"));
    }

    #[test]
    fn test_validate_document_empty_sentences() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let doc = RustDoc {
            id: "empty_doc".to_string(),
            metadata: vec![],
            sentences: vec![],  // No sentences
        };
        assert!(parser.validate_document(&doc).is_ok());
    }

    #[test]
    fn test_validate_document_multiple_sentences() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let doc = RustDoc {
            id: "multi_sentence".to_string(),
            metadata: vec![],
            sentences: vec![
                Sentence {
                    numTokens: 2,
                    fields: vec![
                        DocField::TokensField {
                            name: "word".to_string(),
                            tokens: vec!["Hello".to_string(), "world".to_string()],
                        },
                        DocField::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![0],
                            edges: vec![(0, 1, "det".to_string())],
                        },
                    ],
                },
                Sentence {
                    numTokens: 3,
                    fields: vec![
                        DocField::TokensField {
                            name: "word".to_string(),
                            tokens: vec!["Good".to_string(), "bye".to_string(), "now".to_string()],
                        },
                        DocField::GraphField {
                            name: "dependencies".to_string(),
                            roots: vec![0],
                            edges: vec![
                                (0, 1, "amod".to_string()),
                                (0, 2, "advmod".to_string()),
                            ],
                        },
                    ],
                },
            ],
        };
        assert!(parser.validate_document(&doc).is_ok());
    }

    // ==================== JSON Parsing Tests ====================

    #[test]
    fn test_parse_json_single_document() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let json = r#"{"id": "doc1", "metadata": [], "sentences": []}"#;
        let result = parser.parse_json(json);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "doc1");
    }

    #[test]
    fn test_parse_json_array_of_documents() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let json = r#"[{"id": "doc1", "metadata": [], "sentences": []}, {"id": "doc2", "metadata": [], "sentences": []}]"#;
        let result = parser.parse_json(json);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id, "doc1");
        assert_eq!(docs[1].id, "doc2");
    }

    #[test]
    fn test_parse_json_invalid() {
        let schema = create_complete_schema();
        let parser = DocumentParser::new(schema);
        let json = "not valid json";
        let result = parser.parse_json(json);
        assert!(result.is_err());
    }
} 