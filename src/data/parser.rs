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

/// Parser for JSON and gzipped JSON documents
pub struct DocumentParser {
    schema: Schema,
}

impl DocumentParser {
    /// Create a new DocumentParser with the given schema
    pub fn new(schema: Schema) -> Self {
        Self { schema }
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
                        match name.as_str() {
                            
                            "raw" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("raw")?, token);
                                }
                            }
                            "word" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("word")?, token);
                                }
                            }
                            "lemma" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("lemma")?, token);
                                }
                            }
                            "pos" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("pos")?, token);
                                }
                            }
                            "tag" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("tag")?, token);
                                }
                            }
                            "chunk" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("chunk")?, token);
                                }
                            }
                            "entity" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("entity")?, token);
                                }
                            }
                            "norm" => {
                                for token in tokens {
                                    tantivy_doc.add_text(self.schema.get_field("norm")?, token);
                                }
                            }
                            _ => {
                                if let Ok(field) = self.schema.get_field(name) {
                                    tantivy_doc.add_text(field, &tokens.join(" "));
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
                                        Err(_) => {
                                            // Graph serialization failed, continue without it
                                        }
                                    }
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