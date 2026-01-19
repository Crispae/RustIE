//! Document management methods for ExtractorEngine

use crate::data::document::Document;
use crate::data::parser::DocumentParser;
use crate::engine::constants::*;
use crate::engine::core::ExtractorEngine;
use crate::results::rustie_results::SentenceResult;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tantivy::{
    schema::{TantivyDocument, Value},
    DocAddress, Score,
};

impl ExtractorEngine {
    /// Get a document by its address
    pub fn doc(&self, doc_address: DocAddress) -> Result<TantivyDocument> {
        self.reader
            .searcher()
            .doc(doc_address)
            .map_err(anyhow::Error::from)
    }

    /// Extract sentence result from a Tantivy document
    pub fn extract_sentence_result(
        &self,
        doc: &TantivyDocument,
        score: Score,
    ) -> Result<SentenceResult> {
        let document_id = self
            .extract_field_value(doc, FIELD_DOC_ID)
            .unwrap_or_else(|| "unknown".to_string());

        let sentence_id = self
            .extract_field_value(doc, FIELD_SENTENCE_ID)
            .unwrap_or_else(|| "0".to_string());

        let mut field_values = HashMap::new();
        for field_name in &self.output_fields {
            let values = self.extract_field_values(doc, field_name);
            field_values.insert(field_name.clone(), values);
        }

        Ok(SentenceResult::new(
            document_id,
            sentence_id,
            score,
            Vec::new(), // Matches will be populated separately if needed
            field_values,
        ))
    }

    /// Extract a single field value from a Tantivy document
    pub fn extract_field_value(&self, doc: &TantivyDocument, field_name: &str) -> Option<String> {
        if let Ok(field) = self.schema.get_field(field_name) {
            doc.get_first(field).and_then(|value| {
                if let Some(text) = value.as_str() {
                    Some(text.to_string())
                } else if let Some(u64_val) = value.as_u64() {
                    Some(u64_val.to_string())
                } else {
                    None
                }
            })
        } else {
            None
        }
    }

    /// Extract multiple field values from a Tantivy document
    /// For token fields (word, lemma, pos, etc.), decodes the position-aware format
    pub fn extract_field_values(&self, doc: &TantivyDocument, field_name: &str) -> Vec<String> {
        if let Ok(field) = self.schema.get_field(field_name) {
            let raw_values: Vec<String> = doc
                .get_all(field)
                .filter_map(|value| {
                    if let Some(text) = value.as_str() {
                        Some(text.to_string())
                    } else if let Some(u64_val) = value.as_u64() {
                        Some(u64_val.to_string())
                    } else {
                        None
                    }
                })
                .collect();

            // For token fields stored in position-aware format (e.g., "John|eats|pizza"),
            // decode by splitting on | to get individual tokens
            if TOKEN_FIELDS.contains(&field_name) && raw_values.len() == 1 {
                raw_values[0].split('|').map(|s| s.to_string()).collect()
            } else {
                raw_values
            }
        } else {
            Vec::new()
        }
    }

    /// Add a document to the index
    pub fn add_document(&mut self, document: &Document) -> Result<()> {
        let mut parser = DocumentParser::new(self.schema.clone());
        let tantivy_docs = parser.to_tantivy_document(document)?;

        if let Some(writer) = &mut self.writer {
            for tantivy_doc in tantivy_docs {
                writer.add_document(tantivy_doc)?;
            }
        } else {
            return Err(anyhow!(
                "Cannot add document: Engine is in READ-ONLY mode (index lock could not be acquired)"
            ));
        }

        Ok(())
    }

    /// Add multiple documents to the index
    pub fn add_documents(&mut self, documents: &[Document]) -> Result<()> {
        for document in documents {
            self.add_document(document)?;
        }
        Ok(())
    }

    /// Commit changes to the index
    pub fn commit(&mut self) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.commit()?;
        } else {
            return Err(anyhow!("Cannot commit: Engine is in READ-ONLY mode"));
        }
        // Refresh the reader to see the new documents
        self.reader = self.index.reader()?;
        Ok(())
    }

    /// Get the index writer (for advanced usage)
    pub fn writer(&mut self) -> Result<&mut tantivy::IndexWriter> {
        self.writer
            .as_mut()
            .ok_or_else(|| anyhow!("Engine is in READ-ONLY mode"))
    }

    /// Get output field names
    pub fn get_output_field_names(&self) -> Vec<&String> {
        self.output_fields.iter().collect()
    }
}
