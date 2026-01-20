use serde::{Deserialize, Serialize};
use tantivy::DocAddress;
use tantivy::Score;
use crate::types::SpanWithCaptures;

/// Represents a complete sentence result with tokens, document info, and score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceResult {
    /// Document ID from the original document
    pub document_id: String,
    /// Sentence ID within the document
    pub sentence_id: String,
    /// Document score
    
    pub score: Score,
    /// Array of matches found in this sentence

    pub matches: Vec<SpanWithCaptures>,
    /// Dynamic field storage for all configured fields
    pub fields: std::collections::HashMap<String, Vec<String>>,
}

impl SentenceResult {
    pub fn new(
        document_id: String,
        sentence_id: String,
        score: Score,
        matches: Vec<SpanWithCaptures>,
        fields: std::collections::HashMap<String, Vec<String>>,
    ) -> Self {
        Self {
            document_id,
            sentence_id,
            score,
            matches,
            fields,
        }
    }

    /// Get the sentence as a string
    pub fn sentence_text(&self) -> String {
        if let Some(tokens) = self.fields.get("word") {
            tokens.join(" ")
        } else {
            String::new()
        }
    }

    /// Get match text for a specific span
    pub fn get_match_text(&self, span: &crate::types::Span) -> Option<String> {
        if let Some(tokens) = self.fields.get("word") {
            if span.start < tokens.len() && span.end <= tokens.len() && span.start < span.end {
                Some(tokens[span.start..span.end].join(" "))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get a specific field value
    pub fn get_field(&self, field_name: &str) -> Option<&Vec<String>> {
        self.fields.get(field_name)
    }

    /// Get all field names
    pub fn get_field_names(&self) -> Vec<&String> {
        self.fields.keys().collect()
    }

    /// Get tokens (convenience method)
    pub fn tokens(&self) -> Option<&Vec<String>> {
        self.fields.get("word")
    }

    /// Get lemmas (convenience method)
    pub fn lemmas(&self) -> Option<&Vec<String>> {
        self.fields.get("lemma")
    }

    /// Get POS tags (convenience method)
    pub fn pos_tags(&self) -> Option<&Vec<String>> {
        self.fields.get("pos")
    }

    /// Get entity tags (convenience method)
    pub fn entity_tags(&self) -> Option<&Vec<String>> {
        self.fields.get("entity")
    }

    /// Get dependencies (convenience method)
    pub fn dependencies(&self) -> Option<&Vec<String>> {
        self.fields.get("dependencies")
    }

    /// Format this sentence result as a pretty JSON string
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| format!("{:?}", self))
    }

    /// Get sentence result as JSON with custom indentation
    pub fn to_json_with_indent(&self, indent: usize) -> String {
        let indent_str = " ".repeat(indent);
        let mut buffer = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(indent_str.as_bytes());
        let mut serializer = serde_json::Serializer::with_formatter(
            std::io::Cursor::new(&mut buffer),
            formatter
        );
        
        if let Ok(()) = self.serialize(&mut serializer) {
            String::from_utf8(buffer).unwrap_or_else(|_| format!("{:?}", self))
        } else {
            format!("{:?}", self)
        }
    }

    /// Get sentence result as JSON with 2-space indentation (standard)
    pub fn to_json_standard(&self) -> String {
        self.to_json_with_indent(2)
    }

    /// Get sentence result as JSON with 4-space indentation
    pub fn to_json_indented(&self) -> String {
        self.to_json_with_indent(4)
    }

    /// Get a formatted string representation
    pub fn to_formatted_string(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Document: {}, Sentence: {}\n", self.document_id, self.sentence_id));
        output.push_str(&format!("Score: {:.6}\n", self.score));
        output.push_str(&format!("Text: {}\n", self.sentence_text()));
        
        // Add field information
        for (field_name, values) in &self.fields {
            if field_name != "word" { // Skip word field as it's already shown as text
                output.push_str(&format!("{}: [{}]\n", field_name, values.join(", ")));
            }
        }
        
        output
    }
}

/// Represents a scored document in search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustieDoc {
    pub doc: DocAddress,
    pub score: Score,
    pub shard_index: Option<i32>,
    /// Matches found in this document
    pub matches: Vec<SpanWithCaptures>,
}

impl RustieDoc {
    pub fn new(doc: DocAddress, score: Score) -> Self {
        Self {
            doc,
            score,
            shard_index: None,
            matches: Vec::new(),
        }
    }

    pub fn with_shard_index(doc: DocAddress, score: Score, shard_index: i32) -> Self {
        Self {
            doc,
            score,
            shard_index: Some(shard_index),
            matches: Vec::new(),
        }
    }

    pub fn with_matches(doc: DocAddress, score: Score, matches: Vec<SpanWithCaptures>) -> Self {
        Self {
            doc,
            score,
            shard_index: None,
            matches,
        }
    }

    /// Add a match to this document
    pub fn add_match(&mut self, match_span: SpanWithCaptures) {
        self.matches.push(match_span);
    }

    /// Get all matches for this document
    pub fn get_matches(&self) -> &[SpanWithCaptures] {
        &self.matches
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustIeResult {
    pub total_hits: usize,
    #[serde(skip_serializing, skip_deserializing)]
    /// Internal: scored docs for engine logic, not exposed to user
    pub score_docs: Vec<RustieDoc>,
    pub sentence_results: Vec<SentenceResult>,
    pub max_score: Option<Score>,
}

impl RustIeResult {
    /// Create empty results
    pub fn empty() -> Self {
        Self {
            total_hits: 0,
            score_docs: Vec::new(),
            sentence_results: Vec::new(),
            max_score: None,
        }
    }

    /// Create results with the given data
    pub fn new(total_hits: usize, score_docs: Vec<RustieDoc>, max_score: Option<Score>) -> Self {
        Self {
            total_hits,
            score_docs,
            sentence_results: Vec::new(),
            max_score,
        }
    }

    /// Create results with sentence information
    pub fn with_sentences(
        total_hits: usize, 
        score_docs: Vec<RustieDoc>, 
        sentence_results: Vec<SentenceResult>,
        max_score: Option<Score>
    ) -> Self {
        Self {
            total_hits,
            score_docs,
            sentence_results,
            max_score,
        }
    }

    /// Get the number of documents returned
    pub fn len(&self) -> usize {
        self.score_docs.len()
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.score_docs.is_empty()
    }

    /// Get a reference to the score docs
    pub fn score_docs(&self) -> &[RustieDoc] {
        &self.score_docs
    }

    /// Get a reference to the sentence results
    pub fn sentence_results(&self) -> &[SentenceResult] {
        &self.sentence_results
    }

    /// Get a mutable reference to the score docs
    pub fn score_docs_mut(&mut self) -> &mut Vec<RustieDoc> {
        &mut self.score_docs
    }

    /// Get a mutable reference to the sentence results
    pub fn sentence_results_mut(&mut self) -> &mut Vec<SentenceResult> {
        &mut self.sentence_results
    }

    /// Add a score doc to the results
    pub fn add_score_doc(&mut self, score_doc: RustieDoc) {
        self.score_docs.push(score_doc);
    }

    /// Add a sentence result to the results
    pub fn add_sentence_result(&mut self, sentence_result: SentenceResult) {
        self.sentence_results.push(sentence_result);
    }

    /// Sort results by score (descending)
    pub fn sort_by_score(&mut self) {
        self.score_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        self.sentence_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Take only the top N results
    pub fn take_top(&mut self, n: usize) {
        if self.score_docs.len() > n {
            self.score_docs.truncate(n);
        }
        if self.sentence_results.len() > n {
            self.sentence_results.truncate(n);
        }
    }

    /// Merge multiple result sets
    pub fn merge(results: Vec<RustIeResult>, top_n: usize) -> Self {
        let mut all_score_docs = Vec::new();
        let mut all_sentence_results = Vec::new();
        let mut total_hits = 0;
        let mut max_score = None;

        for result in results {
            total_hits += result.total_hits;
            all_score_docs.extend(result.score_docs);
            all_sentence_results.extend(result.sentence_results);
            
            if let Some(score) = result.max_score {
                max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
            }
        }

        // Sort by score and take top N
        all_score_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_sentence_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_score_docs.truncate(top_n);
        all_sentence_results.truncate(top_n);

        Self {
            total_hits,
            score_docs: all_score_docs,
            sentence_results: all_sentence_results,
            max_score,
        }
    }

    /// Format results as pretty JSON string with proper indentation
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| format!("{:?}", self))
    }

    /// Format results as compact JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| format!("{:?}", self))
    }

    /// Get results as JSON with custom indentation
    pub fn to_json_with_indent(&self, indent: usize) -> String {
        let indent_str = " ".repeat(indent);
        let mut buffer = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(indent_str.as_bytes());
        let mut serializer = serde_json::Serializer::with_formatter(
            std::io::Cursor::new(&mut buffer),
            formatter
        );
        
        if let Ok(()) = self.serialize(&mut serializer) {
            String::from_utf8(buffer).unwrap_or_else(|_| format!("{:?}", self))
        } else {
            format!("{:?}", self)
        }
    }

    /// Get results as JSON with 2-space indentation (standard)
    pub fn to_json_standard(&self) -> String {
        self.to_json_with_indent(2)
    }

    /// Get results as JSON with 4-space indentation
    pub fn to_json_indented(&self) -> String {
        self.to_json_with_indent(4)
    }

    /// Get a formatted summary of the results
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Total hits: {}\n", self.total_hits));
        summary.push_str(&format!("Max score: {:?}\n", self.max_score));
        summary.push_str(&format!("Found {} sentence(s):\n", self.sentence_results.len()));
        
        for (i, sentence) in self.sentence_results.iter().enumerate() {
            summary.push_str(&format!("  Sentence {}:\n", i + 1));
            summary.push_str(&format!("    Document ID: {}\n", sentence.document_id));
            summary.push_str(&format!("    Sentence ID: {}\n", sentence.sentence_id));
            summary.push_str(&format!("    Score: {:.6}\n", sentence.score));
            
            // Add sentence text
            if let Some(words) = sentence.fields.get("word") {
                summary.push_str(&format!("    Text: [{}]\n", words.join(", ")));
            }
            
            // Add other important fields
            if let Some(lemmas) = sentence.fields.get("lemma") {
                summary.push_str(&format!("    Lemmas: [{}]\n", lemmas.join(", ")));
            }
            if let Some(pos_tags) = sentence.fields.get("pos") {
                summary.push_str(&format!("    POS: [{}]\n", pos_tags.join(", ")));
            }
            if let Some(entities) = sentence.fields.get("entity") {
                summary.push_str(&format!("    Entities: [{}]\n", entities.join(", ")));
            }
        }
        
        summary
    }

    /// Get a minimal formatted output for quick viewing
    pub fn to_minimal_string(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Results: {} hits, max_score: {:?}\n", self.total_hits, self.max_score));
        
        for (i, sentence) in self.sentence_results.iter().enumerate() {
            output.push_str(&format!("  {}. {} (score: {:.3})\n", 
                i + 1, 
                sentence.sentence_text(), 
                sentence.score));
        }
        
        output
    }
}

impl Default for RustIeResult {
    fn default() -> Self {
        Self::empty()
    }
} 