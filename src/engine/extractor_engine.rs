use crate::results::rustie_results::{RustIeResult, RustieDoc, SentenceResult};
use crate::compiler::QueryCompiler;
use crate::data::document::Document;
use crate::data::parser::DocumentParser;
use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
// Import from tantivy
use tantivy::{
    doc, Index, IndexReader, IndexWriter,
    query::Query, schema::{Field, Schema, TantivyDocument, Value},
    DocAddress, Score, collector::TopDocs, directory::MmapDirectory
};
use crate::tantivy_integration::concat_query::RustieConcatQuery;
use crate::tantivy_integration::named_capture_query::RustieNamedCaptureQuery;
use rayon::prelude::*;

// Field name constants for consistency across the codebase
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

#[derive(Debug, Deserialize)]
pub struct SchemaConfig {
    pub output_fields: Option<Vec<String>>,
    pub fields: Vec<FieldConfig>,
}

#[derive(Debug, Deserialize)]
pub struct FieldConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub stored: bool,
}

/// Main engine for information extraction using Tantivy
pub struct ExtractorEngine {
    index: Index,
    reader: IndexReader,
    writer: Option<IndexWriter>,
    schema: Schema,
    default_field: Field, // Can be directly used from schema
    sentence_length_field: Field, // Can be directly used from schema
    dependencies_binary_field: Field, // Can be directly used from schema
    incoming_edges_field: Field,
    outgoing_edges_field: Field,
    parent_doc_id_field: String, // Can be directly used from schema
    output_fields: Vec<String>, // Can be directly used from schema
}


/// Engine for information extraction using Tantivy
impl ExtractorEngine {
    /// Create a new ExtractorEngine from an index directory with a required schema file
    pub fn new(index_dir: &Path, schema_path: &Path) -> Result<Self> {
        let (schema, output_fields) = Self::create_schema_from_yaml(schema_path)?;
        let dir = MmapDirectory::open(index_dir)?;
        let index = Index::open_or_create(dir, schema.clone())?;

        // Register custom tokenizers for position-aware indexing (Odinson-style)
        // 1. Edge tokenizer: for incoming_edges and outgoing_edges fields
        index.tokenizers().register(
            "edge_position_tokenizer",
            crate::tantivy_integration::position_tokenizer::PositionAwareEdgeTokenizer,
        );
        // 2. Token tokenizer: for word, lemma, pos, tag, entity, etc.
        index.tokenizers().register(
            "token_position_tokenizer",
            crate::tantivy_integration::position_tokenizer::PositionAwareTokenTokenizer,
        );
        log::info!("Registered position-aware tokenizers (edge and token)");

        let reader = index.reader()?;
        let writer = match index.writer(50_000_000) {
            Ok(w) => Some(w),
            Err(tantivy::TantivyError::LockFailure(e, _)) => {
                log::warn!("Could not acquire index lock, running in READ-ONLY mode: {}", e);
                None
            },
             Err(e) => return Err(anyhow::Error::from(e)),
        };

        let default_field = schema.get_field(FIELD_WORD).map_err(|_| anyhow!("Default field '{}' not found in schema", FIELD_WORD))?;
        let sentence_length_field = schema.get_field(FIELD_SENTENCE_LENGTH).map_err(|_| anyhow!("Sentence length field '{}' not found in schema", FIELD_SENTENCE_LENGTH))?;
        let dependencies_binary_field = schema.get_field(FIELD_DEPENDENCIES_BINARY).map_err(|_| anyhow!("Dependencies binary field '{}' not found in schema", FIELD_DEPENDENCIES_BINARY))?;
        let incoming_edges_field = schema.get_field(FIELD_INCOMING_EDGES).map_err(|_| anyhow!("Incoming edges field '{}' not found in schema", FIELD_INCOMING_EDGES))?;
        let outgoing_edges_field = schema.get_field(FIELD_OUTGOING_EDGES).map_err(|_| anyhow!("Outgoing edges field '{}' not found in schema", FIELD_OUTGOING_EDGES))?;

        Ok(Self {
            index,
            reader,
            writer,
            schema,
            default_field,
            sentence_length_field,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
            parent_doc_id_field: FIELD_DOC_ID.to_string(),
            output_fields,
        })
    }

    /// Create schema from YAML file - required, no default
    pub fn create_schema_from_yaml<P: AsRef<Path>>(schema_path: P) -> Result<(tantivy::schema::Schema, Vec<String>)> {
        let path = schema_path.as_ref();
        if !path.exists() {
            return Err(anyhow!("Schema file not found: {}", path.display()));
        }

        let yaml_str = fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read schema file {}: {}", path.display(), e))?;

        let config: SchemaConfig = serde_yaml::from_str(&yaml_str)
            .map_err(|e| anyhow!("Invalid YAML schema in {}: {}", path.display(), e))?;

        let mut builder = tantivy::schema::Schema::builder();
        for field in config.fields {
            match field.field_type.as_str() {
                "text" => {
                    // Index text fields with positions (like Odinson) for constraint prefiltering
                    use tantivy::schema::{TextFieldIndexing, TextOptions, IndexRecordOption};
                    let indexing = TextFieldIndexing::default()
                        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
                    let mut options = TextOptions::default().set_indexing_options(indexing);
                    if field.stored {
                        options = options.set_stored();
                    }
                    builder.add_text_field(&field.name, options);
                    log::debug!("Added text field '{}' with positions", field.name);
                }
                "string" => {
                    // Index string fields with positions (Odinson-style) for constraint prefiltering
                    // String fields are token fields (word, tag, entity, etc.)
                    // CRITICAL: Use position-aware tokenizer to ensure each token gets correct position
                    // Input format: "John|eats|pizza" -> positions 0, 1, 2
                    use tantivy::schema::{TextFieldIndexing, TextOptions, IndexRecordOption};
                    let indexing = TextFieldIndexing::default()
                        .set_tokenizer("token_position_tokenizer")  // Position-aware tokenizer
                        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
                    let mut options = TextOptions::default().set_indexing_options(indexing);
                    if field.stored {
                        options = options.set_stored();
                    }
                    builder.add_text_field(&field.name, options);
                    log::debug!("Added string field '{}' with position-aware tokenizer", field.name);
                }
                // New type: position-aware edge fields (Odinson-style)
                // Uses custom tokenizer that preserves token positions
                "edge_positions" => {
                    use tantivy::schema::{TextFieldIndexing, TextOptions, IndexRecordOption};

                    // Use position-aware indexing with custom tokenizer
                    let indexing = TextFieldIndexing::default()
                        .set_tokenizer("edge_position_tokenizer")
                        .set_index_option(IndexRecordOption::WithFreqsAndPositions);

                    let mut options = TextOptions::default().set_indexing_options(indexing);
                    if field.stored {
                        options = options.set_stored();
                    }
                    builder.add_text_field(&field.name, options);
                    log::info!("Added position-aware edge field: {}", field.name);
                }
                "u64" => {
                    // For sentence_length, we need it as both STORED and FAST for efficient postings path
                    if field.name == FIELD_SENTENCE_LENGTH {
                        builder.add_u64_field(&field.name, tantivy::schema::STORED | tantivy::schema::FAST);
                        log::debug!("Added u64 field '{}' as STORED and FAST for postings path", field.name);
                    } else {
                        builder.add_u64_field(&field.name, tantivy::schema::STORED);
                    }
                }
                "bytes" => {
                    builder.add_bytes_field(&field.name, tantivy::schema::STORED);
                }
                _ => return Err(anyhow!("Unknown field type in schema: {}", field.field_type)),
            }
        }

        // Get output fields from config, with defaults if not specified
        let output_fields = config.output_fields.unwrap_or_else(|| {
            vec![FIELD_WORD.to_string(), FIELD_LEMMA.to_string(), FIELD_POS.to_string(), FIELD_ENTITY.to_string()]
        });

        let schema = builder.build();
        log::info!("Schema created: all text/string fields indexed with positions (Odinson-style)");
        
        Ok((schema, output_fields))
    }

    /// Get the number of documents in the index
    pub fn num_docs(&self) -> usize {
        self.reader.searcher().num_docs().try_into().unwrap()
    }

    /// Get a document by its address
    pub fn doc(&self, doc_address: DocAddress) -> Result<TantivyDocument> {
        self.reader.searcher().doc(doc_address).map_err(anyhow::Error::from)
    }

    /// Execute a query string and return results
    pub fn query(&self, query: &str) -> Result<RustIeResult> {
        self.query_with_limit(query, self.num_docs())
    }

    /// Execute a query string with a limit on results
    pub fn query_with_limit(&self, query: &str, limit: usize) -> Result<RustIeResult> {
        // Parse the query string to get the original pattern
        use crate::compiler::parser::QueryParser;
        let parser = QueryParser::new(FIELD_WORD.to_string());
        let pattern = parser.parse_query(query)?;
        log::debug!("Pattern AST type = {:?}", std::any::type_name_of_val(&pattern));
        log::debug!("Pattern AST = {:?}", pattern);
        let tantivy_query = self.compiler().compile(query)?;
        log::debug!("Compiled query type = {:?}", std::any::type_name_of_val(tantivy_query.as_ref()));
        
        // Try to downcast to see if it's actually an OptimizedGraphTraversalQuery
        /*
        // MAJOR BUG: We were passing &tantivy_query (a reference to a Box<dyn Query>) 
        // to the downcast_ref method, rather than tantivy_query.as_ref() which gives us 
        // the actual &dyn Query trait object. This prevented proper downcasting to 
        // OptimizedGraphTraversalQuery and caused the graph traversal optimization to be 
        // bypassed entirely, falling back to slower pattern matching.
        //
        // Additionally, when graph matches are available, we should prioritize them and 
        // skip the pattern matching entirely for better performance, but the previous code 
        // was still attempting pattern matching regardless.
        */
        if let Some(_graph_query) = tantivy_query.as_any().downcast_ref::<crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery>() {
            log::debug!("Query is actually an OptimizedGraphTraversalQuery!");
        } else {
            log::debug!("Query is NOT an OptimizedGraphTraversalQuery");
        }
        
        // Determine if this is a graph query
        let is_graph_query = tantivy_query.as_any().downcast_ref::<crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery>().is_some();
        
        /*
        Entry point for the query execution
        */
        self.execute_query(tantivy_query.as_ref(),
                           limit,
                           &pattern,
                           is_graph_query)
    }

    /// Execute a compiled query, now with the original pattern for match extraction
    pub fn execute_query(&self, query: &dyn Query,
                         limit: usize,
                         pattern: &crate::compiler::ast::Pattern,
                         is_graph_query: bool) -> Result<RustIeResult> {
        
        log::debug!("=== EXECUTION PATH ANALYSIS ===");
        log::debug!("Pattern type: {:?}", std::any::type_name_of_val(pattern));
        log::debug!("is_graph_query: {}", is_graph_query);
        log::debug!("Query type: {:?}", std::any::type_name_of_val(query));

        // CLEAR SEPARATION: Choose execution path based on pattern type
        match pattern {
            // Graph traversal pattern
            crate::compiler::ast::Pattern::GraphTraversal { .. } => {
                log::debug!("=== GRAPH TRAVERSAL EXECUTION PATH ===");
                self.execute_graph_traversal(query, limit, pattern)
            }
            // Pattern matching patterns (Concatenated and Constraint)
            crate::compiler::ast::Pattern::Concatenated { .. } |
            crate::compiler::ast::Pattern::Constraint { .. } => {
                log::debug!("=== PATTERN MATCHING EXECUTION PATH ===");
                self.execute_pattern_matching(query, limit, pattern)
            }
            _ => {
                log::debug!("=== FALLBACK EXECUTION PATH ===");
                self.execute_fallback(query, limit, pattern)
            }
        }
    }

    /// Execute graph traversal queries using dependency graph edges
    /// OPTIMIZED: Parallel segment processing + Single-pass collection
    fn execute_graph_traversal(&self, query: &dyn Query, limit: usize, _pattern: &crate::compiler::ast::Pattern) -> Result<RustIeResult> {
        log::debug!("Using GRAPH TRAVERSAL mechanism (PARALLEL + SINGLE-PASS)");

        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        // Downcast to graph query once
        let graph_query = match query
            .as_any()
            .downcast_ref::<crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery>()
        {
            Some(gq) => gq,
            None => {
                log::warn!("Expected graph query but downcast failed!");
                return Ok(RustIeResult {
                    total_hits: 0,
                    score_docs: Vec::new(),
                    sentence_results: Vec::new(),
                    max_score: None,
                });
            }
        };

        // PARALLEL: Process all segments concurrently using Rayon
        // Each segment is processed independently, results are merged at the end
        let segment_results: Vec<(Vec<(SentenceResult, Score)>, u32)> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);

                // Create weight and scorer for this segment
                let weight = match graph_query.weight(tantivy::query::EnableScoring::Enabled {
                    searcher: &searcher,
                    statistics_provider: &searcher
                }) {
                    Ok(w) => w,
                    Err(e) => {
                        log::warn!("Failed to create weight for segment {}: {}", segment_ord, e);
                        return None;
                    }
                };

                let mut scorer = match weight.scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("Failed to create scorer for segment {}: {}", segment_ord, e);
                        return None;
                    }
                };

                let mut segment_sentence_results = Vec::new();

                // Iterate through all matching documents in this segment
                loop {
                    let doc_id = scorer.doc();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);

                    // Get matches from scorer (already computed during advance)
                    let matches = if let Some(graph_scorer) = scorer
                        .as_any()
                        .downcast_ref::<crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalScorer>()
                    {
                        graph_scorer.get_current_doc_matches().to_vec()
                    } else {
                        Vec::new()
                    };

                    // Extract sentence result - need to get doc from searcher
                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) = self.extract_sentence_result(&doc, score) {
                            sentence_result.matches = matches;
                            segment_sentence_results.push((sentence_result, score));
                        }
                    }

                    // Advance to next matching document
                    if scorer.advance() == tantivy::TERMINATED {
                        break;
                    }
                }

                Some((segment_sentence_results, segment_ord as u32))
            })
            .collect();

        // MERGE: Combine results from all segments, sort by score, apply limit
        let mut all_results: Vec<(SentenceResult, Score)> = segment_results
            .into_iter()
            .flat_map(|(results, _)| results)
            .collect();

        // Sort by score descending (highest scores first)
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // DEDUPLICATE: Remove duplicates based on (document_id, sentence_id), keeping highest score
        let mut seen: std::collections::HashMap<String, SentenceResult> = std::collections::HashMap::new();
        for (result, score) in all_results {
            let key = format!("{}:{}", result.document_id, result.sentence_id);
            match seen.get(&key) {
                Some(existing) => {
                    // Keep the one with higher score
                    if score > existing.score {
                        seen.insert(key, result);
                    }
                }
                None => {
                    seen.insert(key, result);
                }
            }
        }
        
        // Build deduplicated results, sorted by score (already sorted from HashMap iteration, but ensure order)
        let mut deduplicated: Vec<SentenceResult> = seen.into_values().collect();
        deduplicated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply limit after deduplication
        deduplicated.truncate(limit);

        let max_score = deduplicated.first().map(|r| r.score);

        log::debug!("Graph traversal found {} results using parallel segments", deduplicated.len());

        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(),
            sentence_results: deduplicated,
            max_score,
        })
    }

    /// Execute pattern matching queries using token sequence matching
    fn execute_pattern_matching(&self, query: &dyn Query,
         limit: usize,
         pattern: &crate::compiler::ast::Pattern) -> Result<RustIeResult> {
        log::debug!("Using PATTERN MATCHING mechanism");

        // Check if this is a RustieConcatQuery
        if let Some(pattern_query) = query.as_any().downcast_ref::<crate::tantivy_integration::concat_query::RustieConcatQuery>() {
            log::debug!("Detected RustieConcatQuery - using custom pattern matching scorer");
            return self.execute_optimized_pattern_matching(pattern_query, limit);
        }

        // Check if this is a RustieNamedCaptureQuery
        if let Some(named_query) = query.as_any().downcast_ref::<crate::tantivy_integration::named_capture_query::RustieNamedCaptureQuery>() {
            log::debug!("Detected RustieNamedCaptureQuery - using custom named capture scorer");
            return self.execute_named_capture_matching(named_query, limit);
        }

        log::debug!("Using standard Tantivy pattern matching");

        let searcher = self.reader.searcher();
        let top_docs = searcher.search(query, &TopDocs::with_limit(limit)).map_err(anyhow::Error::from)?;
        log::debug!("top_docs = {:?}", top_docs);

        let mut sentence_results = Vec::new();
        let mut score_docs = Vec::new();
        let mut max_score = None;

        for (score, doc_address) in top_docs {
            score_docs.push(RustieDoc::new(doc_address, score));
            
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;
                let tokens = self.extract_field_values(&doc, FIELD_WORD);

                log::debug!("Extracting pattern matches using direct token positions");
                log::debug!("Tokens in document: {:?}", tokens);

                // Use pattern's built-in extract_matching_positions instead of redundant function
                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);
                log::debug!("Match positions found: {:?}", match_positions);

                let mut pattern_matches = Vec::new();

                // Create captures for each matching position
                for (i, &pos) in match_positions.iter().enumerate() {
                    let span = crate::types::Span { start: pos, end: pos + 1 };
                    let capture = crate::types::NamedCapture::new(format!("c{}", i), span.clone());
                    pattern_matches.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }

                log::debug!("Pattern matches = {:?}", pattern_matches);
                log::debug!("Number of pattern matches: {}", pattern_matches.len());
                sentence_result.matches = pattern_matches;
                sentence_results.push(sentence_result);
            }

            max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
        }
        
        // DEDUPLICATE: Remove duplicates based on (document_id, sentence_id), keeping highest score
        let mut seen: std::collections::HashMap<String, (usize, Score)> = std::collections::HashMap::new();
        for (idx, result) in sentence_results.iter().enumerate() {
            let key = format!("{}:{}", result.document_id, result.sentence_id);
            match seen.get(&key) {
                Some(&(existing_idx, existing_score)) => {
                    // Keep the one with higher score
                    if result.score > existing_score {
                        seen.insert(key, (idx, result.score));
                    }
                }
                None => {
                    seen.insert(key, (idx, result.score));
                }
            }
        }
        
        // Build deduplicated results, preserving order by score (highest first)
        let mut deduplicated: Vec<SentenceResult> = seen
            .into_iter()
            .map(|(_, (idx, _))| sentence_results[idx].clone())
            .collect();
        
        // Sort by score descending to maintain order
        deduplicated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply limit after deduplication
        deduplicated.truncate(limit);
        
        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs,
            sentence_results: deduplicated,
            max_score,
        })
    }

    /// Execute optimized pattern matching queries using custom scorer
    /// OPTIMIZED: Parallel segment processing
    fn execute_optimized_pattern_matching(&self, pattern_query: &crate::tantivy_integration::concat_query::RustieConcatQuery, limit: usize) -> Result<RustIeResult> {
        log::debug!("=== OPTIMIZED PATTERN MATCHING (PARALLEL) ===");

        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        // PARALLEL: Process all segments concurrently using Rayon
        let segment_results: Vec<Vec<(SentenceResult, Score)>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);

                // Create weight and scorer for this segment
                let weight = match pattern_query.weight(tantivy::query::EnableScoring::Enabled {
                    searcher: &searcher,
                    statistics_provider: &searcher
                }) {
                    Ok(w) => w,
                    Err(e) => {
                        log::warn!("Failed to create weight for segment {}: {}", segment_ord, e);
                        return None;
                    }
                };

                let mut scorer = match weight.scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("Failed to create scorer for segment {}: {}", segment_ord, e);
                        return None;
                    }
                };

                let mut segment_sentence_results = Vec::new();

                // Iterate through all matching documents in this segment
                loop {
                    let doc_id = scorer.advance();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);

                    // Get matches from scorer (already computed during advance/pattern matching)
                    let matches = if let Some(pattern_scorer) = scorer
                        .as_any()
                        .downcast_ref::<crate::tantivy_integration::concat_query::RustieConcatScorer>()
                    {
                        pattern_scorer.get_current_doc_matches().to_vec()
                    } else {
                        Vec::new()
                    };

                    // Extract sentence result
                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) = self.extract_sentence_result(&doc, score) {
                            sentence_result.matches = matches;
                            segment_sentence_results.push((sentence_result, score));
                        }
                    }
                }

                Some(segment_sentence_results)
            })
            .collect();

        // MERGE: Combine results from all segments, sort by score, apply limit
        let mut all_results: Vec<(SentenceResult, Score)> = segment_results
            .into_iter()
            .flatten()
            .collect();

        // Sort by score descending (highest scores first)
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // DEDUPLICATE: Remove duplicates based on (document_id, sentence_id), keeping highest score
        let mut seen: std::collections::HashMap<String, SentenceResult> = std::collections::HashMap::new();
        for (result, score) in all_results {
            let key = format!("{}:{}", result.document_id, result.sentence_id);
            match seen.get(&key) {
                Some(existing) => {
                    // Keep the one with higher score
                    if score > existing.score {
                        seen.insert(key, result);
                    }
                }
                None => {
                    seen.insert(key, result);
                }
            }
        }
        
        // Build deduplicated results, sorted by score
        let mut deduplicated: Vec<SentenceResult> = seen.into_values().collect();
        deduplicated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply limit after deduplication
        deduplicated.truncate(limit);

        let max_score = deduplicated.first().map(|r| r.score);

        log::debug!("Optimized pattern matching found {} results using parallel segments", deduplicated.len());

        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(),
            sentence_results: deduplicated,
            max_score,
        })
    }

    /// Execute named capture pattern matching queries using custom scorer
    fn execute_named_capture_matching(&self, named_query: &crate::tantivy_integration::named_capture_query::RustieNamedCaptureQuery, limit: usize) -> Result<RustIeResult> {
        log::debug!("=== NAMED CAPTURE EXECUTION PATH ===");

        let searcher = self.reader.searcher();
        let top_docs = searcher.search(named_query, &TopDocs::with_limit(limit)).map_err(anyhow::Error::from)?;
        log::debug!("top_docs = {:?}", top_docs);

        let mut sentence_results = Vec::new();
        let mut max_score = None;

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;

                // Get the segment order and document ID
                let (segment_ord, _) = (doc_address.segment_ord, doc_address.doc_id);
                let segment_reader = searcher.segment_reader(segment_ord);

                log::debug!("Creating weight for named capture query");
                let weight = named_query.weight(tantivy::query::EnableScoring::Enabled {
                    searcher: &searcher,
                    statistics_provider: &searcher
                })?;

                log::debug!("Creating scorer from weight");
                let scorer = weight.scorer(segment_reader, 1.0)?;

                // Get matches from the custom scorer
                if let Some(named_scorer) = scorer.as_any().downcast_ref::<crate::tantivy_integration::named_capture_query::RustieNamedCaptureScorer>() {
                    let matches = named_scorer.get_current_doc_matches();
                    log::debug!("Named captures = {:?}", matches);
                    sentence_result.matches = matches.to_vec();
                } else {
                    log::debug!("Could not downcast to RustieNamedCaptureScorer");
                    sentence_result.matches = Vec::new();
                }

                sentence_results.push(sentence_result);
            }

            max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
        }
        
        // DEDUPLICATE: Remove duplicates based on (document_id, sentence_id), keeping highest score
        let mut seen: std::collections::HashMap<String, SentenceResult> = std::collections::HashMap::new();
        for result in sentence_results {
            let key = format!("{}:{}", result.document_id, result.sentence_id);
            match seen.get(&key) {
                Some(existing) => {
                    // Keep the one with higher score
                    if result.score > existing.score {
                        seen.insert(key, result);
                    }
                }
                None => {
                    seen.insert(key, result);
                }
            }
        }
        
        // Build deduplicated results, sorted by score
        let mut deduplicated: Vec<SentenceResult> = seen.into_values().collect();
        deduplicated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply limit after deduplication
        deduplicated.truncate(limit);
        
        // Update max_score from deduplicated results
        let max_score = deduplicated.first().map(|r| r.score).or(max_score);
        
        Ok(RustIeResult {
            total_hits: deduplicated.len(),
            score_docs: Vec::new(),
            sentence_results: deduplicated,
            max_score,
        })
    }

    /// Execute fallback for other pattern types
    fn execute_fallback(&self, query: &dyn Query, limit: usize, pattern: &crate::compiler::ast::Pattern) -> Result<RustIeResult> {
        log::debug!("Using FALLBACK mechanism for pattern type: {:?}", std::any::type_name_of_val(pattern));
        
        let searcher = self.reader.searcher();
        let top_docs = searcher.search(query, &TopDocs::with_limit(limit)).map_err(anyhow::Error::from)?;
        
        let mut sentence_results = Vec::new();
        let mut max_score = None;

        for (score, doc_address) in top_docs {
            if let Ok(doc) = self.doc(doc_address) {
                let mut sentence_result = self.extract_sentence_result(&doc, score)?;
                let tokens = self.extract_field_values(&doc, FIELD_WORD);

                // Use the pattern to find all matching positions (single-token matches)
                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);
                // For single-token matches, assign a random name to each capture
                use rand::{distributions::Alphanumeric, Rng};
                let mut fallback_matches = Vec::new();
                for start in match_positions {
                    let span = crate::types::Span { start, end: start + 1 };
                    let rand_name: String = rand::thread_rng()
                        .sample_iter(&Alphanumeric)
                        .take(8)
                        .map(char::from)
                        .collect();
                    let capture = crate::types::NamedCapture::new(rand_name, span.clone());
                    fallback_matches.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }
                
                sentence_result.matches = fallback_matches;
                sentence_results.push(sentence_result);
            }

            max_score = max_score.map(|s: Score| s.max(score)).or(Some(score));
        }
        
        Ok(RustIeResult {
            total_hits: sentence_results.len(),
            score_docs: Vec::new(),
            sentence_results,
            max_score,
        })
    }

    /// Extract sentence result from a Tantivy document
    fn extract_sentence_result(&self, doc: &TantivyDocument, score: Score) -> Result<SentenceResult> {

        // Extract document fields
        let document_id = self.extract_field_value(doc, FIELD_DOC_ID)
            .unwrap_or_else(|| "unknown".to_string());

        let sentence_id = self.extract_field_value(doc, FIELD_SENTENCE_ID)
            .unwrap_or_else(|| "0".to_string());
        
        // Extract all configured output fields dynamically
        let mut field_values = std::collections::HashMap::new();
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
    fn extract_field_value(&self, doc: &TantivyDocument, field_name: &str) -> Option<String> {
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
    fn extract_field_values(&self, doc: &TantivyDocument, field_name: &str) -> Vec<String> {
        if let Ok(field) = self.schema.get_field(field_name) {
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

    /// Add a document to the index
    pub fn add_document(&mut self, document: &Document) -> Result<()> {
        let mut parser = DocumentParser::new(self.schema.clone());
        
        // Use the engine's vocabulary
        // *parser.vocabulary_mut() = self.vocabulary.clone(); // Removed vocabulary usage

        let tantivy_docs = parser.to_tantivy_document(document)?;
        
        // Update the engine's vocabulary with any new labels
        // self.vocabulary = parser.vocabulary().clone(); // Removed vocabulary usage
        
        if let Some(writer) = &mut self.writer {
             for tantivy_doc in tantivy_docs {
                writer.add_document(tantivy_doc)?;
            }
        } else {
            return Err(anyhow!("Cannot add document: Engine is in READ-ONLY mode (index lock could not be acquired)"));
        }
        
        Ok(())
    }

    /// Add multiple documents to the index
    pub fn add_documents(&mut self, documents: &[Document]) -> Result<()> {
        
        // Here documents will be added in batches using multiple threads to speed up the process
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
        
        // Save the vocabulary
        // self.save_vocabulary()?; // Removed vocabulary saving
        
        Ok(())
    }

    /// Get the index writer (for advanced usage)
    pub fn writer(&mut self) -> Result<&mut IndexWriter> {
        self.writer.as_mut().ok_or_else(|| anyhow!("Engine is in READ-ONLY mode"))
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the default field
    pub fn default_field(&self) -> Field {
        self.default_field
    }

    /// Get the sentence length field
    pub fn sentence_length_field(&self) -> Field {
        self.sentence_length_field
    }

    /// Get the dependencies binary field
    pub fn dependencies_binary_field(&self) -> Field {
        self.dependencies_binary_field
    }

    /// Get the searcher for direct access
    pub fn searcher(&self) -> tantivy::Searcher {
        self.reader.searcher()
    }

    /// Get the parent document ID field name
    pub fn parent_doc_id_field(&self) -> &str {
        &self.parent_doc_id_field
    }

    /// Get the output fields
    pub fn output_fields(&self) -> &[String] {
        &self.output_fields
    }

    /// Check if a field is an output field
    pub fn is_output_field(&self, field_name: &str) -> bool {
        self.output_fields.iter().any(|f| f == field_name)
    }

    /// Get output field names
    pub fn get_output_field_names(&self) -> Vec<&String> {
        self.output_fields.iter().collect()
    }

    /// Get the query compiler
    pub fn compiler(&self) -> QueryCompiler {
        QueryCompiler::new(self.schema.clone())
    }
}



impl ExtractorEngine {
    /// Create an ExtractorEngine from a string path
    pub fn from_path(index_dir: &str) -> Result<Self> {
        let schema_path = Path::new("configs/schema.yaml");
        Self::new(Path::new(index_dir), schema_path)
    }
} 