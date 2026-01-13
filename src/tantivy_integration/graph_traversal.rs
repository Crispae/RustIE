use tantivy::{
    query::{Query, Weight, EnableScoring, Scorer},
    schema::{Field, Value},
    DocId, Score, SegmentReader,
    Result as TantivyResult,
    DocSet,
    store::StoreReader,
};
use log::debug;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::compiler::ast::FlatPatternStep;
use crate::digraph::graph::DirectedGraph;
use crate::compiler::ast::{Pattern, Constraint, Traversal};

// Global counter for generating unique capture names (much faster than rand)
static CAPTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);



// Optimized graph traversal query that first finds documents containing both source and destination tokens
#[derive(Debug)]
pub struct OptimizedGraphTraversalQuery {
    default_field: Field,
    dependencies_binary_field: Field,
    src_query: Box<dyn Query>,
    traversal: crate::compiler::ast::Traversal,
    dst_query: Box<dyn Query>,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
}


impl OptimizedGraphTraversalQuery {
    pub fn new(
        default_field: Field,
        dependencies_binary_field: Field,
        src_query: Box<dyn Query>,
        traversal: crate::compiler::ast::Traversal,
        dst_query: Box<dyn Query>,
        src_pattern: crate::compiler::ast::Pattern,
        dst_pattern: crate::compiler::ast::Pattern,
    ) -> Self {
        Self {
            default_field,
            dependencies_binary_field,
            src_query,
            traversal,
            dst_query,
            src_pattern,
            dst_pattern,
        }
    }
}

impl Query for OptimizedGraphTraversalQuery {

    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let src_weight = self.src_query.weight(scoring)?;
        let dst_weight = self.dst_query.weight(scoring)?;

        // Pre-compute flattened pattern once at Weight creation (not per document)
        let full_pattern = Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        };
        let mut flat_steps = Vec::new();
        flatten_graph_traversal_pattern(&full_pattern, &mut flat_steps);

        Ok(Box::new(OptimizedGraphTraversalWeight {
            src_weight,
            dst_weight,
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
            traversal: self.traversal.clone(),
            dependencies_binary_field: self.dependencies_binary_field,
            flat_steps, // Cached flattened pattern
        }))
    }
}

impl tantivy::query::QueryClone for OptimizedGraphTraversalQuery {
    fn box_clone(&self) -> Box<dyn Query> {
        Box::new(OptimizedGraphTraversalQuery {
            default_field: self.default_field,
            dependencies_binary_field: self.dependencies_binary_field,
            src_query: self.src_query.box_clone(),
            traversal: self.traversal.clone(),
            dst_query: self.dst_query.box_clone(),
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
        })
    }
}

/// Optimized weight for graph traversal queries
struct OptimizedGraphTraversalWeight {
    src_weight: Box<dyn Weight>,
    dst_weight: Box<dyn Weight>,
    traversal: crate::compiler::ast::Traversal,
    dependencies_binary_field: Field,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
    /// Pre-computed flattened pattern steps (cached once per query)
    flat_steps: Vec<FlatPatternStep>,
}

impl Weight for OptimizedGraphTraversalWeight {

    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        // Create scorers once (fixed: was previously creating twice wastefully)
        let src_scorer = self.src_weight.scorer(reader, boost)?;
        let dst_scorer = self.dst_weight.scorer(reader, boost)?;

        // Cache the store reader (created once, reused for all documents in this segment)
        let store_reader = reader.get_store_reader(1)?;

        // Pre-extract constraint field names from flat_steps (computed once, not per document)
        let constraint_field_names: Vec<String> = self.flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(pat) = step {
                    let field_name = match pat {
                        crate::compiler::ast::Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, .. }) => name.clone(),
                        _ => "word".to_string(),
                    };
                    Some(field_name)
                } else {
                    None
                }
            })
            .collect();

        let mut scorer = OptimizedGraphTraversalScorer {
            src_scorer,
            dst_scorer,
            traversal: self.traversal.clone(),
            dependencies_binary_field: self.dependencies_binary_field,
            reader: reader.clone(),
            store_reader,
            current_doc: None,
            current_matches: Vec::new(),
            match_index: 0,
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
            current_doc_matches: Vec::new(),
            boost,
            // Pass cached flattened pattern (computed once per query, not per document)
            flat_steps: self.flat_steps.clone(),
            constraint_field_names,
        };

        // Advance to the first document
        let _ = scorer.advance();

        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("OptimizedGraphTraversalQuery", Score::default()))
    }
}

/// Optimized scorer for graph traversal queries
pub struct OptimizedGraphTraversalScorer {
    src_scorer: Box<dyn Scorer>,
    dst_scorer: Box<dyn Scorer>,
    traversal: crate::compiler::ast::Traversal,
    dependencies_binary_field: Field,
    reader: SegmentReader,
    /// Cached store reader (created once, reused for all documents)
    store_reader: StoreReader,
    current_doc: Option<DocId>,
    current_matches: Vec<(DocId, Score)>,
    match_index: usize,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
    /// Boost factor from weight creation
    boost: Score,
    /// Pre-computed flattened pattern steps (cached from Weight)
    flat_steps: Vec<FlatPatternStep>,
    /// Pre-extracted constraint field names (cached from flat_steps)
    constraint_field_names: Vec<String>,
}

impl OptimizedGraphTraversalScorer {
    /// Compute sloppy frequency factor based on span width (Odinson-style)
    /// Uses the formula: 1.0 / (1.0 + distance) where distance = span width
    /// Shorter spans get higher scores
    fn compute_slop_factor(span_width: usize) -> Score {
        1.0 / (1.0 + span_width as f32)
    }

    /// Compute Odinson-style score for the current document
    /// Accumulates sloppy frequency for all matches, similar to Lucene/Odinson
    fn compute_odinson_score(&self) -> Score {
        if self.current_doc_matches.is_empty() {
            return 0.0;
        }

        // Accumulate sloppy frequency for all matches (Odinson approach)
        let mut acc_sloppy_freq: Score = 0.0;

        for span_match in &self.current_doc_matches {
            let span_width = span_match.span.end.saturating_sub(span_match.span.start);
            acc_sloppy_freq += Self::compute_slop_factor(span_width);
        }

        // Get base scores from sub-scorers (if available)
        // This incorporates Tantivy's BM25 scoring from the term queries
        let src_score = 1.0; // Base score since we already matched
        let dst_score = 1.0;
        let base_score = (src_score + dst_score) / 2.0;

        // Final score: base_score * accumulated_sloppy_freq * boost
        // This follows Odinson's: docScorer.score(docID(), accSloppyFreq)
        let final_score = base_score * acc_sloppy_freq * self.boost;

        // Ensure minimum score of 1.0 for any match (normalized)
        final_score.max(1.0)
    }
}

impl Scorer for OptimizedGraphTraversalScorer {

    fn score(&mut self) -> Score {
        if let Some((_, score)) = self.current_matches.get(self.match_index) {
            *score
        } else {
            Score::default()
        }
    }
}

impl tantivy::DocSet for OptimizedGraphTraversalScorer {
    
    fn advance(&mut self) -> DocId {
        
        loop {

            let src_doc = self.src_scorer.doc();
            let dst_doc = self.dst_scorer.doc();

            // If either scorer is exhausted, we're done
            if src_doc == tantivy::TERMINATED || dst_doc == tantivy::TERMINATED {
                self.current_doc = None;
                debug!("advance() terminated: src_doc = {}, dst_doc = {}", src_doc, dst_doc);
                return tantivy::TERMINATED;
            }
            debug!("advance() considering src_doc = {}, dst_doc = {}", src_doc, dst_doc);
            if src_doc < dst_doc {
                self.src_scorer.advance();
            } else if dst_doc < src_doc {
                self.dst_scorer.advance();
            } else {
                // src_doc == dst_doc: both have matches in this doc
                let doc_id = src_doc;
                debug!("advance() found candidate doc_id = {}", doc_id);
                if self.check_graph_traversal(doc_id) {
                    debug!("advance() doc_id {} MATCHED graph traversal", doc_id);
                    self.current_doc = Some(doc_id);
                    // Compute Odinson-style score based on span widths and match count
                    let score = self.compute_odinson_score();
                    debug!("Odinson-style score for doc_id {}: {}", doc_id, score);
                    self.current_matches.push((doc_id, score));
                    self.match_index = self.current_matches.len() - 1;
                    // Advance both scorers for next call
                    self.src_scorer.advance();
                    self.dst_scorer.advance();
                    return doc_id;
                } else {
                    debug!("advance() doc_id {} did NOT match graph traversal", doc_id);
                    // No match, advance both scorers
                    self.src_scorer.advance();
                    self.dst_scorer.advance();
                }
            }
        }
    }

    fn doc(&self) -> DocId {
        let doc = self.current_doc.unwrap_or(tantivy::TERMINATED);
        doc
    }

    fn size_hint(&self) -> u32 {
        // Not meaningful in this mode
        0
    }
}

impl OptimizedGraphTraversalScorer {
    

    /// Check if a document has valid graph traversal from source to destination
    fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {
        use std::time::Instant;
        // Static counters for profiling
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        static GRAPH_DESER_COUNT: AtomicUsize = AtomicUsize::new(0);

        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);

        // Clear matches from previous document
        self.current_doc_matches.clear();

        // Phase 1: Get document using cached store_reader (fast!)
        let doc = match self.store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => {
                return false;
            }
        };

        // Phase 2: Use cached flat_steps and constraint_field_names
        let flat_steps = &self.flat_steps;

        // Extract tokens for each constraint field (using cached field names)
        let mut constraint_fields_and_tokens: Vec<(String, Vec<String>)> =
            Vec::with_capacity(self.constraint_field_names.len());
        for field_name in &self.constraint_field_names {
            let tokens = self.extract_tokens_from_field(&doc, field_name);
            constraint_fields_and_tokens.push((field_name.clone(), tokens));
        }

        // Find the first constraint in the flat_steps
        let first_constraint = flat_steps.iter().find_map(|step| {
            if let FlatPatternStep::Constraint(pat) = step {
                Some(pat)
            } else {
                None
            }
        });

        // Use the tokens for the first constraint to find source positions
        let src_positions = match first_constraint {
            Some(pat) => {
                if let Some((_, tokens)) = constraint_fields_and_tokens.get(0) {
                    self.find_positions_in_tokens(tokens, pat)
                } else {
                    vec![]
                }
            },
            None => vec![],
        };

        if src_positions.is_empty() {
            debug!("src_positions empty");
            return false;
        }

        // Phase 4: Get binary dependency graph
        let binary_data = match doc.get_first(self.dependencies_binary_field).and_then(|v| v.as_bytes()) {
            Some(data) => data,
            None => {
                debug!("No binary dependency graph");
                return false;
            }
        };

        // Phase 5: Deserialize the graph
        GRAPH_DESER_COUNT.fetch_add(1, Ordering::Relaxed);
        let graph = match DirectedGraph::from_bytes(binary_data) {
            Ok(graph) => graph,
            Err(_) => {
                debug!("Failed to deserialize graph");
                return false;
            }
        };

        // Log candidate count periodically
        if call_num > 0 && call_num % 500 == 0 {
            let deser_count = GRAPH_DESER_COUNT.load(Ordering::Relaxed);
            log::info!("Graph traversal stats: {} candidates checked, {} graphs deserialized",
                call_num, deser_count);
        }

        // Phase 7: Run automaton traversal for each src_pos
        let traversal_engine = crate::digraph::traversal::GraphTraversal::new(graph);
        let mut found = false;

        for &src_pos in &src_positions {
            let all_paths = traversal_engine.automaton_query_paths(&flat_steps, &[src_pos], &constraint_fields_and_tokens);
            for path in &all_paths {
                // Collect captures for each constraint step
                debug!("flat_steps = {:?}", flat_steps);
                debug!("path = {:?}", path);
                let mut captures = Vec::with_capacity(path.len());
                let mut constraint_idx = 0;
                for (_step_idx, step) in flat_steps.iter().enumerate() {
                    if let FlatPatternStep::Constraint(ref pat) = step {
                        if let Some(&node_idx) = path.get(constraint_idx) {
                            let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                            let name = match pat {
                                crate::compiler::ast::Pattern::NamedCapture { name, .. } => name.clone(),
                                _ => {
                                    // Use fast atomic counter instead of random generation
                                    let id = CAPTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
                                    format!("c{}", id)
                                }
                            };
                            captures.push(crate::types::NamedCapture::new(name, span));
                        }
                        constraint_idx += 1;
                    }
                }
                if !path.is_empty() {
                    let min_pos = *path.iter().min().unwrap();
                    let max_pos = *path.iter().max().unwrap();
                    let span = crate::types::Span { start: min_pos, end: max_pos + 1 };
                    debug!("Adding SpanWithCaptures: span = {:?}, captures = {:?}", span, captures);
                    self.current_doc_matches.push(crate::types::SpanWithCaptures::with_captures(span, captures));
                }
            }
            if !all_paths.is_empty() {
                found = true;
            }
        }
        found
    }

    /// Extract the field name from a pattern
    fn get_field_name_from_pattern<'a>(&self, pattern: &'a crate::compiler::ast::Pattern) -> &'a str {
        match pattern {
            crate::compiler::ast::Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, .. }) => {
                name.as_str()
            }
            _ => "word", // default to word field
        }
    }
    
    /// Extract tokens from a specific field in the document
    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument, field_name: &str) -> Vec<String> {
        if let Ok(field) = self.reader.schema().get_field(field_name) {
            let tokens: Vec<String> = doc.get_all(field)
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            tokens
        } else {
            vec![]
        }
    }
    
    /// Find positions that match a pattern (for backward compatibility)
    fn find_positions_matching_pattern(&self, tokens: &[String], pattern: &crate::compiler::ast::Pattern) -> Vec<usize> {
        self.find_positions_in_tokens(tokens, pattern)
    }

    // NEW: expose matches for the current doc
    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        debug!("get_current_doc_matches called, current_doc_matches.len() = {}", self.current_doc_matches.len());
        debug!("current_doc_matches = {:?}", self.current_doc_matches);
        &self.current_doc_matches
    }

    /// Helper: Convert traversal AST to Pattern (for now, just wrap in GraphTraversal)
    fn traversal_to_pattern(&self) -> Pattern {
        Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        }
    }

    /// Find positions in tokens that match a given pattern (string, regex, or wildcard for any field)
    fn find_positions_in_tokens(&self, tokens: &[String], pattern: &crate::compiler::ast::Pattern) -> Vec<usize> {
        use crate::compiler::ast::{Pattern, Constraint, Matcher};
        let mut positions = Vec::new();
        match pattern {
            Pattern::Constraint(Constraint::Field { name: _, matcher }) => {
                // Supports any field - tokens are already extracted from the correct field
                match matcher {
                    Matcher::String(s) => {
                        for (i, token) in tokens.iter().enumerate() {
                            if token == s {
                                positions.push(i);
                            }
                        }
                    }
                    // Use the pre-compiled regex from the Matcher for performance
                    Matcher::Regex { regex, .. } => {
                        for (i, token) in tokens.iter().enumerate() {
                            if regex.is_match(token) {
                                positions.push(i);
                            }
                        }
                    }
                }
            }
            Pattern::Constraint(Constraint::Wildcard) => {
                // Optimize: use (0..tokens.len()).collect() instead of pushing one by one
                positions = (0..tokens.len()).collect();
            }
            _ => {}
        }
        positions
    }
}


/// Flatten a nested Pattern::GraphTraversal AST into a flat Vec<FlatPatternStep>
pub fn flatten_graph_traversal_pattern(pattern: &crate::compiler::ast::Pattern, steps: &mut Vec<FlatPatternStep>) {
    match pattern {
        Pattern::GraphTraversal { src, traversal, dst } => {
            // Always flatten src first
            flatten_graph_traversal_pattern(src, steps);
            // Then the traversal
            steps.push(FlatPatternStep::Traversal(traversal.clone()));
            // Then flatten dst
            flatten_graph_traversal_pattern(dst, steps);
        }
        Pattern::Constraint(_) => {
            steps.push(FlatPatternStep::Constraint(pattern.clone()));
        }
        // Optionally, handle other pattern types if needed
        _ => {}
    }
}
