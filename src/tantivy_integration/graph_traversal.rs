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
use crate::compiler::ast::{Pattern, Traversal};

// Global counter for generating unique capture names (much faster than rand)
static CAPTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);



// Optimized graph traversal query that first finds documents containing both source and destination tokens
#[derive(Debug)]
pub struct OptimizedGraphTraversalQuery {
    default_field: Field,
    dependencies_binary_field: Field,
    incoming_edges_field: Field,
    outgoing_edges_field: Field,
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
        incoming_edges_field: Field,
        outgoing_edges_field: Field,
        src_query: Box<dyn Query>,
        traversal: crate::compiler::ast::Traversal,
        dst_query: Box<dyn Query>,
        src_pattern: crate::compiler::ast::Pattern,
        dst_pattern: crate::compiler::ast::Pattern,
    ) -> Self {
        Self {
            default_field,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
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
        
        // Note: Edge label filters are now included in src_query and dst_query 
        // (added in graph_compiler.rs when creating combined_query)
        // So we can use them directly without additional wrapping
        
        log::info!("Using combined_query (includes constraints + edge label filters) for candidate selection");
        
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
            incoming_edges_field: self.incoming_edges_field,
            outgoing_edges_field: self.outgoing_edges_field,
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
    /// 
    /// Optimizations:
    /// 1. Pre-checks ALL constraints before graph deserialization (early exit)
    /// 2. Uses boolean query for early termination during traversal
    /// 3. Tracks skipped graph deserializations for profiling
    fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {
        // Static counters for profiling
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        static GRAPH_DESER_COUNT: AtomicUsize = AtomicUsize::new(0);
        static GRAPH_DESER_SKIPPED: AtomicUsize = AtomicUsize::new(0);

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

        // Early exit if pattern is empty
        if flat_steps.is_empty() {
            debug!("Empty pattern, skipping");
            GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Extract tokens for each constraint field (using cached field names)
        let mut constraint_fields_and_tokens: Vec<(String, Vec<String>)> =
            Vec::with_capacity(self.constraint_field_names.len());
        for field_name in &self.constraint_field_names {
            let tokens = self.extract_tokens_from_field(&doc, field_name);
            constraint_fields_and_tokens.push((field_name.clone(), tokens));
        }

        // OPTIMIZATION 1: Pre-check ALL constraints and CACHE positions before graph deserialization
        // This avoids redundant position finding later
        let mut constraint_count = 0;
        let mut total_constraints = 0;
        let mut cached_positions: Vec<Vec<usize>> = Vec::new();  // Cache positions for each constraint

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(constraint_pat) = step {
                total_constraints += 1;
                // Check if this constraint has matches
                // constraint_count is the index into constraint_fields_and_tokens
                if constraint_count >= constraint_fields_and_tokens.len() {
                    debug!("Constraint index {} out of bounds (len={}), skipping graph deserialization",
                           constraint_count, constraint_fields_and_tokens.len());
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }

                let (field_name, tokens) = &constraint_fields_and_tokens[constraint_count];

                // Handle wildcard constraints - they always match, so skip pre-check
                let is_wildcard = match constraint_pat {
                    crate::compiler::ast::Pattern::Constraint(
                        crate::compiler::ast::Constraint::Wildcard
                    ) => true,
                    _ => false,
                };

                if !is_wildcard {
                    // For non-wildcard constraints, find and cache positions
                    let positions = self.find_positions_in_tokens(tokens, constraint_pat);

                    if positions.is_empty() {
                        debug!("Constraint {} (field: {}) has no matches, skipping graph deserialization",
                               constraint_count, field_name);
                        GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                        return false;  // Early exit - don't deserialize graph
                    }
                    cached_positions.push(positions);
                } else {
                    // Wildcard matches all positions
                    cached_positions.push((0..tokens.len()).collect());
                }

                constraint_count += 1;
            }
            // Traversals don't need pre-checking
        }

        // Log why we're not skipping (for debugging)
        if call_num > 0 && call_num % 500 == 0 {
            debug!("Document {}: {} constraints checked, all passed (no skip)", doc_id, total_constraints);
        }

        // Use cached positions for src (first constraint)
        let src_positions = cached_positions.get(0).cloned().unwrap_or_default();

        // If no source positions and we have constraints, early exit
        if constraint_count > 0 && src_positions.is_empty() {
            debug!("src_positions empty and pattern has constraints, skipping");
            GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Phase 4: Get binary dependency graph (only if all constraints passed)
        let binary_data = match doc.get_first(self.dependencies_binary_field).and_then(|v| v.as_bytes()) {
            Some(data) => data,
            None => {
                debug!("No binary dependency graph");
                GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };

        // Phase 5: Deserialize the graph (only reached if all constraints have matches)
        GRAPH_DESER_COUNT.fetch_add(1, Ordering::Relaxed);
        let graph = match DirectedGraph::from_bytes(binary_data) {
            Ok(graph) => graph,
            Err(_) => {
                debug!("Failed to deserialize graph");
                return false;
            }
        };

        // Log candidate count periodically with skipped count
        if call_num > 0 && call_num % 500 == 0 {
            let deser_count = GRAPH_DESER_COUNT.load(Ordering::Relaxed);
            let skipped_count = GRAPH_DESER_SKIPPED.load(Ordering::Relaxed);
            let skip_rate = if call_num > 0 {
                (skipped_count as f64 / call_num as f64) * 100.0
            } else {
                0.0
            };
            log::info!("Graph traversal stats: {} candidates checked, {} graphs deserialized ({} skipped, {:.1}% skip rate)",
                call_num, deser_count, skipped_count, skip_rate);
        }

        // Phase 7: Run automaton traversal for each src_pos with early termination
        let traversal_engine = crate::digraph::traversal::GraphTraversal::new(graph);
        let mut found = false;

        // Handle traversal-only patterns (no constraints)
        if constraint_count == 0 && src_positions.is_empty() {
            // For traversal-only patterns, we'd need to start from root nodes or all nodes
            // This is a complex edge case - for now, return false
            debug!("Traversal-only pattern without source positions not yet supported");
            return false;
        }

        // OPTIMIZATION: Use cached positions for dst (last constraint)
        // This is like Odinson's mkInvIndex - we know which positions could be valid endpoints
        let dst_positions: std::collections::HashSet<usize> = if constraint_count > 0 {
            cached_positions.last()
                .map(|pos| pos.iter().cloned().collect())
                .unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        };

        // Early exit: if dst_positions is empty and we have constraints, no match possible
        if !dst_positions.is_empty() || constraint_count == 0 {
            // Continue with traversal
        } else {
            debug!("dst_positions empty, no possible matches");
            return false;
        }

        for &src_pos in &src_positions {
            // OPTIMIZATION: Skip if src == dst for multi-hop patterns (they must be different positions)
            // For single-constraint patterns, this check is not needed
            if constraint_count > 1 && dst_positions.len() == 1 && dst_positions.contains(&src_pos) {
                // Only one dst position and it's the same as src - skip for multi-hop patterns
                continue;
            }

            // Directly collect paths (removed double traversal - automaton_query was redundant)
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
                // Early exit after first matching src_pos (we found at least one match)
                // This significantly speeds up queries where we only care about existence
                break;
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
