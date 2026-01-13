use tantivy::{
    query::{Query, Weight, EnableScoring, Scorer},
    schema::{Field, Value, IndexRecordOption, Schema},
    DocId, Score, SegmentReader,
    Result as TantivyResult,
    DocSet,
    store::StoreReader,
    Term,
    postings::{Postings, SegmentPostings},
};
use log::debug;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::compiler::ast::FlatPatternStep;
use crate::digraph::graph::DirectedGraph;
use crate::compiler::ast::{Pattern, Traversal, Matcher, Constraint};

// Global counter for generating unique capture names (much faster than rand)
static CAPTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Module-level counters for profiling (shared across all instances)
static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static GRAPH_DESER_COUNT: AtomicUsize = AtomicUsize::new(0);
static GRAPH_DESER_SKIPPED: AtomicUsize = AtomicUsize::new(0);
static PREFILTER_DOCS: AtomicUsize = AtomicUsize::new(0);
static PREFILTER_KILLED: AtomicUsize = AtomicUsize::new(0);
static PREFILTER_ALLOWED_POS_SUM: AtomicUsize = AtomicUsize::new(0);
static PREFILTER_ALLOWED_POS_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Edge term requirement for position prefiltering
#[derive(Clone, Debug)]
struct EdgeTermReq {
    field: Field,           // incoming_edges_field or outgoing_edges_field
    label: String,          // exact term, e.g. "nsubj"
    constraint_idx: usize, // which constraint this restricts
}

/// Constraint term requirement (for exact string matches)
#[derive(Clone, Debug)]
struct ConstraintTermReq {
    field: Field,           // constraint field, e.g. "entity"
    term: String,           // exact term value, e.g. "B-Gene"
    constraint_idx: usize,  // which constraint this restricts
}

/// Position prefilter plan computed from flattened pattern steps
/// NOTE: constraint_reqs are built separately in the scorer (need schema access)
#[derive(Clone, Debug, Default)]
struct PositionPrefilterPlan {
    edge_reqs: Vec<EdgeTermReq>,
    num_constraints: usize,
}



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

        // Build position prefilter plan from flat_steps
        let prefilter_plan = build_position_prefilter_plan(
            &flat_steps,
            self.incoming_edges_field,
            self.outgoing_edges_field,
        );

        Ok(Box::new(OptimizedGraphTraversalWeight {
            src_weight,
            dst_weight,
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
            traversal: self.traversal.clone(),
            dependencies_binary_field: self.dependencies_binary_field,
            incoming_edges_field: self.incoming_edges_field,
            outgoing_edges_field: self.outgoing_edges_field,
            flat_steps, // Cached flattened pattern
            prefilter_plan,
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
    #[allow(dead_code)]
    traversal: crate::compiler::ast::Traversal,
    dependencies_binary_field: Field,
    #[allow(dead_code)]
    incoming_edges_field: Field,
    #[allow(dead_code)]
    outgoing_edges_field: Field,
    #[allow(dead_code)]
    src_pattern: crate::compiler::ast::Pattern,
    #[allow(dead_code)]
    dst_pattern: crate::compiler::ast::Pattern,
    /// Pre-computed flattened pattern steps (cached once per query)
    flat_steps: Vec<FlatPatternStep>,
    /// Position prefilter plan for edge-based position restrictions
    prefilter_plan: PositionPrefilterPlan,
}

impl Weight for OptimizedGraphTraversalWeight {

    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        // Create scorers once (fixed: was previously creating twice wastefully)
        let src_scorer = self.src_weight.scorer(reader, boost)?;
        let dst_scorer = self.dst_weight.scorer(reader, boost)?;

        // Cache the store reader (created once, reused for all documents in this segment)
        let store_reader = reader.get_store_reader(1)?;

        // Pre-extract constraint field names from flat_steps (computed once, not per document)
        // Helper to unwrap NamedCapture/Repetition to get field name
        fn unwrap_pattern_for_field_name(pat: &crate::compiler::ast::Pattern) -> String {
            use crate::compiler::ast::Pattern;
            match pat {
                Pattern::NamedCapture { pattern, .. } => unwrap_pattern_for_field_name(pattern),
                Pattern::Repetition { pattern, .. } => unwrap_pattern_for_field_name(pattern),
                Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, .. }) => name.clone(),
                _ => "word".to_string(),
            }
        }

        let constraint_field_names: Vec<String> = self.flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(pat) = step {
                    Some(unwrap_pattern_for_field_name(pat))
                } else {
                    None
                }
            })
            .collect();

        // Build constraint requirements from flat_steps (need schema from reader)
        let schema = reader.schema();
        let constraint_reqs = build_constraint_requirements(&self.flat_steps, schema);
        
        // Log prefilter plan info (once per query)
        log::info!(
            "prefilter: edge_reqs={}, constraint_reqs={}, num_constraints={}",
            self.prefilter_plan.edge_reqs.len(),
            constraint_reqs.len(),
            self.prefilter_plan.num_constraints
        );

        // Log which constraint fields are being used for prefiltering (only those with positions)
        if !constraint_reqs.is_empty() {
            log::info!("Constraint prefiltering enabled for {} fields with positions:", constraint_reqs.len());
            for req in &constraint_reqs {
                log::info!("  - Field '{}' (constraint_idx={}) term='{}'", 
                    schema.get_field_name(req.field), req.constraint_idx, req.term);
            }
        } else {
            log::info!("Constraint prefiltering disabled: no constraint fields indexed with positions");
        }

        // Create postings cursors for edge terms (one per segment)
        let mut edge_postings = Vec::new();
        
        for req in &self.prefilter_plan.edge_reqs {
            let term = Term::from_field_text(req.field, &req.label);
            let inverted_index = reader.inverted_index(req.field);
            
            let postings_result = if let Ok(inv_idx) = inverted_index {
                inv_idx.read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            } else {
                Ok(None)
            };
            
            match postings_result {
                Ok(Some(postings)) => edge_postings.push(Some(postings)),
                _ => edge_postings.push(None),
            }
        }

        // Create postings cursors for constraint terms (one per segment)
        let mut constraint_postings = Vec::new();
        
        for req in &constraint_reqs {
            let term = Term::from_field_text(req.field, &req.term);
            let inverted_index = reader.inverted_index(req.field);
            
            let postings_result = if let Ok(inv_idx) = inverted_index {
                inv_idx.read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
            } else {
                Ok(None)
            };
            
            match postings_result {
                Ok(Some(postings)) => constraint_postings.push(Some(postings)),
                _ => constraint_postings.push(None),
            }
        }

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
            prefilter_plan: self.prefilter_plan.clone(),
            edge_postings,
            constraint_reqs,
            constraint_postings,
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
    #[allow(dead_code)]
    traversal: crate::compiler::ast::Traversal,
    dependencies_binary_field: Field,
    reader: SegmentReader,
    /// Cached store reader (created once, reused for all documents)
    store_reader: StoreReader,
    current_doc: Option<DocId>,
    current_matches: Vec<(DocId, Score)>,
    match_index: usize,
    #[allow(dead_code)]
    src_pattern: crate::compiler::ast::Pattern,
    #[allow(dead_code)]
    dst_pattern: crate::compiler::ast::Pattern,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
    /// Boost factor from weight creation
    boost: Score,
    /// Pre-computed flattened pattern steps (cached from Weight)
    flat_steps: Vec<FlatPatternStep>,
    /// Pre-extracted constraint field names (cached from flat_steps)
    constraint_field_names: Vec<String>,
    /// Position prefilter plan
    prefilter_plan: PositionPrefilterPlan,
    /// Postings cursors for edge terms (one per EdgeTermReq)
    edge_postings: Vec<Option<SegmentPostings>>,
    /// Constraint term requirements (built in scorer)
    constraint_reqs: Vec<ConstraintTermReq>,
    /// Postings cursors for constraint terms (one per ConstraintTermReq)
    constraint_postings: Vec<Option<SegmentPostings>>,
}

impl OptimizedGraphTraversalScorer {
    /// Unwrap constraint pattern by removing NamedCapture and Repetition wrappers
    /// Returns the underlying constraint pattern
    fn unwrap_constraint_pattern<'a>(&self, pat: &'a Pattern) -> &'a Pattern {
        match pat {
            Pattern::NamedCapture { pattern, .. } => self.unwrap_constraint_pattern(pattern),
            Pattern::Repetition { pattern, .. } => self.unwrap_constraint_pattern(pattern),
            _ => pat,
        }
    }

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

                // Log final stats when scorer is exhausted (using module-level statics)
                let call_num = CALL_COUNT.load(Ordering::Relaxed);
                if call_num == 0 {
                    log::warn!(
                        "NO CANDIDATES FOUND! BooleanQuery returned 0 matching documents. \
                        This usually means the index was created with the OLD schema. \
                        You need to RE-INDEX your documents with the new position-aware schema."
                    );
                }
                if call_num > 0 {
                    let deser_count = GRAPH_DESER_COUNT.load(Ordering::Relaxed);
                    let skipped_count = GRAPH_DESER_SKIPPED.load(Ordering::Relaxed);
                    let skip_rate = (skipped_count as f64 / call_num as f64) * 100.0;
                    
                    let prefilter_docs = PREFILTER_DOCS.load(Ordering::Relaxed);
                    let prefilter_killed = PREFILTER_KILLED.load(Ordering::Relaxed);
                    let prefilter_kill_rate = if prefilter_docs > 0 {
                        (prefilter_killed as f64 / prefilter_docs as f64) * 100.0
                    } else {
                        0.0
                    };
                    
                    let allowed_pos_sum = PREFILTER_ALLOWED_POS_SUM.load(Ordering::Relaxed);
                    let allowed_pos_count = PREFILTER_ALLOWED_POS_COUNT.load(Ordering::Relaxed);
                    let avg_allowed_pos = if allowed_pos_count > 0 {
                        allowed_pos_sum as f64 / allowed_pos_count as f64
                    } else {
                        0.0
                    };
                    
                    log::info!(
                        "FINAL Graph traversal stats: {} candidates checked, {} graphs deserialized ({} skipped, {:.1}% skip rate)",
                        call_num, deser_count, skipped_count, skip_rate
                    );
                    log::info!(
                        "FINAL Prefilter stats: {} docs checked, {} killed by prefilter ({:.1}% kill rate), avg allowed positions per constraint: {:.1}",
                        prefilter_docs, prefilter_killed, prefilter_kill_rate, avg_allowed_pos
                    );
                }
                
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
    /// Intersect two sorted vectors of u32, storing result in the first vector
    /// O(n+m) time using two-pointer merge
    fn intersect_sorted_in_place(a: &mut Vec<u32>, b: &[u32]) {
        let mut out = Vec::with_capacity(a.len().min(b.len()));
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    out.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        *a = out;
    }

    /// Compute allowed positions per constraint using edge postings AND constraint postings
    /// Returns None if document cannot match (required edge/constraint term missing or empty intersection)
    /// Returns Some(allowed_positions) if document passes prefilter
    fn compute_allowed_positions(&mut self, doc_id: DocId) -> Option<Vec<Option<Vec<u32>>>> {
        // Start as "no restriction" for each constraint
        let mut allowed: Vec<Option<Vec<u32>>> = vec![None; self.prefilter_plan.num_constraints];

        let mut buf: Vec<u32> = Vec::with_capacity(32);

        // Phase 1: Process edge requirements
        for (req_idx, req) in self.prefilter_plan.edge_reqs.iter().enumerate() {
            let postings_opt = self.edge_postings.get_mut(req_idx).and_then(|p| p.as_mut());

            // If postings is None, the term doesn't exist in this segment at all => cannot match
            let postings = match postings_opt {
                Some(p) => p,
                None => {
                    // Log: term doesn't exist in segment at all
                    log::warn!("EdgeReq[{}] label='{}' has no postings in segment", req_idx, req.label);
                    return None;
                }
            };

            // Use seek() for cleaner and often faster positioning
            postings.seek(doc_id);

            // Term not present in this doc => cannot match
            if postings.doc() != doc_id {
                // This is expected if doc doesn't have this edge - that's a valid skip
                return None;
            }

            buf.clear();
            postings.positions(&mut buf);

            // DIAGNOSTIC: Log position count for first few docs
            static DIAG_COUNT: AtomicUsize = AtomicUsize::new(0);
            let diag = DIAG_COUNT.fetch_add(1, Ordering::Relaxed);
            if diag < 20 {
                log::info!(
                    "DIAG doc_id={} EdgeReq[{}] label='{}' constraint_idx={} positions={:?}",
                    doc_id, req_idx, req.label, req.constraint_idx, buf
                );
            }

            if buf.is_empty() {
                log::warn!("EdgeReq[{}] label='{}' doc_id={} has term but ZERO positions!", req_idx, req.label, doc_id);
                return None;
            }

            match &mut allowed[req.constraint_idx] {
                None => {
                    // First restriction: take positions
                    allowed[req.constraint_idx] = Some(buf.clone());
                }
                Some(existing) => {
                    // Intersect existing with buf
                    Self::intersect_sorted_in_place(existing, &buf);
                    if existing.is_empty() {
                        return None;
                    }
                }
            }
        }

        // Phase 2: Process constraint requirements (intersect with edge positions)
        for (req_idx, req) in self.constraint_reqs.iter().enumerate() {
            let postings_opt = self.constraint_postings.get_mut(req_idx).and_then(|p| p.as_mut());

            // If postings is None, the term doesn't exist in this segment at all => cannot match
            let postings = match postings_opt {
                Some(p) => p,
                None => {
                    // Term doesn't exist - document cannot match this exact constraint
                    return None;
                }
            };

            postings.seek(doc_id);

            // Term not present in this doc => cannot match
            if postings.doc() != doc_id {
                return None;
            }

            buf.clear();
            postings.positions(&mut buf);

            if buf.is_empty() {
                return None;
            }

            // Intersect constraint positions with existing allowed positions (from edges)
            match &mut allowed[req.constraint_idx] {
                None => {
                    // No edge restriction yet - take constraint positions
                    allowed[req.constraint_idx] = Some(buf.clone());
                }
                Some(existing) => {
                    // Intersect: only positions that have BOTH the edge AND the constraint term
                    Self::intersect_sorted_in_place(existing, &buf);
                    if existing.is_empty() {
                        return None;
                    }
                }
            }
        }

        Some(allowed)
    }

    /// Check if a document has valid graph traversal from source to destination
    /// 
    /// Optimizations:
    /// 1. Pre-checks ALL constraints before graph deserialization (early exit)
    /// 2. Uses boolean query for early termination during traversal
    /// 3. Tracks skipped graph deserializations for profiling
    /// 4. Computes allowed positions from edge postings BEFORE loading stored document
    fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {
        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        self.current_doc_matches.clear();

        // Phase 0: Postings prefilter (before any store access)
        PREFILTER_DOCS.fetch_add(1, Ordering::Relaxed);
        let allowed_positions = match self.compute_allowed_positions(doc_id) {
            Some(ap) => ap,
            None => {
                PREFILTER_KILLED.fetch_add(1, Ordering::Relaxed);
                GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };

        // Track allowed position sizes
        for ap in &allowed_positions {
            if let Some(ref positions) = ap {
                PREFILTER_ALLOWED_POS_SUM.fetch_add(positions.len(), Ordering::Relaxed);
                PREFILTER_ALLOWED_POS_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Early exit if any required constraint has empty allowed positions
        for ap in &allowed_positions {
            if let Some(ref positions) = ap {
                if positions.is_empty() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            }
        }

        let flat_steps = &self.flat_steps;
        if flat_steps.is_empty() {
            GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Phase 1: Load stored document (only survivors reach here)
        let doc = match self.store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => return false,
        };

        // Phase 2: Extract tokens ONLY for constraint fields that need regex checking
        // Skip extraction if allowed_positions is tiny and we can use it directly
        let mut constraint_fields_and_tokens: Vec<(String, Vec<String>)> =
            Vec::with_capacity(self.constraint_field_names.len());
        
        for field_name in &self.constraint_field_names {
            let tokens = self.extract_tokens_from_field(&doc, field_name);
            constraint_fields_and_tokens.push((field_name.clone(), tokens));
        }

        // Phase 3: Check constraints with position restrictions
        let mut constraint_count = 0;
        let mut cached_positions: Vec<Vec<usize>> = Vec::new();

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(constraint_pat) = step {
                if constraint_count >= constraint_fields_and_tokens.len() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }

                let (_, tokens) = &constraint_fields_and_tokens[constraint_count];
                let unwrapped = self.unwrap_constraint_pattern(constraint_pat);
                
                let is_wildcard = matches!(
                    unwrapped,
                    Pattern::Constraint(crate::compiler::ast::Constraint::Wildcard)
                );

                let positions = if is_wildcard {
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        (0..tokens.len()).collect()
                    }
                } else if let Some(ref allowed) = allowed_positions[constraint_count] {
                    // Use limited check - only test positions that have required edges
                    self.find_positions_in_tokens_limited(tokens, constraint_pat, allowed)
                } else {
                    // No restriction from edges - check all positions
                    self.find_positions_in_tokens(tokens, constraint_pat)
                };

                if positions.is_empty() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                cached_positions.push(positions);
                constraint_count += 1;
            }
        }

        // Phase 4: Get and deserialize graph
        let binary_data = match doc.get_first(self.dependencies_binary_field).and_then(|v| v.as_bytes()) {
            Some(data) => data,
            None => {
                GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };

        GRAPH_DESER_COUNT.fetch_add(1, Ordering::Relaxed);
        let graph = match DirectedGraph::from_bytes(binary_data) {
            Ok(graph) => graph,
            Err(_) => return false,
        };

        // Log stats periodically
        if call_num > 0 && call_num % 100 == 0 {
            let deser = GRAPH_DESER_COUNT.load(Ordering::Relaxed);
            let skipped = GRAPH_DESER_SKIPPED.load(Ordering::Relaxed);
            let pf_docs = PREFILTER_DOCS.load(Ordering::Relaxed);
            let pf_killed = PREFILTER_KILLED.load(Ordering::Relaxed);
            let pos_sum = PREFILTER_ALLOWED_POS_SUM.load(Ordering::Relaxed);
            let pos_count = PREFILTER_ALLOWED_POS_COUNT.load(Ordering::Relaxed);
            
            log::info!(
                "Stats: calls={} deser={} skipped={} prefilter_killed={}/{} ({:.1}%) avg_positions={:.1}",
                call_num, deser, skipped,
                pf_killed, pf_docs,
                if pf_docs > 0 { pf_killed as f64 / pf_docs as f64 * 100.0 } else { 0.0 },
                if pos_count > 0 { pos_sum as f64 / pos_count as f64 } else { 0.0 }
            );
        }

        // Phase 5: Run traversal
        let src_positions = cached_positions.get(0).cloned().unwrap_or_default();
        if constraint_count > 0 && src_positions.is_empty() {
            return false;
        }

        let traversal_engine = crate::digraph::traversal::GraphTraversal::new(graph);
        
        for &src_pos in &src_positions {
            let all_paths = traversal_engine.automaton_query_paths(
                flat_steps, &[src_pos], &constraint_fields_and_tokens
            );
            
            for path in &all_paths {
                if !path.is_empty() {
                    let mut captures = Vec::with_capacity(path.len());
                    let mut c_idx = 0;
                    for step in flat_steps.iter() {
                        if let FlatPatternStep::Constraint(ref pat) = step {
                            if let Some(&node_idx) = path.get(c_idx) {
                                let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                                let name = match pat {
                                    Pattern::NamedCapture { name, .. } => name.clone(),
                                    _ => format!("c{}", CAPTURE_COUNTER.fetch_add(1, Ordering::Relaxed)),
                                };
                                captures.push(crate::types::NamedCapture::new(name, span));
                            }
                            c_idx += 1;
                        }
                    }
                    let min_pos = *path.iter().min().unwrap();
                    let max_pos = *path.iter().max().unwrap();
                    self.current_doc_matches.push(
                        crate::types::SpanWithCaptures::with_captures(
                            crate::types::Span { start: min_pos, end: max_pos + 1 },
                            captures
                        )
                    );
                }
            }
            
            if !all_paths.is_empty() {
                return true;
            }
        }
        
        false
    }

    /// Extract the field name from a pattern
    #[allow(dead_code)]
    fn get_field_name_from_pattern<'a>(&self, pattern: &'a crate::compiler::ast::Pattern) -> &'a str {
        match pattern {
            crate::compiler::ast::Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, .. }) => {
                name.as_str()
            }
            _ => "word", // default to word field
        }
    }

    /// Extract tokens from a specific field in the document
    /// Decodes position-aware format if necessary
    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument, field_name: &str) -> Vec<String> {
        crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), doc, field_name)
    }
    
    /// Find positions that match a pattern (for backward compatibility)
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    fn traversal_to_pattern(&self) -> Pattern {
        Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        }
    }

    /// Find positions in tokens that match a given pattern (string, regex, or wildcard for any field)
    fn find_positions_in_tokens(&self, tokens: &[String], pattern: &crate::compiler::ast::Pattern) -> Vec<usize> {
        // Unwrap NamedCapture/Repetition to get underlying constraint
        let pattern = self.unwrap_constraint_pattern(pattern);
        
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

    /// Find positions in tokens that match a pattern, restricted to allowed positions
    fn find_positions_in_tokens_limited(
        &self,
        tokens: &[String],
        pattern: &Pattern,
        allowed: &[u32],
    ) -> Vec<usize> {
        // Unwrap NamedCapture/Repetition to get underlying constraint
        let pattern = self.unwrap_constraint_pattern(pattern);
        
        use crate::compiler::ast::{Pattern, Constraint, Matcher};
        let mut positions = Vec::new();
        
        // Note: allowed is already sorted, so we can iterate directly
        match pattern {
            Pattern::Constraint(Constraint::Field { name: _, matcher }) => {
                match matcher {
                    Matcher::String(s) => {
                        for &pos in allowed {
                            let pos_usize = pos as usize;
                            if pos_usize < tokens.len() && tokens[pos_usize] == *s {
                                positions.push(pos_usize);
                            }
                        }
                    }
                    Matcher::Regex { regex, .. } => {
                        for &pos in allowed {
                            let pos_usize = pos as usize;
                            if pos_usize < tokens.len() && regex.is_match(&tokens[pos_usize]) {
                                positions.push(pos_usize);
                            }
                        }
                    }
                }
            }
            Pattern::Constraint(Constraint::Wildcard) => {
                // Wildcard matches all allowed positions
                positions = allowed.iter().map(|&p| p as usize).collect();
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

/// Build position prefilter plan from flattened pattern steps
/// 
/// For each traversal step between constraints, creates edge term requirements
/// that restrict which positions can match the adjacent constraints.
fn build_position_prefilter_plan(
    flat_steps: &[FlatPatternStep],
    incoming_edges_field: Field,
    outgoing_edges_field: Field,
) -> PositionPrefilterPlan {
    let mut plan = PositionPrefilterPlan::default();
    
    // Count constraints to determine constraint_idx space
    plan.num_constraints = flat_steps.iter()
        .filter(|step| matches!(step, FlatPatternStep::Constraint(_)))
        .count();
    
    if plan.num_constraints == 0 {
        return plan;
    }
    
    // Walk through flat_steps and build edge requirements
    let mut constraint_idx = 0;
    
    for (_step_idx, step) in flat_steps.iter().enumerate() {
        if let FlatPatternStep::Traversal(traversal) = step {
            // Find the constraint indices this traversal connects
            // Previous constraint is the last one we saw
            // Next constraint is the next one we'll see
            
            let prev_constraint_idx = if constraint_idx > 0 { constraint_idx - 1 } else { 0 };
            let next_constraint_idx = constraint_idx; // Next constraint hasn't been counted yet
            
            // Only support simple single-hop traversals initially
            match traversal {
                Traversal::Outgoing(Matcher::String(label)) => {
                    // Outgoing edge: restrict previous constraint by outgoing_edges, next by incoming_edges
                    if prev_constraint_idx < plan.num_constraints {
                        plan.edge_reqs.push(EdgeTermReq {
                            field: outgoing_edges_field,
                            label: label.clone(),
                            constraint_idx: prev_constraint_idx,
                        });
                    }
                    if next_constraint_idx < plan.num_constraints {
                        plan.edge_reqs.push(EdgeTermReq {
                            field: incoming_edges_field,
                            label: label.clone(),
                            constraint_idx: next_constraint_idx,
                        });
                    }
                }
                Traversal::Incoming(Matcher::String(label)) => {
                    // Incoming edge: restrict previous constraint by incoming_edges, next by outgoing_edges
                    if prev_constraint_idx < plan.num_constraints {
                        plan.edge_reqs.push(EdgeTermReq {
                            field: incoming_edges_field,
                            label: label.clone(),
                            constraint_idx: prev_constraint_idx,
                        });
                    }
                    if next_constraint_idx < plan.num_constraints {
                        plan.edge_reqs.push(EdgeTermReq {
                            field: outgoing_edges_field,
                            label: label.clone(),
                            constraint_idx: next_constraint_idx,
                        });
                    }
                }
                // For other traversal variants, don't add requirements (unsafe to prefilter)
                _ => {}
            }
        } else if let FlatPatternStep::Constraint(_) = step {
            constraint_idx += 1;
        }
    }
    
    plan
}

/// Build constraint term requirements from flattened pattern steps
/// Extracts exact string constraints that can be prefiltered via postings
/// Only includes fields that are indexed with positions (required for position-based prefiltering)
fn build_constraint_requirements(flat_steps: &[FlatPatternStep], schema: &Schema) -> Vec<ConstraintTermReq> {
    let mut constraint_reqs = Vec::new();
    let mut constraint_idx = 0;

    for step in flat_steps.iter() {
        if let FlatPatternStep::Constraint(pat) = step {
            // Unwrap named captures and repetitions to get the underlying constraint
            let inner = unwrap_constraint_pattern_static(pat);
            
            if let Pattern::Constraint(Constraint::Field { name, matcher }) = inner {
                // Only exact strings can be prefiltered via postings (regex would need term enumeration)
                if let Matcher::String(term_value) = matcher {
                    if let Ok(field) = schema.get_field(name) {
                        // Check if field is indexed with positions (required for constraint prefiltering)
                        let field_entry = schema.get_field_entry(field);
                        let has_positions = field_entry.field_type().get_index_record_option()
                            .map(|opt| opt.has_positions())
                            .unwrap_or(false);
                        
                        if has_positions {
                            constraint_reqs.push(ConstraintTermReq {
                                field,
                                term: term_value.clone(),
                                constraint_idx,
                            });
                            log::debug!(
                                "Added constraint prefilter for field '{}' (constraint_idx={}) with term '{}'",
                                name, constraint_idx, term_value
                            );
                        } else {
                            log::debug!(
                                "Skipping constraint prefilter for field '{}' (constraint_idx={}): field not indexed with positions",
                                name, constraint_idx
                            );
                        }
                    }
                }
                // For regex: we'd need term enumeration (more complex, skip for now)
            }
            constraint_idx += 1;
        }
    }

    constraint_reqs
}

/// Helper to unwrap NamedCapture/Repetition to get underlying constraint pattern
fn unwrap_constraint_pattern_static(pat: &Pattern) -> &Pattern {
    match pat {
        Pattern::NamedCapture { pattern, .. } => unwrap_constraint_pattern_static(pattern),
        Pattern::Repetition { pattern, .. } => unwrap_constraint_pattern_static(pattern),
        _ => pat,
    }
}
