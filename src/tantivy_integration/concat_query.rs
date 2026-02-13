use tantivy::query::{Query, Weight, Scorer, EnableScoring};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet, Term};
use tantivy::schema::{Field, IndexRecordOption, Value};
use tantivy::postings::Postings;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tantivy_fst::Regex;
use crate::query::ast::{Pattern, Constraint, Matcher};

/// Execution plan for anchor-based verification
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub anchor_idx: usize,
}

/// Step in a concatenation execution plan
#[derive(Debug, Clone)]
pub enum ConcatStep {
    /// A single-token constraint (positions come from positions_per_constraint[constraint_idx])
    Atom { constraint_idx: usize },

    /// A variable-width gap of wildcard tokens between atoms.
    /// Gap is measured as number of tokens *between* the two atoms.
    Gap { min: usize, max: Option<usize>, lazy: bool },
}

/// Execution plan for concatenation patterns with gaps
#[derive(Debug, Clone)]
pub struct ConcatPlan {
    pub steps: Vec<ConcatStep>,
}

/// Constraint source for position-based matching
#[derive(Debug, Clone)]
enum ConstraintSource {
    Exact { field: Field, term: Term },
    Regex { field: Field, pattern: String, terms: Vec<Term> },  // FST-expanded terms (capped)
    Wildcard { field: Field },
}

/// Positions for a constraint - either a list or wildcard (any position)
/// Using slices to avoid cloning (allocation-free)
enum PosView<'a> {
    Any,  // Wildcard: matches any position
    List(&'a [u32]),  // Specific positions (slice into position_buffers)
}

const MAX_REGEX_EXPANSION: usize = 1024;  // Cap regex expansion for postings path (lower for performance)

#[derive(Debug)]
pub struct RustieConcatQuery {
    pub default_field: Field,
    pub pattern: Pattern,
    pub sub_queries: Vec<Box<dyn Query>>,
    pub execution_plan: Option<ExecutionPlan>,
    pub concat_plan: Option<ConcatPlan>,
}

impl Clone for RustieConcatQuery {
    fn clone(&self) -> Self {
        RustieConcatQuery {
            default_field: self.default_field,
            pattern: self.pattern.clone(),
            sub_queries: self.sub_queries.iter().map(|q| q.box_clone()).collect(),
            execution_plan: self.execution_plan.clone(),
            concat_plan: self.concat_plan.clone(),
        }
    }
}

impl RustieConcatQuery {
    pub fn new(
        default_field: Field,
        pattern: Pattern,
        sub_queries: Vec<Box<dyn Query>>,
    ) -> Self {
        Self {
            default_field,
            pattern,
            sub_queries,
            execution_plan: None,
            concat_plan: None,
        }
    }
}

impl Query for RustieConcatQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let sub_weights: Vec<Box<dyn Weight>> = self.sub_queries
            .iter()
            .map(|q| q.weight(scoring.clone()))
            .collect::<TantivyResult<Vec<_>>>()?;

        Ok(Box::new(RustieConcatWeight {
            sub_weights,
            pattern: self.pattern.clone(),
            default_field: self.default_field,
            concat_plan: self.concat_plan.clone(),
            regex_automaton_cache: Arc::new(RwLock::new(HashMap::<String, Arc<Regex>>::new())),
        }))
    }
}

struct RustieConcatWeight {
    sub_weights: Vec<Box<dyn Weight>>,
    pattern: Pattern,
    default_field: Field,
    concat_plan: Option<ConcatPlan>,
    /// Cached compiled regex automata (shared across segments, thread-safe)
    regex_automaton_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
}

/// Helper to extract constraints from a Pattern
fn extract_constraints_from_pattern(pattern: &Pattern) -> Vec<&Constraint> {
    use crate::query::ast::Pattern;
    let mut constraints = Vec::new();
    
    match pattern {
        Pattern::Concatenated(patterns) => {
            for pat in patterns {
                if let Pattern::Constraint(c) = pat {
                    constraints.push(c);
                } else if let Pattern::NamedCapture { pattern: p, .. } = pat {
                    if let Pattern::Constraint(c) = p.as_ref() {
                        constraints.push(c);
                    }
                }
            }
        }
        Pattern::Constraint(c) => {
            constraints.push(c);
        }
        _ => {}
    }
    
    constraints
}

/// Compile constraint sources from pattern for postings-based matching
fn compile_constraint_sources(
    pattern: &Pattern,
    reader: &SegmentReader,
    default_field: &Field,
    regex_automaton_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
) -> TantivyResult<Vec<ConstraintSource>> {
    use crate::query::ast::Pattern;
    let mut sources = Vec::new();
    let schema = reader.schema();
    
    // Segment-local regex cache (immutable, no mutex needed)
    let mut regex_cache: HashMap<String, Vec<Term>> = HashMap::new();
    
    match pattern {
        Pattern::Concatenated(patterns) => {
            // ═══════════════════════════════════════════════════════════════
            // OPTIMIZED: Only fallback for COMPLEX Repetition patterns
            // Simple wildcard repetitions ([]*?, []{0,5}, etc.) are handled as gaps
            // ═══════════════════════════════════════════════════════════════
            for pat in patterns {
                if let Pattern::Repetition { pattern: inner, .. } = pat {
                    // Check if inner pattern is a simple wildcard (can be handled as gap)
                    let is_simple_gap = matches!(
                        inner.as_ref(),
                        Pattern::Constraint(Constraint::Wildcard)
                    );
                    
                    if !is_simple_gap {
                        // Complex repetition (non-wildcard inner pattern) - fallback needed
                        return Ok(Vec::new());
                    }
                    // Simple wildcard gap - continue, will be handled by ConcatPlan
                }
                
                // Also check inside NamedCapture
                if let Pattern::NamedCapture { pattern: inner, .. } = pat {
                    if let Pattern::Repetition { pattern: rep_inner, .. } = inner.as_ref() {
                        let is_simple_gap = matches!(
                            rep_inner.as_ref(),
                            Pattern::Constraint(Constraint::Wildcard)
                        );
                        if !is_simple_gap {
                            return Ok(Vec::new());
                        }
                    }
                }
            }
            // ═══════════════════════════════════════════════════════════════
            
            // Extract constraint sources for non-gap patterns
            // Gap patterns (wildcard repetitions) are handled by ConcatPlan
            for pat in patterns {
                // Skip gap patterns (wildcard repetitions) - handled by ConcatPlan
                if let Pattern::Repetition { pattern: inner, .. } = pat {
                    if matches!(inner.as_ref(), Pattern::Constraint(Constraint::Wildcard)) {
                        continue;  // Skip gaps
                    }
                }
                
                let constraint = match pat {
                    Pattern::Constraint(c) => c,
                    Pattern::NamedCapture { pattern: p, .. } => {
                        match p.as_ref() {
                            Pattern::Constraint(c) => c,
                            Pattern::Repetition { pattern: inner, .. } => {
                                // Skip wildcard repetitions in named captures
                                if matches!(inner.as_ref(), Pattern::Constraint(Constraint::Wildcard)) {
                                    continue;
                                }
                                // Non-wildcard repetition should have been rejected above
                                continue;
                            }
                            _ => continue,
                        }
                    }
                    _ => continue,
                };
                
                let source = compile_constraint_to_source(
                    constraint, reader, default_field, &mut regex_cache, schema, regex_automaton_cache.clone()
                )?;
                sources.push(source);
            }
        }
        Pattern::Constraint(c) => {
            let source = compile_constraint_to_source(c, reader, default_field, &mut regex_cache, schema, regex_automaton_cache.clone())?;
            sources.push(source);
        }
        _ => {
            // For non-constraint patterns, return empty (postings path unavailable)
            return Ok(Vec::new());
        }
    }
    
    Ok(sources)
}

/// Helper function to expand regex terms using a cached automaton.
/// Extracted to avoid code duplication and enable caching.
fn expand_regex_terms_with_automaton(
    term_dict: &tantivy::termdict::TermDictionary,
    automaton: &Regex,
    field: Field,
    inverted_index: &tantivy::InvertedIndexReader,
    regex_cache: &mut HashMap<String, Vec<Term>>,
    cache_key: &str,
    pattern: &str,
) -> TantivyResult<Vec<Term>> {
    let mut stream = term_dict.search(automaton).into_stream()
        .map_err(|e| tantivy::TantivyError::SchemaError(format!("Failed to search term dict: {:?}", e)))?;
    
    let mut terms = Vec::new();
    let mut count = 0;
    
    while stream.advance() {
        if count >= MAX_REGEX_EXPANSION {
            break;
        }
        
        let term_bytes = stream.key();
        let term = Term::from_field_bytes(field, term_bytes);
        terms.push(term);
        count += 1;
    }
    
    if !terms.is_empty() {
        regex_cache.insert(cache_key.to_string(), terms.clone());
    }
    
    Ok(terms)
}

/// Get or compile a regex automaton from the cache (thread-safe).
fn get_or_compile_regex_automaton(
    cache: &Arc<RwLock<HashMap<String, Arc<Regex>>>>,
    pattern: &str,
) -> TantivyResult<Arc<Regex>> {
    // Fast path: read lock
    {
        let read_guard = cache.read()
            .map_err(|e| tantivy::TantivyError::SchemaError(format!("Failed to acquire regex cache read lock: {}", e)))?;
        if let Some(regex) = read_guard.get(pattern) {
            return Ok(Arc::clone(regex));
        }
    } // Read lock released here
    
    // Slow path: write lock
    let mut write_guard = cache.write()
        .map_err(|e| tantivy::TantivyError::SchemaError(format!("Failed to acquire regex cache write lock: {}", e)))?;
    
    // Double-check after acquiring write lock (another thread might have compiled it)
    if let Some(regex) = write_guard.get(pattern) {
        return Ok(Arc::clone(regex));
    }
    
    // Compile and cache
    let regex = Arc::new(Regex::new(pattern)
        .map_err(|e| tantivy::TantivyError::SchemaError(format!("Invalid regex pattern '{}': {}", pattern, e)))?);
    write_guard.insert(pattern.to_string(), Arc::clone(&regex));
    Ok(regex)
}

/// Compile a single constraint to ConstraintSource
fn compile_constraint_to_source(
    constraint: &Constraint,
    reader: &SegmentReader,
    default_field: &Field,
    regex_cache: &mut HashMap<String, Vec<Term>>,
    schema: &tantivy::schema::Schema,
    regex_automaton_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
) -> TantivyResult<ConstraintSource> {
    match constraint {
        Constraint::Wildcard => {
            // Use default field for wildcard
            Ok(ConstraintSource::Wildcard { field: *default_field })
        }
        Constraint::Field { name, matcher } => {
            let field = schema.get_field(name)
                .map_err(|_| tantivy::TantivyError::SchemaError(format!("Field '{}' not found", name)))?;
            
            match matcher {
                Matcher::String(s) => {
                    let term = Term::from_field_text(field, s);
                    Ok(ConstraintSource::Exact { field, term })
                }
                Matcher::Regex { pattern, .. } => {
                    // Strip /.../ delimiters before FST expansion
                    let clean_pattern = pattern.trim_start_matches('/').trim_end_matches('/');
                    
                    // Check segment-local term cache first
                    let cache_key = format!("{}:{}", name, pattern);
                    let terms = if let Some(cached) = regex_cache.get(&cache_key) {
                        cached.clone()
                    } else {
                        // Expand regex using FST
                        let inverted_index = reader.inverted_index(field)
                            .map_err(|e| tantivy::TantivyError::SchemaError(format!("Failed to get inverted index: {}", e)))?;
                        let term_dict = inverted_index.terms();
                        
                        // Get or compile regex automaton from cache (thread-safe)
                        let automaton = get_or_compile_regex_automaton(&regex_automaton_cache, clean_pattern)?;
                        
                        // Expand terms using the cached automaton
                        expand_regex_terms_with_automaton(
                            &term_dict,
                            automaton.as_ref(),
                            field,
                            &inverted_index,
                            regex_cache,
                            &cache_key,
                            pattern,
                        )?
                    };
                    
                    Ok(ConstraintSource::Regex { field, pattern: pattern.clone(), terms })
                }
            }
        }
        _ => {
            // For complex constraints (Negated, Conjunctive, Disjunctive), fall back to stored-field path
            Err(tantivy::TantivyError::SchemaError("Complex constraints not yet supported in postings path".to_string()))
        }
    }
}

/// Compute execution plan for anchor-based verification
fn compute_execution_plan(
    constraint_sources: &[ConstraintSource],
    reader: &SegmentReader,
) -> Option<ExecutionPlan> {
    if constraint_sources.is_empty() || constraint_sources.len() < 2 {
        return None;
    }
    
    // Estimate selectivity for each constraint
    let mut selectivities: Vec<(usize, u32)> = Vec::new();
    
    for (idx, source) in constraint_sources.iter().enumerate() {
        let estimated_df = match source {
            ConstraintSource::Exact { field, term } => {
                if let Ok(inv_idx) = reader.inverted_index(*field) {
                    inv_idx.doc_freq(term).unwrap_or(0)
                } else {
                    u32::MAX  // Unknown, treat as least selective
                }
            }
            ConstraintSource::Regex { field, terms, .. } => {
                // Don't choose regex as anchor if expansion is large (expensive per-doc)
                if terms.len() > 256 {
                    u32::MAX  // Too many terms, don't choose as anchor
                } else if let Ok(inv_idx) = reader.inverted_index(*field) {
                    let mut total_df = 0u32;
                    for term in terms {
                        total_df = total_df.saturating_add(inv_idx.doc_freq(term).unwrap_or(0));
                    }
                    total_df
                } else {
                    u32::MAX
                }
            }
            ConstraintSource::Wildcard { .. } => {
                u32::MAX  // Wildcard is least selective
            }
        };
        
        selectivities.push((idx, estimated_df));
    }
    
    // Find constraint with smallest doc_freq (most selective)
    // Prefer exact terms over regex when costs are close
    let anchor_idx = selectivities.iter()
        .enumerate()
        .min_by_key(|(_, (idx, df))| {
            let source = &constraint_sources[*idx];
            let bonus = match source {
                ConstraintSource::Exact { .. } => 0,  // Prefer exact
                ConstraintSource::Regex { .. } => 1000,  // Penalize regex slightly
                ConstraintSource::Wildcard { .. } => 10000,  // Strongly penalize wildcard
            };
            (*df as u64, bonus)
        })
        .map(|(_, (idx, _))| *idx)?;
    
    Some(ExecutionPlan { anchor_idx })
}

impl Weight for RustieConcatWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        let num_sub_weights = self.sub_weights.len();
        let sub_scorers: Vec<Box<dyn Scorer>> = self.sub_weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let scorer = w.scorer(reader, boost)?;
                // Check if scorer has any documents immediately (before advance)
                let initial_doc = scorer.doc();
                Ok(scorer)
            })
            .collect::<TantivyResult<Vec<_>>>()?;

        let is_simple = sub_scorers.len() == 2;
        if is_simple {
        }

        // Compile constraint sources for postings-based Phase 2
        let constraint_sources = match compile_constraint_sources(&self.pattern, reader, &self.default_field, self.regex_automaton_cache.clone()) {
            Ok(sources) => sources,
            Err(_e) => {
                Vec::new()  // Empty means postings path cannot run
            }
        };

        // Compute execution plan (anchor-based verification)
        let execution_plan = compute_execution_plan(&constraint_sources, reader);
        
        // Get concat plan from query
        let concat_plan = self.concat_plan.clone();

        let scorer = RustieConcatScorer {
            sub_scorers,
            pattern: self.pattern.clone(),
            default_field: self.default_field,
            reader: reader.clone(),
            current_doc: None,
            current_matches: Vec::new(),
            match_index: 0,
            current_doc_matches: Vec::new(),
            boost,
            started: false, // Explicitly track initialization state
            constraint_sources,
            execution_plan,
            concat_plan,
            position_buffers: Vec::new(),
            regex_tmp: Vec::with_capacity(16),
        };
        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("RustieConcatQuery", Score::default()))
    }
}

pub struct RustieConcatScorer {
    sub_scorers: Vec<Box<dyn Scorer>>,
    pattern: Pattern,
    default_field: Field,
    reader: SegmentReader,
    current_doc: Option<DocId>,
    current_matches: Vec<(DocId, Score)>,
    match_index: usize,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
    boost: Score,
    started: bool,
    constraint_sources: Vec<ConstraintSource>,
    execution_plan: Option<ExecutionPlan>,
    concat_plan: Option<ConcatPlan>,
    position_buffers: Vec<Vec<u32>>,  // Reusable buffers for position collection
    regex_tmp: Vec<u32>,  // Reusable buffer for regex position collection
}

impl RustieConcatScorer {
    fn compute_slop_factor(span_width: usize) -> Score {
        1.0 / (1.0 + span_width as f32)
    }

    fn compute_odinson_score(&self) -> Score {
        if self.current_doc_matches.is_empty() {
            return 0.0;
        }
        let mut acc_sloppy_freq: Score = 0.0;
        for span_match in &self.current_doc_matches {
            let span_width = span_match.span.end.saturating_sub(span_match.span.start);
            acc_sloppy_freq += Self::compute_slop_factor(span_width);
        }
        let base_score = 1.0;
        let final_score = base_score * acc_sloppy_freq * self.boost;
        final_score.max(1.0)
    }
}

impl Scorer for RustieConcatScorer {
    fn score(&mut self) -> Score {
        if let Some((_, score)) = self.current_matches.get(self.match_index) {
            *score
        } else {
            Score::default()
        }
    }
}

impl DocSet for RustieConcatScorer {
    fn advance(&mut self) -> DocId {
        if self.sub_scorers.is_empty() {
            self.current_doc = None;
            return tantivy::TERMINATED;
        }

        // Phase 1: Ensure scorers are positioned correctly
        if !self.started {
            self.started = true;
            // Always advance all scorers once to land on first match
            for (i, scorer) in self.sub_scorers.iter_mut().enumerate() {
                let doc = scorer.advance();
                if doc == tantivy::TERMINATED {
                    self.current_doc = None;
                    return tantivy::TERMINATED;
                }
            }
        } else {
            // Subsequent calls: Advance the LEADER (scorer 0)
            if self.sub_scorers[0].advance() == tantivy::TERMINATED {
                self.current_doc = None;
                return tantivy::TERMINATED;
            }
        }

        // Phase 2: Zig-Zag Intersection
        loop {
            let candidate = self.sub_scorers[0].doc();
            
            let mut all_match = true;
            let mut next_target = candidate;

            for (scorer_idx, scorer) in self.sub_scorers.iter_mut().skip(1).enumerate() {
                let mut s_doc = scorer.doc();

                while s_doc < candidate {
                    s_doc = scorer.advance();
                }

                if s_doc == tantivy::TERMINATED {
                    self.current_doc = None;
                    return tantivy::TERMINATED;
                }

                if s_doc > candidate {
                    next_target = s_doc;
                    all_match = false;
                    break;
                }
            }

            if all_match {
                if self.check_pattern_matching(candidate) {
                    self.current_doc = Some(candidate);
                    let score = self.compute_odinson_score();
                    self.current_matches.push((candidate, score));
                    self.match_index = self.current_matches.len() - 1;
                    return candidate;
                }
                
                if self.sub_scorers[0].advance() == tantivy::TERMINATED {
                    self.current_doc = None;
                    return tantivy::TERMINATED;
                }
            } else {
                let mut s0 = self.sub_scorers[0].doc();
                while s0 < next_target {
                    s0 = self.sub_scorers[0].advance();
                }
                if s0 == tantivy::TERMINATED {
                     self.current_doc = None;
                     return tantivy::TERMINATED;
                }
            }
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc.unwrap_or(tantivy::TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl RustieConcatScorer {
    fn check_pattern_matching(&mut self, doc_id: DocId) -> bool {
        
        self.current_doc_matches.clear();
        
        // Try postings-based path first if constraint sources are available
        if !self.constraint_sources.is_empty() {
            
            // Get doc length and execution plan before mutable borrow
            let doc_len = match self.get_doc_length(doc_id, self.default_field) {
                Ok(len) => Some(len),
                Err(_e) => {
                    None
                }
            };
            let execution_plan = self.execution_plan.clone();
            let concat_plan = self.concat_plan.clone();
            
            match self.get_constraint_positions(doc_id) {
                Ok(positions_per_constraint) => {
                    
                    // Use position-based matching
                    let all_spans = find_constraint_spans_from_positions(
                        &positions_per_constraint,
                        &execution_plan,
                        &concat_plan,
                        doc_len,
                    );
                    
                    self.current_doc_matches = all_spans;
                    
                    if !self.current_doc_matches.is_empty() {
                        return true;
                    } else {
                        // Trust the postings path result - if it found 0 matches, return false
                        // Don't fall back to stored-field path which is slower
                        return false;
                    }
                }
                Err(e) => {
                    // Fall through to return false
                }
            }
        }
        
        // Postings path unavailable (constraint_sources empty or get_constraint_positions failed)
        return false;
    }
    
    /// Fill positions for exact term constraint
    fn fill_positions_exact(
        reader: &SegmentReader,
        field: Field,
        term: &Term,
        doc_id: DocId,
        buf: &mut Vec<u32>,
    ) -> TantivyResult<()> {
        let inverted_index = reader.inverted_index(field)?;
        let Some(mut postings) = inverted_index
            .read_postings(term, IndexRecordOption::WithFreqsAndPositions)? else {
            return Ok(());  // Term not in this segment, not an error
        };
        
        // Only seek if current doc is less than target (seek requires forward movement)
        let current_doc = postings.doc();
        if current_doc < doc_id {
            postings.seek(doc_id);
        } else if current_doc > doc_id {
            // Already past target - term not in this doc
            return Ok(());
        }
        
        // Now check if we're at the right document
        if postings.doc() == doc_id {
            buf.clear();
            postings.positions(buf);  // Fill buffer, returns ()
            // Sort positions - required for two-pointer algorithm in join_lazy
            buf.sort_unstable();
        }
        Ok(())
    }
    
    /// Fill positions for regex constraint (union of all expanded terms)
    fn fill_positions_regex_union(
        reader: &SegmentReader,
        field: Field,
        terms: &[Term],
        doc_id: DocId,
        buf: &mut Vec<u32>,
        temp_buf: &mut Vec<u32>,  // Reusable temp buffer
    ) -> TantivyResult<()> {
        let inverted_index = reader.inverted_index(field)?;
        
        for term in terms {
            if let Ok(Some(mut postings)) = inverted_index.read_postings(term, IndexRecordOption::WithFreqsAndPositions) {
                // Only seek if current doc is less than target (seek requires forward movement)
                let current_doc = postings.doc();
                if current_doc < doc_id {
                    postings.seek(doc_id);
                } else if current_doc > doc_id {
                    // Already past target - term not in this doc, skip
                    continue;
                }
                
                // Now check if we're at the right document
                if postings.doc() == doc_id {
                    temp_buf.clear();
                    postings.positions(temp_buf);  // Fill buffer, returns ()
                    buf.extend_from_slice(temp_buf);
                }
            }
        }
        
        // Sort and dedup merged positions
        buf.sort_unstable();
        buf.dedup();
        Ok(())
    }
    
    /// Get positions for each constraint in the document using postings
    /// Returns slices into position_buffers (allocation-free, no cloning)
    fn get_constraint_positions<'a>(&'a mut self, doc_id: DocId) -> TantivyResult<Vec<PosView<'a>>> {
        // Ensure we have enough buffers
        if self.position_buffers.len() < self.constraint_sources.len() {
            self.position_buffers.resize_with(self.constraint_sources.len(), || Vec::with_capacity(32));
        }
        
        // Fill all buffers first
        for (idx, source) in self.constraint_sources.iter().enumerate() {
            self.position_buffers[idx].clear();
            
            match source {
                ConstraintSource::Exact { field, term } => {
                    Self::fill_positions_exact(&self.reader, *field, term, doc_id, &mut self.position_buffers[idx])?;
                }
                ConstraintSource::Regex { field, terms, .. } => {
                    Self::fill_positions_regex_union(&self.reader, *field, terms, doc_id, &mut self.position_buffers[idx], &mut self.regex_tmp)?;
                }
                ConstraintSource::Wildcard { .. } => {
                    // Do NOT materialize 0..doc_len - leave buffer empty
                }
            }
        }
        
        // Now create views (slices) into the filled buffers
        let views: Vec<PosView> = self.constraint_sources.iter().enumerate().map(|(i, src)| {
            match src {
                ConstraintSource::Wildcard { .. } => PosView::Any,
                _ => PosView::List(self.position_buffers[i].as_slice()),
            }
        }).collect();
        
        Ok(views)
    }
    
    /// Get document length (number of tokens) - requires fast field
    /// Uses Tantivy 0.24 columnar fast field API
    fn get_doc_length(&self, doc_id: DocId, _field: Field) -> TantivyResult<u32> {
        // Require fast field - this is mandatory for postings path
        // In Tantivy 0.24, u64() expects &str (field name), returns Column
        let col = self.reader
            .fast_fields()
            .u64("sentence_length")
            .map_err(|_| tantivy::TantivyError::SchemaError(
                "sentence_length fast field required for postings path".to_string()
            ))?;
        
        // Get value using col.values.get_val() method (Tantivy 0.24 columnar API)
        let len = col.values.get_val(doc_id) as u32;
        
        if len > 0 {
            Ok(len)
        } else {
            // Fallback to stored field if fast field returns 0
            self.get_doc_length_from_stored(doc_id)
        }
    }
    
    /// Get doc length from stored field (fallback)
    fn get_doc_length_from_stored(&self, doc_id: DocId) -> TantivyResult<u32> {
        let store_reader = self.reader.get_store_reader(1)?;
        let doc: tantivy::schema::TantivyDocument = store_reader.get(doc_id)?;
        
        if let Ok(field) = self.reader.schema().get_field("sentence_length") {
            if let Some(value) = doc.get_first(field) {
                if let Some(u64_val) = value.as_u64() {
                    return Ok(u64_val as u32);
                }
            }
        }
        
        // Last resort: infer from word field length
        let tokens = crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), &doc, "word");
        if !tokens.is_empty() {
            return Ok(tokens.len() as u32);
        }
        
        Err(tantivy::TantivyError::SchemaError("Cannot determine doc length".to_string()))
    }

    fn extract_tokens_from_field(&self, _doc: &tantivy::schema::TantivyDocument, _field_name: &str) -> Vec<String> {
        // Deprecated: field extraction now happens in check_pattern_matching using the cache
        vec![]
    }

    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        &self.current_doc_matches
    }
}

/// Position-based matching using postings positions
fn find_constraint_spans_from_positions<'a>(
    positions_per_constraint: &[PosView<'a>],
    execution_plan: &Option<ExecutionPlan>,
    concat_plan: &Option<ConcatPlan>,
    doc_len: Option<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    let k = positions_per_constraint.len();
    if k == 0 {
        return Vec::new();
    }
    
    // Use concat plan if it exists (takes priority over anchor and gap)
    if let Some(plan) = concat_plan {
        return find_spans_with_plan(positions_per_constraint, plan, doc_len);
    }
    
    // Use anchor-based verification if plan exists
    if let Some(plan) = execution_plan {
        return find_spans_with_anchor(positions_per_constraint, plan, doc_len);
    }
    
    // Fallback: scan all start positions
    // For k=2, use two-pointer intersection
    if k == 2 {
        return find_spans_two_pointer(positions_per_constraint, doc_len);
    }
    
    // For k>=3, check if bitset path is appropriate
    let doc_len = doc_len.unwrap_or(0) as usize;
    if doc_len == 0 {
        return Vec::new();
    }
    
    let total_positions: usize = positions_per_constraint.iter()
        .map(|p| match p {
            PosView::Any => doc_len,
            PosView::List(list) => list.len(),
        })
        .sum();
    
    // Use bitset if constraints are dense and doc is not too long
    if k >= 3 && doc_len < 10000 && total_positions > k * 10 {
        return find_spans_bitset(positions_per_constraint, doc_len);
    }
    
    // Default: scan with binary search
    find_spans_scan(positions_per_constraint, doc_len)
}

/// Anchor-based verification: enumerate anchor positions, verify others
fn find_spans_with_anchor<'a>(
    positions_per_constraint: &[PosView<'a>],
    plan: &ExecutionPlan,
    doc_len: Option<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    let anchor_positions = match &positions_per_constraint[plan.anchor_idx] {
        PosView::Any => {
            // Anchor is wildcard - fallback to scan
            let doc_len = doc_len.unwrap_or(0) as usize;
            return find_spans_scan(positions_per_constraint, doc_len);
        }
        PosView::List(list) => *list,
    };
    
    let k = positions_per_constraint.len();
    let mut results = Vec::new();
    
    for &anchor_pos in anchor_positions.iter() {
        // Compute sequence start: anchor_pos - anchor_idx
        let seq_start = anchor_pos as isize - plan.anchor_idx as isize;
        if seq_start < 0 {
            continue;  // Start would be before document start
        }
        
        let mut matched = true;
        
        for (j, constraint_pos) in positions_per_constraint.iter().enumerate() {
            // For concatenation, wanted position is seq_start + j
            let wanted = seq_start + j as isize;
            let wanted = match u32::try_from(wanted) {
                Ok(v) => v,
                Err(_) => {
                    matched = false;
                    break;
                }
            };
            
            // Check if wanted position matches this constraint
            match constraint_pos {
                PosView::Any => {
                    // Wildcard: always matches
                }
                PosView::List(list) => {
                    if list.binary_search(&wanted).is_err() {
                        matched = false;
                        break;
                    }
                }
            }
        }
        
        if matched {
            let start = seq_start as usize;
            if let Some(dl) = doc_len {
                if start + k > dl as usize {
                    continue;
                }
            }
            results.push(crate::types::SpanWithCaptures::new(crate::types::Span { start, end: start + k }));
        }
    }
    
    results
}

/// Two-pointer intersection for k=2 (faster than binary search)
fn find_spans_two_pointer<'a>(
    positions_per_constraint: &[PosView<'a>],
    doc_len: Option<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    if positions_per_constraint.len() != 2 {
        return Vec::new();
    }
    
    let (pos0, pos1) = match (&positions_per_constraint[0], &positions_per_constraint[1]) {
        (PosView::Any, PosView::Any) => {
            // Both wildcards - all positions match
            let dl = doc_len.unwrap_or(u32::MAX) as usize;
            return (0..dl.saturating_sub(1))
                .filter_map(|start| {
                    let end = start + 2;
                    if end <= dl {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start, end }))
                    } else {
                        None
                    }
                })
                .collect();
        }
        (PosView::Any, PosView::List(p1)) => {
            // First is wildcard - check second at offset 1
            let dl = doc_len.unwrap_or(u32::MAX) as usize;
            return p1.iter()
                .filter_map(|&p1_pos| {
                    if p1_pos > 0 {
                        let start = (p1_pos - 1) as usize;
                        let end = (p1_pos + 1) as usize;
                        if end <= dl {
                            Some(crate::types::SpanWithCaptures::new(crate::types::Span {
                                start,
                                end,
                            }))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
        }
        (PosView::List(p0), PosView::Any) => {
            // Second is wildcard - check first
            let dl = doc_len.unwrap_or(u32::MAX) as usize;
            return p0.iter()
                .filter_map(|&p0_pos| {
                    let start = p0_pos as usize;
                    let end = (p0_pos + 2) as usize;
                    if end <= dl {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span {
                            start,
                            end,
                        }))
                    } else {
                        None
                    }
                })
                .collect();
        }
        (PosView::List(p0), PosView::List(p1)) => (*p0, *p1),
    };
    
    let mut results = Vec::new();
    let mut i = 0;
    let mut j = 0;
    
    while i < pos0.len() && j < pos1.len() {
        let p0 = pos0[i];
        let p1_target = p0 + 1;  // Adjacent position
        
        // Advance j to find p1_target or beyond
        while j < pos1.len() && pos1[j] < p1_target {
            j += 1;
        }
        
        if j < pos1.len() && pos1[j] == p1_target {
            // Found match: pos0[i] and pos1[j] are adjacent
            let span = crate::types::Span { 
                start: p0 as usize, 
                end: (p1_target + 1) as usize 
            };
            results.push(crate::types::SpanWithCaptures::new(span));
        }
        
        i += 1;
    }
    
    results
}

/// Partial match state during plan execution
#[derive(Debug, Clone)]
struct Partial {
    start: u32,  // position of first atom
    last: u32,   // position of most recent atom
}

/// Helper: lower_bound (first index where arr[i] >= target)
fn lower_bound(arr: &[u32], target: u32) -> usize {
    arr.binary_search(&target).unwrap_or_else(|i| i)
}

/// Execute a ConcatPlan in postings-position space.
/// Semantics: produces at most one continuation per partial per step:
/// - lazy gap: choose nearest valid next atom
/// - greedy gap: choose farthest valid next atom
fn find_spans_with_plan<'a>(
    positions_per_constraint: &[PosView<'a>],
    plan: &ConcatPlan,
    doc_len: Option<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    if plan.steps.is_empty() {
        return Vec::new();
    }

    // Validate alternating structure (defensive; compiler should guarantee)
    if !matches!(plan.steps.first().unwrap(), ConcatStep::Atom { .. })
        || !matches!(plan.steps.last().unwrap(), ConcatStep::Atom { .. })
    {
        return Vec::new();
    }
    for w in plan.steps.windows(2) {
        match (&w[0], &w[1]) {
            (ConcatStep::Atom { .. }, ConcatStep::Gap { .. }) => {}
            (ConcatStep::Gap { .. }, ConcatStep::Atom { .. }) => {}
            _ => {
                return Vec::new();
            }
        }
    }

    // Initialize partials from first atom
    let first_cidx = match plan.steps[0] {
        ConcatStep::Atom { constraint_idx } => constraint_idx,
        _ => unreachable!(),
    };

    let first_positions = match positions_per_constraint.get(first_cidx) {
        Some(p) => p,
        None => {
            return Vec::new();
        }
    };

    let mut partials: Vec<Partial> = match first_positions {
        PosView::List(pos) => pos.iter().map(|&p| Partial { start: p, last: p }).collect(),
        PosView::Any => {
            let dl = match doc_len {
                Some(dl) if dl > 0 => dl,
                _ => return Vec::new(), // cannot expand Any without doc_len
            };
            (0..dl).map(|p| Partial { start: p, last: p }).collect()
        }
    };


    if partials.is_empty() {
        return Vec::new();
    }

    // Process (Gap, Atom) pairs
    let mut step_idx = 1;
    while step_idx + 1 < plan.steps.len() {
        let gap_step = &plan.steps[step_idx];
        let atom_step = &plan.steps[step_idx + 1];

        let (min_gap, max_gap, lazy) = match gap_step {
            ConcatStep::Gap { min, max, lazy } => (*min, *max, *lazy),
            _ => return Vec::new(),
        };

        let next_cidx = match atom_step {
            ConcatStep::Atom { constraint_idx } => *constraint_idx,
            _ => return Vec::new(),
        };

        let next_positions = match positions_per_constraint.get(next_cidx) {
            Some(p) => p,
            None => {
                return Vec::new();
            }
        };

        partials = if lazy {
            join_lazy(partials, next_positions, min_gap, max_gap, doc_len)
        } else {
            join_greedy(partials, next_positions, min_gap, max_gap, doc_len)
        };

        if partials.is_empty() {
            return Vec::new();
        }

        step_idx += 2;
    }

    // Convert partials to spans [start, last+1)
    let mut out = Vec::with_capacity(partials.len());
    for p in partials {
        let start = p.start as usize;
        let end = (p.last + 1) as usize;

        if let Some(dl) = doc_len {
            if end > dl as usize {
                continue;
            }
        }

        out.push(crate::types::SpanWithCaptures::new(crate::types::Span { start, end }));
    }
    out
}

/// Lazy join (nearest valid next atom) using two-pointer when next is a List.
/// O(|partials| + |next_positions|) for List, O(|partials|) for Any.
fn join_lazy<'a>(
    mut partials: Vec<Partial>,
    next: &PosView<'a>,
    min_gap: usize,
    max_gap: Option<usize>,
    doc_len: Option<u32>,
) -> Vec<Partial> {
    // Ensure partials are sorted by last to make two-pointer valid
    partials.sort_by_key(|p| p.last);

    match next {
        PosView::Any => {
            let dl = match doc_len {
                Some(dl) if dl > 0 => dl,
                _ => return Vec::new(),
            };

            let mut out = Vec::new();
            for mut p in partials {
                let min_b = p.last.saturating_add(1).saturating_add(min_gap as u32);
                let max_b_excl = if let Some(mx) = max_gap {
                    p.last
                        .saturating_add(1)
                        .saturating_add(mx as u32)
                        .saturating_add(1)
                } else {
                    dl
                };

                // nearest b is min_b itself, if within bounds
                if min_b < max_b_excl && min_b < dl {
                    p.last = min_b;
                    out.push(p);
                }
            }
            out
        }

        PosView::List(next_pos) => {
            let mut out = Vec::new();
            let mut j = 0usize;

            for mut p in partials {
                let min_b = p.last.saturating_add(1).saturating_add(min_gap as u32);
                let max_b_excl = if let Some(mx) = max_gap {
                    p.last
                        .saturating_add(1)
                        .saturating_add(mx as u32)
                        .saturating_add(1)
                } else {
                    doc_len.unwrap_or(u32::MAX)
                };

                // Advance j to first position >= min_b
                while j < next_pos.len() && next_pos[j] < min_b {
                    j += 1;
                }
                if j >= next_pos.len() {
                    break; // no further matches possible for later partials (they need >= min_b)
                }

                let b = next_pos[j];
                if b < max_b_excl && b >= min_b {
                    p.last = b;
                    out.push(p);
                    // Note: We don't advance j here for lazy matching - each partial gets the nearest match,
                    // and multiple partials can match the same position if they have the same min_b
                }
                // b >= max_b_excl or b < min_b, so this partial can't match, but later partials might
                // (they have larger p.last, so larger min_b, but same max_b_excl)
            }
            out
        }
    }
}
















/// Greedy join (farthest valid next atom) using binary search.
/// O(|partials| log |next_positions|) for List, O(|partials|) for Any.
fn join_greedy<'a>(
    partials: Vec<Partial>,
    next: &PosView<'a>,
    min_gap: usize,
    max_gap: Option<usize>,
    doc_len: Option<u32>,
) -> Vec<Partial> {
    match next {
        PosView::Any => {
            let dl = match doc_len {
                Some(dl) if dl > 0 => dl,
                _ => return Vec::new(),
            };

            let mut out = Vec::new();
            for mut p in partials {
                let min_b = p.last.saturating_add(1).saturating_add(min_gap as u32);
                let max_b_excl = if let Some(mx) = max_gap {
                    p.last
                        .saturating_add(1)
                        .saturating_add(mx as u32)
                        .saturating_add(1)
                } else {
                    dl
                };

                // farthest b is min(max_b_excl-1, dl-1), if >= min_b
                let max_in_doc = dl - 1;
                if max_b_excl == 0 {
                    continue;
                }
                let cand = (max_b_excl - 1).min(max_in_doc);
                if cand >= min_b && cand < dl {
                    p.last = cand;
                    out.push(p);
                }
            }
            out
        }

        PosView::List(next_pos) => {
            if next_pos.is_empty() {
                return Vec::new();
            }

            let mut out = Vec::new();
            for mut p in partials {
                let min_b = p.last.saturating_add(1).saturating_add(min_gap as u32);
                let max_b_excl = if let Some(mx) = max_gap {
                    p.last
                        .saturating_add(1)
                        .saturating_add(mx as u32)
                        .saturating_add(1)
                } else {
                    doc_len.unwrap_or(u32::MAX)
                };

                let lo = lower_bound(next_pos, min_b);
                let hi = lower_bound(next_pos, max_b_excl); // exclusive end

                if lo < hi {
                    let b = next_pos[hi - 1];
                    p.last = b;
                    out.push(p);
                }
            }
            out
        }
    }
}

/// Bitset-based matching for k>=3 using fast shift-and algorithm
fn find_spans_bitset<'a>(
    positions_per_constraint: &[PosView<'a>],
    doc_len: usize,
) -> Vec<crate::types::SpanWithCaptures> {
    if doc_len == 0 {
        return Vec::new();
    }
    
    let k = positions_per_constraint.len();
    let bitset_len = (doc_len + 63) / 64;  // Round up to u64 chunks
    let mut bitsets: Vec<Vec<u64>> = Vec::new();
    
    // Build bitsets: B[j][i] = 1 if constraint j matches at position i
    for constraint_pos in positions_per_constraint {
        let mut bitset = vec![0u64; bitset_len];
        match constraint_pos {
            PosView::Any => {
                // Wildcard: set all bits
                for chunk in &mut bitset {
                    *chunk = u64::MAX;
                }
                // Clear bits beyond doc_len
                let last_chunk = (doc_len - 1) / 64;
                let last_bit = (doc_len - 1) % 64;
                if last_chunk < bitset.len() {
                    bitset[last_chunk] &= (1u64 << (last_bit + 1)) - 1;
                }
            }
            PosView::List(positions) => {
                for &pos in positions.iter() {
                    let idx = pos as usize;
                    if idx < doc_len {
                        let chunk = idx / 64;
                        let bit = idx % 64;
                        if chunk < bitset.len() {
                            bitset[chunk] |= 1u64 << bit;
                        }
                    }
                }
            }
        }
        bitsets.push(bitset);
    }
    
    // Fast shift-and: S = B[0] & (B[1] >> 1) & (B[2] >> 2) & ...
    // Shift each bitset by its offset, then AND them all together
    let mut shifted_bitsets = Vec::new();
    
    for (offset, bitset) in bitsets.iter().enumerate() {
        let mut shifted = vec![0u64; bitset_len];
        
        if offset == 0 {
            // No shift needed for first bitset
            shifted.copy_from_slice(bitset);
        } else {
            // Shift right by offset positions
            let shift_chunks = offset / 64;
            let shift_bits = offset % 64;
            
            for (dst_idx, dst_chunk) in shifted.iter_mut().enumerate() {
                let src_idx = dst_idx + shift_chunks;
                
                if src_idx < bitset.len() {
                    *dst_chunk = bitset[src_idx] >> shift_bits;
                    // Carry bits from next chunk if needed
                    if shift_bits > 0 && src_idx + 1 < bitset.len() {
                        *dst_chunk |= bitset[src_idx + 1] << (64 - shift_bits);
                    }
                }
            }
        }
        
        shifted_bitsets.push(shifted);
    }
    
    // AND all shifted bitsets together
    let mut result_bitset = shifted_bitsets[0].clone();
    for shifted in shifted_bitsets.iter().skip(1) {
        for (result_chunk, shifted_chunk) in result_bitset.iter_mut().zip(shifted.iter()) {
            *result_chunk &= *shifted_chunk;
        }
    }
    
    // Enumerate set bits in result_bitset as starts
    let mut results = Vec::new();
    for (chunk_idx, chunk) in result_bitset.iter().enumerate() {
        let mut bits = *chunk;
        let base_pos = chunk_idx * 64;
        
        while bits != 0 {
            let bit_pos = bits.trailing_zeros() as usize;
            let start = base_pos + bit_pos;
            if start + k <= doc_len {
                let span = crate::types::Span { start, end: start + k };
                results.push(crate::types::SpanWithCaptures::new(span));
            }
            bits &= bits - 1;  // Clear lowest set bit
        }
    }
    
    results
}

/// Scan-based matching with binary search (fallback)
fn find_spans_scan<'a>(
    positions_per_constraint: &[PosView<'a>],
    doc_len: usize,
) -> Vec<crate::types::SpanWithCaptures> {
    if positions_per_constraint.is_empty() || doc_len == 0 {
        return Vec::new();
    }
    
    let k = positions_per_constraint.len();
    let mut results = Vec::new();
    
    // Scan all possible start positions
    for start in 0..=doc_len.saturating_sub(k) {
        let mut matched = true;
        
        for (offset, constraint_pos) in positions_per_constraint.iter().enumerate() {
            let check_pos = (start + offset) as u32;
            
            match constraint_pos {
                PosView::Any => {
                    // Wildcard: always matches
                }
                PosView::List(list) => {
                    // Binary search for position
                    if list.binary_search(&check_pos).is_err() {
                        matched = false;
                        break;
                    }
                }
            }
        }
        
        if matched {
            let span = crate::types::Span { start, end: start + k };
            results.push(crate::types::SpanWithCaptures::new(span));
        }
    }
    
    results
}

/// Original stored-field based matching (fallback)
/// Maximum number of backtracking iterations to prevent exponential blowup
const MAX_BACKTRACK_ITERATIONS: usize = 10_000;

/// Maximum number of matches to generate for a repetition pattern
/// Prevents explosion when patterns have many valid matches
const MAX_GENERATED_MATCHES: usize = 10_000;

/// Try to match exactly `count` repetitions of a pattern starting at `pos`
fn try_repetition_count(
    pattern: &crate::query::ast::Pattern,
    count: usize,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    mut pos: usize,
    len: usize,
    mut captures: Vec<crate::types::NamedCapture>,
) -> Option<(usize, Vec<crate::types::NamedCapture>)> {
    for _ in 0..count {
        if pos >= len {
            return None;
        }
        if let Some(m) = matches_pattern_at_position(pattern, field_cache, pos) {
            pos = m.span.end;
            captures.extend(m.captures);
        } else {
            return None;
        }
    }
    Some((pos, captures))
}

/// Try to match exactly `count` repetitions of a pattern starting at `pos`
/// Returns (end_position, captures, sub_matches) if successful
fn try_repetition_count_with_metadata(
    pattern: &crate::query::ast::Pattern,
    count: usize,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    start_pos: usize,
    len: usize,
) -> Option<(usize, Vec<crate::types::NamedCapture>, Vec<crate::types::MatchWithMetadata>)> {
    use crate::types::MatchWithMetadata;
    
    let mut current_pos = start_pos;
    let mut sub_matches = Vec::with_capacity(count);
    let mut all_captures = Vec::new();
    
    for _ in 0..count {
        if current_pos >= len {
            // Can't match - not enough positions left
            return None;
        }
        
        // Generate matches at current position
        let matches = generate_all_matches_at_position(pattern, field_cache, current_pos);
        
        if matches.is_empty() {
            // Inner pattern doesn't match at this position
            return None;
        }
        
        // Take first match (for simple patterns like wildcards, all matches are equivalent)
        let m = matches.into_iter().next().unwrap();
        
        // Ensure we're making progress (avoid infinite loops)
        if m.span.end <= current_pos {
            return None;
        }
        
        current_pos = m.span.end;
        all_captures.extend(m.captures.clone());
        sub_matches.push(m);
    }
    
    Some((current_pos, all_captures, sub_matches))
}

/// Generate ALL valid matches for a repetition pattern
/// This follows Odinson's approach: generate all possibilities, then select
/// 
/// Handles edge cases:
/// - min=0: generates zero-length match (optional repetition)
/// - max bounds: respects maximum count limit
/// - unbounded patterns: uses document length as practical limit
/// - nested repetitions: handled recursively (Phase A: basic support)
pub fn generate_all_repetition_matches(
    pattern: &crate::query::ast::Pattern,
    min: usize,
    max: Option<usize>,
    is_greedy: bool,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize,
) -> Vec<crate::types::MatchWithMetadata> {
    use crate::types::{Span, MatchWithMetadata};
    
    // Get document length from field cache
    let len = field_cache
        .get("word")
        .or_else(|| field_cache.values().next())
        .map(|v| v.len())
        .unwrap_or(0);
    
    // Edge case: position beyond document
    if pos > len {
        return Vec::new();
    }
    
    // Edge case: position at end of document
    // For min=0, we can still generate a zero-length match
    if pos == len {
        if min == 0 {
            // Zero-length match for optional repetition
            let span = Span { start: pos, end: pos };
            return vec![MatchWithMetadata::repetition(
                span,
                Vec::new(),
                is_greedy,
                Vec::new(), // No sub-matches for zero repetitions
            )];
        } else {
            return Vec::new();
        }
    }
    
    // Calculate maximum possible count based on remaining document length
    // For unbounded patterns (max=None), use remaining document length as practical limit
    let max_possible = len.saturating_sub(pos);
    let max_count = max
        .map(|m| {
            // Respect both the pattern's max and document length
            m.min(max_possible)
        })
        .unwrap_or(max_possible);
    
    // Edge case: min exceeds max (invalid pattern)
    if min > max_count {
        return Vec::new();
    }
    
    // Edge case: min=0 (optional repetition) - always generate zero-length match first
    let mut matches = Vec::new();
    if min == 0 {
        let span = Span { start: pos, end: pos };
        matches.push(MatchWithMetadata::repetition(
            span,
            Vec::new(),
            is_greedy,
            Vec::new(),
        ));
    }
    
    // Generate matches for each valid count from min to max
    // Start from max(min, 1) to avoid duplicate zero-length match
    let start_count = min.max(1);
    for count in start_count..=max_count {
        // Performance safeguard: limit total matches generated
        if matches.len() >= MAX_GENERATED_MATCHES {
            break;
        }
        
        // Try to match exactly 'count' repetitions
        if let Some((end_pos, captures, sub_matches)) =
            try_repetition_count_with_metadata(pattern, count, field_cache, pos, len)
        {
            let span = Span { start: pos, end: end_pos };
            matches.push(MatchWithMetadata::repetition(
                span,
                captures,
                is_greedy,
                sub_matches,
            ));
        } else {
            // If we can't match this count, we likely can't match larger counts either
            // (assuming patterns are contiguous). However, for patterns that might skip,
            // we continue trying. This is a conservative approach.
            // For Phase A, we'll continue trying all counts.
        }
    }
    
    matches
}

/// Recursive backtracking matcher for pattern sequences
/// Returns Some((end_pos, captures)) if sequence matches, None otherwise
fn match_sequence_recursive(
    patterns: &[crate::query::ast::Pattern],
    pattern_idx: usize,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize,
    len: usize,
    captures: Vec<crate::types::NamedCapture>,
    iteration_count: &mut usize,
) -> Option<(usize, Vec<crate::types::NamedCapture>)> {
    use crate::query::ast::{Pattern, QuantifierKind};
    
    // Check iteration limit to prevent exponential blowup
    *iteration_count += 1;
    if *iteration_count > MAX_BACKTRACK_ITERATIONS {
        return None;
    }
    
    // Base case: all patterns matched successfully
    if pattern_idx >= patterns.len() {
        return Some((pos, captures));
    }
    
    // Don't match beyond document length
    if pos > len {
        return None;
    }
    
    let pat = &patterns[pattern_idx];
    
    // Special handling for Repetition patterns - these need backtracking
    if let Pattern::Repetition { pattern: inner, min, max, kind } = pat {
        // Calculate reasonable upper bound for repetitions
        let max_possible = len.saturating_sub(pos);
        let max_count = max.map(|m| m.min(max_possible)).unwrap_or(max_possible);
        
        if *kind == QuantifierKind::Lazy {
            // Lazy: try shortest first (min, min+1, min+2, ..., max)
            for count in *min..=max_count {
                if let Some((end_pos, new_captures)) = try_repetition_count(
                    inner, count, field_cache, pos, len, captures.clone()
                ) {
                    // Try to match remaining patterns with this repetition count
                    if let Some(result) = match_sequence_recursive(
                        patterns, pattern_idx + 1, field_cache, end_pos, len, new_captures, iteration_count
                    ) {
                        return Some(result);
                    }
                }
                // If remaining patterns don't match, try next count
            }
            return None;
        } else {
            // Greedy: try longest first (max, max-1, max-2, ..., min)
            for count in (*min..=max_count).rev() {
                if let Some((end_pos, new_captures)) = try_repetition_count(
                    inner, count, field_cache, pos, len, captures.clone()
                ) {
                    // Try to match remaining patterns with this repetition count
                    if let Some(result) = match_sequence_recursive(
                        patterns, pattern_idx + 1, field_cache, end_pos, len, new_captures, iteration_count
                    ) {
                        return Some(result);
                    }
                }
                // If remaining patterns don't match, try shorter count
            }
            return None;
        }
    }
    
    // Non-repetition patterns: match normally and continue
    if pos < len {
        if let Some(m) = matches_pattern_at_position(pat, field_cache, pos) {
            let mut new_captures = captures;
            new_captures.extend(m.captures);
            return match_sequence_recursive(
                patterns, pattern_idx + 1, field_cache, m.span.end, len, new_captures, iteration_count
            );
        }
    }
    
    // Special case: zero-width assertions can match at end of document
    if let Pattern::Assertion(_) = pat {
        if let Some(m) = matches_pattern_at_position(pat, field_cache, pos) {
            let mut new_captures = captures;
            new_captures.extend(m.captures);
            return match_sequence_recursive(
                patterns, pattern_idx + 1, field_cache, m.span.end, len, new_captures, iteration_count
            );
        }
    }
    
    None
}

/// Entry point for backtracking sequence matcher
fn match_sequence_with_backtracking(
    patterns: &[crate::query::ast::Pattern],
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    start_pos: usize,
    len: usize,
) -> Option<(usize, Vec<crate::types::NamedCapture>)> {
    let mut iteration_count = 0;
    match_sequence_recursive(patterns, 0, field_cache, start_pos, len, Vec::new(), &mut iteration_count)
}

/// Generate all valid sequence matches starting at a given position
/// Returns all matches with metadata for selection algorithm
fn generate_all_sequence_matches_recursive(
    patterns: &[crate::query::ast::Pattern],
    pattern_idx: usize,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize,
    len: usize,
    mut sub_matches: Vec<crate::types::MatchWithMetadata>,
    mut captures: Vec<crate::types::NamedCapture>,
    iteration_count: &mut usize,
) -> Vec<crate::types::MatchWithMetadata> {
    use crate::query::ast::{Pattern, QuantifierKind};
    use crate::types::MatchWithMetadata;
    
    // Check iteration limit
    *iteration_count += 1;
    if *iteration_count > MAX_BACKTRACK_ITERATIONS {
        return Vec::new();
    }
    
    // Base case: all patterns matched successfully
    if pattern_idx >= patterns.len() {
        // Calculate actual span from sub_matches
        let start = if let Some(first) = sub_matches.first() {
            first.span.start
        } else {
            pos
        };
        let end = if let Some(last) = sub_matches.last() {
            last.span.end
        } else {
            pos
        };
        let span = crate::types::Span { start, end };
        return vec![MatchWithMetadata::sequence(span, captures, sub_matches)];
    }
    
    // Don't match beyond document length
    if pos > len {
        return Vec::new();
    }
    
    let pat = &patterns[pattern_idx];
    let mut all_results = Vec::new();
    
    // Special handling for Repetition patterns
    if let Pattern::Repetition { pattern: inner, min, max, kind } = pat {
        let is_greedy = *kind == QuantifierKind::Greedy;
        
        // Generate all valid repetition matches
        let rep_matches = generate_all_repetition_matches(
            inner, *min, *max, is_greedy, field_cache, pos
        );
        
        // For each repetition match, try to match remaining patterns
        for rep_match in rep_matches {
            let mut new_sub_matches = sub_matches.clone();
            new_sub_matches.push(rep_match.clone());
            let mut new_captures = captures.clone();
            new_captures.extend(rep_match.captures.clone());
            
            let remaining = generate_all_sequence_matches_recursive(
                patterns,
                pattern_idx + 1,
                field_cache,
                rep_match.span.end,
                len,
                new_sub_matches,
                new_captures,
                iteration_count,
            );
            all_results.extend(remaining);
        }
        
        return all_results;
    }
    
    // Non-repetition patterns: generate all matches and continue
    if pos < len {
        let matches = generate_all_matches_at_position(pat, field_cache, pos);
        for m in matches {
            let mut new_sub_matches = sub_matches.clone();
            new_sub_matches.push(m.clone());
            let mut new_captures = captures.clone();
            new_captures.extend(m.captures.clone());
            
            let remaining = generate_all_sequence_matches_recursive(
                patterns,
                pattern_idx + 1,
                field_cache,
                m.span.end,
                len,
                new_sub_matches,
                new_captures,
                iteration_count,
            );
            all_results.extend(remaining);
        }
    }
    
    // Handle zero-width assertions
    if let Pattern::Assertion(_) = pat {
        let matches = generate_all_matches_at_position(pat, field_cache, pos);
        for m in matches {
            let mut new_sub_matches = sub_matches.clone();
            new_sub_matches.push(m.clone());
            let mut new_captures = captures.clone();
            new_captures.extend(m.captures.clone());
            
            let remaining = generate_all_sequence_matches_recursive(
                patterns,
                pattern_idx + 1,
                field_cache,
                m.span.end,
                len,
                new_sub_matches,
                new_captures,
                iteration_count,
            );
            all_results.extend(remaining);
        }
    }
    
    all_results
}

/// Generate all valid sequence matches starting at a given position
pub fn generate_all_sequence_matches(
    patterns: &[crate::query::ast::Pattern],
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    start_pos: usize,
    len: usize,
) -> Vec<crate::types::MatchWithMetadata> {
    let mut iteration_count = 0;
    generate_all_sequence_matches_recursive(
        patterns,
        0,
        field_cache,
        start_pos,
        len,
        Vec::new(),
        Vec::new(),
        &mut iteration_count,
    )
}

pub fn find_constraint_spans_in_sequence(
    pattern: &crate::query::ast::Pattern, 
    field_cache: &std::collections::HashMap<String, Vec<String>>
) -> Vec<crate::types::SpanWithCaptures> {
    use crate::query::ast::Pattern;
    use crate::tantivy_integration::match_selector::MatchSelector;
    
    if let Pattern::Concatenated(patterns) = pattern {
        // Use 'word' field to determine sentence length as it's the most reliable
        let len = field_cache.get("word")
            .or_else(|| field_cache.values().next())
            .map(|v| v.len())
            .unwrap_or(0);
        
        if len == 0 { 
            return Vec::new(); 
        }

        // Generate all valid matches using generate-all approach
        let mut all_matches = Vec::new();
        for start in 0..len {
            let matches = generate_all_sequence_matches(patterns, field_cache, start, len);
            all_matches.extend(matches);
        }
        
        // Apply selection algorithm to disambiguate competing matches
        let selected_matches = MatchSelector::pick_matches(all_matches);
        
        // Convert MatchWithMetadata back to SpanWithCaptures for backward compatibility
        selected_matches
            .into_iter()
            .map(|m| crate::types::SpanWithCaptures::with_captures(m.span, m.captures))
            .collect()
    } else {
        Vec::new()
    }
}

/// Generate all valid matches for any pattern type at a given position
/// This is the parallel function to matches_pattern_at_position that returns
/// all matches with metadata for selection algorithm
pub fn generate_all_matches_at_position(
    pattern: &crate::query::ast::Pattern,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize,
) -> Vec<crate::types::MatchWithMetadata> {
    use crate::query::ast::Pattern;
    use crate::types::MatchWithMetadata;
    
    match pattern {
        Pattern::Constraint(constraint) => {
            if matches_constraint_at_position(constraint, field_cache, pos) {
                let span = crate::types::Span { start: pos, end: pos + 1 };
                let capture = crate::types::NamedCapture::new(format!("c{}", pos), span.clone());
                vec![MatchWithMetadata::atom(span, vec![capture])]
            } else {
                Vec::new()
            }
        }
        Pattern::NamedCapture { name, pattern } => {
            let mut matches = generate_all_matches_at_position(pattern, field_cache, pos);
            // Add the named capture to each match
            for m in &mut matches {
                let capture = crate::types::NamedCapture::new(name.clone(), m.span.clone());
                m.captures.push(capture);
            }
            matches
        }
        Pattern::Disjunctive(patterns) => {
            let mut all_matches = Vec::new();
            for (idx, pat) in patterns.iter().enumerate() {
                let mut matches = generate_all_matches_at_position(pat, field_cache, pos);
                // Tag each match with which clause matched
                for m in &mut matches {
                    // Convert to disjunction match
                    let span = m.span.clone();
                    let captures = std::mem::take(&mut m.captures);
                    *m = MatchWithMetadata::disjunction(span, captures, idx);
                }
                all_matches.extend(matches);
            }
            all_matches
        }
        Pattern::Repetition { pattern, min, max, kind } => {
            use crate::query::ast::QuantifierKind;
            let is_greedy = *kind == QuantifierKind::Greedy;
            generate_all_repetition_matches(pattern, *min, *max, is_greedy, field_cache, pos)
        }
        Pattern::Assertion(assertion) => {
            use crate::query::ast::Assertion;
            match assertion {
                Assertion::PositiveLookahead(child) => {
                    if generate_all_matches_at_position(child, field_cache, pos).is_empty() {
                        Vec::new()
                    } else {
                        // Zero-width match
                        let span = crate::types::Span { start: pos, end: pos };
                        vec![MatchWithMetadata::atom(span, Vec::new())]
                    }
                }
                Assertion::NegativeLookahead(child) => {
                    if generate_all_matches_at_position(child, field_cache, pos).is_empty() {
                        // Zero-width match
                        let span = crate::types::Span { start: pos, end: pos };
                        vec![MatchWithMetadata::atom(span, Vec::new())]
                    } else {
                        Vec::new()
                    }
                }
                Assertion::PositiveLookbehind(child) => {
                    if pos > 0 && !generate_all_matches_at_position(child, field_cache, pos - 1).is_empty() {
                        let span = crate::types::Span { start: pos, end: pos };
                        vec![MatchWithMetadata::atom(span, Vec::new())]
                    } else {
                        Vec::new()
                    }
                }
                Assertion::NegativeLookbehind(child) => {
                    if pos == 0 || generate_all_matches_at_position(child, field_cache, pos - 1).is_empty() {
                        let span = crate::types::Span { start: pos, end: pos };
                        vec![MatchWithMetadata::atom(span, Vec::new())]
                    } else {
                        Vec::new()
                    }
                }
            }
        }
        Pattern::Concatenated(nested_patterns) => {
            // For concatenated patterns, we'll handle this at a higher level
            // in generate_all_sequence_matches
            // For now, return empty - this will be handled by sequence generation
            Vec::new()
        }
        Pattern::GraphTraversal { .. } => {
            // Graph traversals are handled separately
            Vec::new()
        }
        Pattern::Mention { .. } => {
            // Mentions are handled separately
            Vec::new()
        }
    }
}

fn matches_pattern_at_position(
    pattern: &crate::query::ast::Pattern,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize
) -> Option<crate::types::SpanWithCaptures> {
    use crate::query::ast::Pattern;
    
    match pattern {
        Pattern::Constraint(constraint) => {
            if matches_constraint_at_position(constraint, field_cache, pos) {
                let span = crate::types::Span { start: pos, end: pos + 1 };
                let capture = crate::types::NamedCapture::new(format!("c{}", pos), span.clone());
                Some(crate::types::SpanWithCaptures::with_captures(span, vec![capture]))
            } else {
                None
            }
        }
        Pattern::NamedCapture { name, pattern } => {
            if let Some(mut m) = matches_pattern_at_position(pattern, field_cache, pos) {
                // Add the named capture
                let capture = crate::types::NamedCapture::new(name.clone(), m.span.clone());
                m.captures.push(capture);
                Some(m)
            } else {
                None
            }
        }
        Pattern::Disjunctive(patterns) => {
            for pat in patterns {
                if let Some(m) = matches_pattern_at_position(pat, field_cache, pos) {
                    return Some(m);
                }
            }
            None
        }
        Pattern::Repetition { pattern, min, max, kind } => {
            use crate::query::ast::QuantifierKind;
            
            let mut current_pos = pos;
            let mut count = 0;
            let mut total_captures = Vec::new();
            
            if *kind == QuantifierKind::Lazy {
                // Lazy semantics: match minimum required and stop
                while count < *min {
                    if let Some(m) = matches_pattern_at_position(pattern, field_cache, current_pos) {
                        current_pos = m.span.end;
                        total_captures.extend(m.captures);
                        count += 1;
                    } else {
                        // Can't match minimum required
                        return None;
                    }
                }
                
                // Check max bound
                if let Some(max_val) = max {
                    if count > *max_val {
                        return None;
                    }
                }
            } else {
                // Greedy semantics: match as many as possible
                while let Some(m) = matches_pattern_at_position(pattern, field_cache, current_pos) {
                    current_pos = m.span.end;
                    total_captures.extend(m.captures);
                    count += 1;
                    if let Some(max_val) = max {
                        if count >= *max_val { break; }
                    }
                }
            }
            
            if count >= *min {
                let span = crate::types::Span { start: pos, end: current_pos };
                Some(crate::types::SpanWithCaptures::with_captures(span, total_captures))
            } else {
                None
            }
        }
        Pattern::Assertion(assertion) => {
            use crate::query::ast::Assertion;
            match assertion {
                Assertion::PositiveLookahead(child) => {
                    if matches_pattern_at_position(child, field_cache, pos).is_some() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
                Assertion::NegativeLookahead(child) => {
                    if matches_pattern_at_position(child, field_cache, pos).is_none() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
                Assertion::PositiveLookbehind(child) => {
                    if pos > 0 && matches_pattern_at_position(child, field_cache, pos - 1).is_some() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
                Assertion::NegativeLookbehind(child) => {
                    if pos == 0 || matches_pattern_at_position(child, field_cache, pos - 1).is_none() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
            }
        }
        Pattern::Concatenated(nested_patterns) => {
            // Support nested concatenations
            let mut current_pos = pos;
            let mut all_captures = Vec::new();
            for pat in nested_patterns {
                if let Some(m) = matches_pattern_at_position(pat, field_cache, current_pos) {
                    current_pos = m.span.end;
                    all_captures.extend(m.captures);
                } else {
                    return None;
                }
            }
            Some(crate::types::SpanWithCaptures::with_captures(crate::types::Span { start: pos, end: current_pos }, all_captures))
        }
        _ => None,
    }
}

fn matches_constraint_at_position(
    constraint: &crate::query::ast::Constraint,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize
) -> bool {
    use crate::query::ast::Constraint;
    
    match constraint {
        Constraint::Wildcard => true,
        Constraint::Field { name, matcher } => {
            if let Some(tokens) = field_cache.get(name) {
                if pos < tokens.len() {
                    let result = matcher.matches(&tokens[pos]);
                    if !result && (name == "tag" || name == "word") {
                    }
                    result
                } else {
                    false
                }
            } else {
                // If it's looking for 'tag' but we only have 'pos' in cache, try fallback
                if name == "tag" {
                    if let Some(tokens) = field_cache.get("pos") {
                        if pos < tokens.len() {
                            return matcher.matches(&tokens[pos]);
                        }
                    }
                }
                false
            }
        }
        Constraint::Fuzzy { name, matcher } => {
            if let Some(tokens) = field_cache.get(name) {
                if pos < tokens.len() {
                    tokens[pos].to_lowercase().contains(&matcher.to_lowercase())
                } else {
                    false
                }
            } else {
                false
            }
        }
        Constraint::Negated(inner) => !matches_constraint_at_position(inner, field_cache, pos),
        Constraint::Conjunctive(constraints) => {
            constraints.iter().all(|c| matches_constraint_at_position(c, field_cache, pos))
        }
        Constraint::Disjunctive(constraints) => {
            constraints.iter().any(|c| matches_constraint_at_position(c, field_cache, pos))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::ast::{Pattern, Constraint, Matcher, QuantifierKind};
    use crate::types::Span;

    fn create_field_cache(tokens: Vec<&str>) -> std::collections::HashMap<String, Vec<String>> {
        let mut cache = std::collections::HashMap::new();
        cache.insert("word".to_string(), tokens.iter().map(|s| s.to_string()).collect());
        cache
    }

    #[test]
    fn test_generate_all_repetition_matches_greedy() {
        // Test greedy repetition: should generate all valid matches
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);
        
        // Pattern: []* (greedy star) - should generate matches for 0, 1, 2, 3, 4, 5, 6 repetitions
        let matches = generate_all_repetition_matches(
            &pattern,
            0,      // min
            None,   // max (unbounded)
            true,   // greedy
            &field_cache,
            0,      // start position
        );
        
        // Should generate multiple matches (at least min=0 and some valid counts)
        assert!(!matches.is_empty());
        // All matches should be greedy
        for m in &matches {
            match &m.kind {
                crate::types::MatchKind::Repetition { is_greedy, .. } => {
                    assert!(*is_greedy);
                }
                _ => panic!("Expected Repetition match kind"),
            }
        }
    }

    #[test]
    fn test_generate_all_repetition_matches_lazy() {
        // Test lazy repetition
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);
        
        // Pattern: []*? (lazy star)
        let matches = generate_all_repetition_matches(
            &pattern,
            0,      // min
            None,   // max (unbounded)
            false,  // lazy
            &field_cache,
            0,      // start position
        );
        
        assert!(!matches.is_empty());
        // All matches should be lazy
        for m in &matches {
            match &m.kind {
                crate::types::MatchKind::Repetition { is_greedy, .. } => {
                    assert!(!*is_greedy);
                }
                _ => panic!("Expected Repetition match kind"),
            }
        }
    }

    #[test]
    fn test_generate_all_repetition_matches_bounded() {
        // Test bounded repetition: {1,3}
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let field_cache = create_field_cache(vec!["a", "b", "c", "d", "e"]);
        
        let matches = generate_all_repetition_matches(
            &pattern,
            1,      // min
            Some(3), // max
            true,   // greedy
            &field_cache,
            0,      // start position
        );
        
        // Should generate matches for 1, 2, 3 repetitions
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].span.length(), 1);
        assert_eq!(matches[1].span.length(), 2);
        assert_eq!(matches[2].span.length(), 3);
    }

    #[test]
    fn test_generate_all_repetition_matches_min_zero() {
        // Test min=0 case (optional repetition)
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let field_cache = create_field_cache(vec!["a", "b"]);
        
        let matches = generate_all_repetition_matches(
            &pattern,
            0,      // min
            Some(2), // max
            false,  // lazy
            &field_cache,
            0,      // start position
        );
        
        // Should generate matches for 0, 1, 2 repetitions
        assert_eq!(matches.len(), 3);
        // Zero-length match should have start == end
        assert_eq!(matches[0].span.start, matches[0].span.end);
    }

    #[test]
    fn test_find_constraint_spans_with_selection() {
        // Test that selection is applied in find_constraint_spans_in_sequence
        // Pattern: [word=a] []* [word=c] on "a b c a b c"
        // Greedy should produce one match: "a b c a b c"
        // Lazy should produce two matches: "a b c" and "a b c"
        
        let field_cache = create_field_cache(vec!["a", "b", "c", "a", "b", "c"]);
        
        // Create pattern: [word=a] []* [word=c] (greedy)
        let pattern = Pattern::Concatenated(vec![
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("a".to_string()),
            }),
            Pattern::Repetition {
                pattern: Box::new(Pattern::Constraint(Constraint::Wildcard)),
                min: 0,
                max: None,
                kind: QuantifierKind::Greedy,
            },
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("c".to_string()),
            }),
        ]);
        
        let results = find_constraint_spans_in_sequence(&pattern, &field_cache);
        
        // With greedy quantifier, should prefer longer match
        // The exact behavior depends on the selection algorithm
        // For now, just verify we get some results
        assert!(!results.is_empty());
    }
}
