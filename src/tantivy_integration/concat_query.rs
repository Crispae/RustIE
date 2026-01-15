use tantivy::query::{Query, Weight, Scorer, EnableScoring};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet, Term};
use tantivy::schema::{Field, IndexRecordOption, Value};
use tantivy::postings::Postings;
use std::collections::HashMap;
use crate::compiler::ast::{Pattern, Constraint, Matcher};

/// Execution plan for anchor-based verification
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub anchor_idx: usize,
}

/// Gap plan for gap-based matching (Constraint A + Repetition(Wildcard) + Constraint B)
#[derive(Debug, Clone)]
pub struct GapPlan {
    pub constraint_a_idx: usize,
    pub constraint_b_idx: usize,
    pub min_gap: usize,
    pub max_gap: Option<usize>,
    pub lazy: bool,
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
    pub gap_plan: Option<GapPlan>,
}

impl Clone for RustieConcatQuery {
    fn clone(&self) -> Self {
        RustieConcatQuery {
            default_field: self.default_field,
            pattern: self.pattern.clone(),
            sub_queries: self.sub_queries.iter().map(|q| q.box_clone()).collect(),
            execution_plan: self.execution_plan.clone(),
            gap_plan: self.gap_plan.clone(),
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
            gap_plan: None,
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
            gap_plan: self.gap_plan.clone(),
        }))
    }
}

struct RustieConcatWeight {
    sub_weights: Vec<Box<dyn Weight>>,
    pattern: Pattern,
    default_field: Field,
    gap_plan: Option<GapPlan>,
}

/// Helper to extract constraints from a Pattern
fn extract_constraints_from_pattern(pattern: &Pattern) -> Vec<&Constraint> {
    use crate::compiler::ast::Pattern;
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
) -> TantivyResult<Vec<ConstraintSource>> {
    use crate::compiler::ast::Pattern;
    let mut sources = Vec::new();
    let schema = reader.schema();
    
    // Segment-local regex cache (immutable, no mutex needed)
    let mut regex_cache: HashMap<String, Vec<Term>> = HashMap::new();
    
    match pattern {
        Pattern::Concatenated(patterns) => {
            for pat in patterns {
                let constraint = match pat {
                    Pattern::Constraint(c) => c,
                    Pattern::NamedCapture { pattern: p, .. } => {
                        if let Pattern::Constraint(c) = p.as_ref() {
                            c
                        } else {
                            continue;  // Skip non-constraint patterns for now
                        }
                    }
                    _ => continue,
                };
                
                let source = compile_constraint_to_source(constraint, reader, default_field, &mut regex_cache, schema)?;
                sources.push(source);
            }
        }
        Pattern::Constraint(c) => {
            let source = compile_constraint_to_source(c, reader, default_field, &mut regex_cache, schema)?;
            sources.push(source);
        }
        _ => {
            // For non-constraint patterns, we'll fall back to stored-field path
            return Ok(Vec::new());
        }
    }
    
    Ok(sources)
}

/// Compile a single constraint to ConstraintSource
fn compile_constraint_to_source(
    constraint: &Constraint,
    reader: &SegmentReader,
    default_field: &Field,
    regex_cache: &mut HashMap<String, Vec<Term>>,
    schema: &tantivy::schema::Schema,
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
                    // Check cache first
                    let cache_key = format!("{}:{}", name, pattern);
                    let terms = if let Some(cached) = regex_cache.get(&cache_key) {
                        cached.clone()
                    } else {
                        // Expand regex using FST
                        let inverted_index = reader.inverted_index(field)
                            .map_err(|e| tantivy::TantivyError::SchemaError(format!("Failed to get inverted index: {}", e)))?;
                        let term_dict = inverted_index.terms();
                        
                        let automaton = tantivy_fst::Regex::new(pattern)
                            .map_err(|e| tantivy::TantivyError::SchemaError(format!("Invalid regex pattern '{}': {}", pattern, e)))?;
                        
                        let mut stream = term_dict.search(&automaton).into_stream()
                            .map_err(|e| tantivy::TantivyError::SchemaError(format!("Failed to search term dict: {:?}", e)))?;
                        
                        let mut terms = Vec::new();
                        let mut count = 0;
                        
                        while stream.advance() {
                            if count >= MAX_REGEX_EXPANSION {
                                log::warn!("Regex pattern '{}' exceeds expansion cap ({}), truncating", pattern, MAX_REGEX_EXPANSION);
                                break;
                            }
                            
                            let term_bytes = stream.key();
                            let term = Term::from_field_bytes(field, term_bytes);
                            terms.push(term);
                            count += 1;
                        }
                        
                        if terms.is_empty() {
                            log::debug!("Regex pattern '{}' matched 0 terms in segment", pattern);
                        } else {
                            log::debug!("Regex pattern '{}' expanded to {} terms", pattern, terms.len());
                            regex_cache.insert(cache_key, terms.clone());
                        }
                        
                        terms
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
                if num_sub_weights == 2 {
                    println!("DEBUG: Created sub_scorer[{}] for segment {}, initial doc={} (before advance)", 
                             i, reader.segment_id(), initial_doc);
                    // Try to see if we can get more info about the scorer
                    if initial_doc != tantivy::TERMINATED {
                        println!("DEBUG: Sub_scorer[{}] already positioned at doc {}", i, initial_doc);
                    }
                }
                Ok(scorer)
            })
            .collect::<TantivyResult<Vec<_>>>()?;

        let is_simple = sub_scorers.len() == 2;
        if is_simple {
            println!("DEBUG: Creating RustieConcatScorer with {} sub_scorers, pattern={:?}", sub_scorers.len(), self.pattern);
        }

        // Compile constraint sources for postings-based Phase 2
        let constraint_sources = match compile_constraint_sources(&self.pattern, reader, &self.default_field) {
            Ok(sources) => sources,
            Err(e) => {
                log::warn!("Failed to compile constraint sources, falling back to stored-field path: {}", e);
                Vec::new()  // Empty means fallback to stored-field path
            }
        };

        // Compute execution plan (anchor-based verification)
        let execution_plan = compute_execution_plan(&constraint_sources, reader);
        
        // Get gap plan from query
        let gap_plan = self.gap_plan.clone();

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
            gap_plan,
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
    gap_plan: Option<GapPlan>,
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
            for scorer in self.sub_scorers.iter_mut() {
                if scorer.advance() == tantivy::TERMINATED {
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

            for scorer in self.sub_scorers.iter_mut().skip(1) {
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
        log::trace!("check_pattern_matching called for doc_id={} with pattern={:?}", doc_id, self.pattern);
        
        self.current_doc_matches.clear();
        
        // Try postings-based path first if constraint sources are available
        if !self.constraint_sources.is_empty() {
            // Get doc length and execution plan before mutable borrow
            let doc_len = self.get_doc_length(doc_id, self.default_field).ok();
            let execution_plan = self.execution_plan.clone();
            let gap_plan = self.gap_plan.clone();
            
            match self.get_constraint_positions(doc_id) {
                Ok(positions_per_constraint) => {
                    // Use position-based matching
                    let all_spans = find_constraint_spans_from_positions(
                        &positions_per_constraint,
                        &execution_plan,
                        &gap_plan,
                        doc_len,
                    );
                    self.current_doc_matches = all_spans;
                    return !self.current_doc_matches.is_empty();
                }
                Err(e) => {
                    log::debug!("Postings-based path failed for doc {}: {}, falling back to stored-field path", doc_id, e);
                    // Fall through to stored-field path
                }
            }
        }
        
        // Fallback to stored-field path
        let store_reader = match self.reader.get_store_reader(1) {
            Ok(reader) => reader,
            Err(_) => return false,
        };
        let doc = match store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => return false,
        };

        let mut field_cache = std::collections::HashMap::new();
        let field_names = ["word", "lemma", "pos", "tag", "chunk", "entity", "norm"];
        for name in field_names {
            let tokens = crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), &doc, name);
            if !tokens.is_empty() {
                field_cache.insert(name.to_string(), tokens);
            }
        }

        // Find all valid spans matching the concatenated pattern
        let all_spans = find_constraint_spans_in_sequence(&self.pattern, &field_cache);
        self.current_doc_matches = all_spans;
        
        !self.current_doc_matches.is_empty()
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
        Ok(self.constraint_sources.iter().enumerate().map(|(i, src)| {
            match src {
                ConstraintSource::Wildcard { .. } => PosView::Any,
                _ => PosView::List(self.position_buffers[i].as_slice()),
            }
        }).collect())
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
    gap_plan: &Option<GapPlan>,
    doc_len: Option<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    let k = positions_per_constraint.len();
    if k == 0 {
        return Vec::new();
    }
    
    // Use gap-based matching if gap plan exists (takes priority over anchor)
    if let Some(gap) = gap_plan {
        #[cfg(debug_assertions)]
        {
            assert!(
                gap.constraint_a_idx < positions_per_constraint.len(),
                "GapPlan constraint_a_idx out of bounds"
            );
            assert!(
                gap.constraint_b_idx < positions_per_constraint.len(),
                "GapPlan constraint_b_idx out of bounds"
            );
        }
        return find_spans_with_gap(positions_per_constraint, gap, doc_len);
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

/// Gap-based matching for Constraint A + Repetition(Wildcard) + Constraint B
/// Lazy: O(|A| + |B|) using two-pointer
/// Greedy: O(|A| log |B|) using binary search
fn find_spans_with_gap<'a>(
    positions_per_constraint: &[PosView<'a>],
    gap: &GapPlan,
    doc_len: Option<u32>,
) -> Vec<crate::types::SpanWithCaptures> {
    // Defensive bounds checking
    if gap.constraint_a_idx >= positions_per_constraint.len()
        || gap.constraint_b_idx >= positions_per_constraint.len()
    {
        return Vec::new();
    }

    // Extract positions for constraint A and B
    let positions_a = match &positions_per_constraint[gap.constraint_a_idx] {
        PosView::Any => return Vec::new(), // shouldn't happen for gap patterns
        PosView::List(list) => *list,
    };

    let positions_b = match &positions_per_constraint[gap.constraint_b_idx] {
        PosView::Any => return Vec::new(), // shouldn't happen for gap patterns
        PosView::List(list) => *list,
    };

    if positions_a.is_empty() || positions_b.is_empty() {
        return Vec::new();
    }

    // Helper: lower bound (first index where arr[i] >= target)
    let lower_bound = |arr: &[u32], target: u32| -> usize {
        arr.binary_search(&target).unwrap_or_else(|i| i)
    };

    // Convert max-gap to an exclusive high bound for b:
    // b_pos < a_pos + 1 + max_gap + 1
    let compute_bounds = |a_pos: u32| -> (u32, u32) {
        let min_b = a_pos
            .saturating_add(1)
            .saturating_add(gap.min_gap as u32);

        let max_b_exclusive = if let Some(max_gap) = gap.max_gap {
            a_pos
                .saturating_add(1)
                .saturating_add(max_gap as u32)
                .saturating_add(1)
        } else {
            // Unbounded: use doc_len if available, otherwise u32::MAX
            doc_len.unwrap_or(u32::MAX)
        };

        (min_b, max_b_exclusive)
    };

    let mut results = Vec::new();

    if gap.lazy {
        // --- Lazy: two-pointer O(|A| + |B|) ---
        let mut j = 0usize;

        for &a_pos in positions_a.iter() {
            let (min_b, max_b_exclusive) = compute_bounds(a_pos);

            // Advance j to the first b >= min_b
            while j < positions_b.len() && positions_b[j] < min_b {
                j += 1;
            }
            if j >= positions_b.len() {
                break; // no more B positions at all
            }

            let b_pos = positions_b[j];

            // Must satisfy exclusive upper bound as well
            if b_pos < max_b_exclusive {
                let start = a_pos as usize;
                let end = (b_pos + 1) as usize;

                if let Some(dl) = doc_len {
                    if end > dl as usize {
                        continue;
                    }
                }

                results.push(crate::types::SpanWithCaptures::new(crate::types::Span { start, end }));
            } else {
                // No valid b for this a (later b's are even larger).
                // Important: do NOT advance j here; for a larger a, this same b might become valid.
            }
        }
    } else {
        // --- Greedy: binary search O(|A| log |B|) ---
        for &a_pos in positions_a.iter() {
            let (min_b, max_b_exclusive) = compute_bounds(a_pos);

            let lo = lower_bound(positions_b, min_b);
            let hi = lower_bound(positions_b, max_b_exclusive); // first >= max_b_exclusive (exclusive end)

            if lo < hi {
                let b_pos = positions_b[hi - 1]; // farthest valid b
                let start = a_pos as usize;
                let end = (b_pos + 1) as usize;

                if let Some(dl) = doc_len {
                    if end > dl as usize {
                        continue;
                    }
                }

                results.push(crate::types::SpanWithCaptures::new(crate::types::Span { start, end }));
            }
        }
    }

    results
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
pub fn find_constraint_spans_in_sequence(
    pattern: &crate::compiler::ast::Pattern, 
    field_cache: &std::collections::HashMap<String, Vec<String>>
) -> Vec<crate::types::SpanWithCaptures> {
    use crate::compiler::ast::Pattern;
    
    if let Pattern::Concatenated(patterns) = pattern {
        let mut results = Vec::new();
        
        // Use 'word' field to determine sentence length as it's the most reliable
        let len = field_cache.get("word").or_else(|| field_cache.values().next()).map(|v| v.len()).unwrap_or(0);
        if len == 0 { return results; }

        for start in 0..len {
            let mut pos = start;
            let mut all_captures = Vec::new();
            let mut matched = true;
            
            for (idx, pat) in patterns.iter().enumerate() {
                if pos >= len {
                    matched = false;
                    break;
                }
                
                if let Some(span_with_caps) = matches_pattern_at_position(pat, field_cache, pos) {
                    pos = span_with_caps.span.end;
                    all_captures.extend(span_with_caps.captures);
                } else {
                    // Log failure for debugging
                    if patterns.len() < 4 { // Only for relatively simple patterns
                         let p_strs: Vec<String> = patterns.iter().map(|p| format!("{:?}", p)).collect();
                         let is_target = p_strs.iter().any(|s| s.contains("JJ") || s.contains("NNP") || s.contains("VBZ"));
                         if is_target {
                             println!("DEBUG: Sequence match fail. Patterns: {:?} | start: {}, failed_at_idx: {}, failed_at_pos: {}", p_strs, start, idx, pos);
                         }
                    }
                    matched = false;
                    break;
                }
            }
            
            if matched {
                let full_span = crate::types::Span { start, end: pos };
                results.push(crate::types::SpanWithCaptures::with_captures(full_span, all_captures));
            }
        }
        results
    } else {
        Vec::new()
    }
}

fn matches_pattern_at_position(
    pattern: &crate::compiler::ast::Pattern,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize
) -> Option<crate::types::SpanWithCaptures> {
    use crate::compiler::ast::Pattern;
    
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
            use crate::compiler::ast::QuantifierKind;
            
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
            use crate::compiler::ast::Assertion;
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
    constraint: &crate::compiler::ast::Constraint,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize
) -> bool {
    use crate::compiler::ast::Constraint;
    
    match constraint {
        Constraint::Wildcard => true,
        Constraint::Field { name, matcher } => {
            if let Some(tokens) = field_cache.get(name) {
                if pos < tokens.len() {
                    let result = matcher.matches(&tokens[pos]);
                    if !result && (name == "tag" || name == "word") {
                        println!("DEBUG: Match failed: field={}, pos={}, token='{}', matcher={:?}", name, pos, tokens[pos], matcher);
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
