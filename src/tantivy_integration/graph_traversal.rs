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
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use tantivy_fst::Regex;

use crate::compiler::ast::FlatPatternStep;
use crate::digraph::zero_copy::ZeroCopyGraph;
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

// Odinson-style collapsed query metrics
static SRC_DRIVER_DOCS: AtomicUsize = AtomicUsize::new(0);
static DST_DRIVER_DOCS: AtomicUsize = AtomicUsize::new(0);
static DRIVER_ALIGNMENT_DOCS: AtomicUsize = AtomicUsize::new(0);
static DRIVER_INTERSECTION_SUM: AtomicUsize = AtomicUsize::new(0);
static DRIVER_INTERSECTION_COUNT: AtomicUsize = AtomicUsize::new(0);
// Optimization: Skip prefilter when all constraints are collapsed (2-constraint patterns)
static PREFILTER_SKIPPED_ALL_COLLAPSED: AtomicUsize = AtomicUsize::new(0);
static TOKEN_EXTRACTION_SKIPPED: AtomicUsize = AtomicUsize::new(0);

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

// =============================================================================
// CandidateDriver Trait and Implementations (Odinson-style collapsed queries)
// =============================================================================

/// Default maximum number of terms to expand for regex patterns.
/// Prevents runaway memory/time on broad patterns like `.*`
pub const DEFAULT_MAX_TERM_EXPANSIONS: usize = 50;

/// Matcher for collapsed specs - supports both exact and regex patterns.
/// Uses pattern string only; term enumeration uses Tantivy's FST-based automaton,
/// not the regex crate.
#[derive(Clone, Debug)]
pub enum CollapsedMatcher {
    /// Exact term match
    Exact(String),
    /// Tantivy/Lucene-style regex pattern (matches whole term, anchored)
    RegexPattern(String),
}

/// Specification for collapsing a constraint + edge into a single driver.
/// Supports both exact string matchers and regex patterns via term enumeration.
#[derive(Clone, Debug)]
pub struct CollapsedSpec {
    /// Field for the constraint (e.g., word, entity, tag)
    pub constraint_field: Field,
    /// Matcher for the constraint (exact or regex)
    pub constraint_matcher: CollapsedMatcher,
    /// Field for the edge (incoming_edges or outgoing_edges)
    pub edge_field: Field,
    /// Matcher for the edge label (exact or regex)
    pub edge_matcher: CollapsedMatcher,
    /// Which constraint this collapses (0 for first, last for dst)
    pub constraint_idx: usize,
}

/// Abstraction for candidate document iteration with optional position access.
/// Avoids downcasting from Box<dyn Scorer> by making position-aware drivers explicit.
trait CandidateDriver: Send {
    /// Current document ID
    fn doc(&self) -> DocId;
    
    /// Advance to the next matching document
    fn advance(&mut self) -> DocId;
    
    /// If this driver is position-aware (collapsed constraint + edge),
    /// return the positions that matched (intersection). Otherwise None.
    fn matching_positions(&self) -> Option<&[u32]>;
}

/// EmptyDriver: Immediately returns TERMINATED.
/// Used when CombinedPositionDriver cannot be built (required term missing from segment)
/// or when CollapsedSpec exists but postings are unavailable.
/// This skips entire segments when required terms are missing, avoiding wasted computation.
struct EmptyDriver;

impl CandidateDriver for EmptyDriver {
    fn doc(&self) -> DocId {
        tantivy::TERMINATED
    }

    fn advance(&mut self) -> DocId {
        tantivy::TERMINATED
    }

    fn matching_positions(&self) -> Option<&[u32]> {
        None
    }
}

/// Drives iteration using two postings cursors (constraint + edge).
/// Computes position intersection at index level via skip lists.
/// This is the core of the Odinson-style collapsed query optimization.
struct CombinedPositionDriver {
    constraint_postings: SegmentPostings,
    edge_postings: SegmentPostings,
    constraint_buf: Vec<u32>,
    edge_buf: Vec<u32>,
    intersection: Vec<u32>,
    current_doc: DocId,
}

impl CombinedPositionDriver {
    fn new(constraint_postings: SegmentPostings, edge_postings: SegmentPostings) -> Self {
        // Position both postings at their first document
        // SegmentPostings starts positioned at first doc after creation
        let mut driver = Self {
            constraint_postings,
            edge_postings,
            constraint_buf: Vec::with_capacity(16),
            edge_buf: Vec::with_capacity(16),
            intersection: Vec::with_capacity(8),
            current_doc: tantivy::TERMINATED,
        };
        // Advance to first matching doc (where positions intersect)
        driver.advance_to_next_match();
        driver
    }
    
    /// Internal advance that finds next doc with overlapping positions
    fn advance_to_next_match(&mut self) -> DocId {
        loop {
            let d1 = self.constraint_postings.doc();
            let d2 = self.edge_postings.doc();
            
            // If either is exhausted, we're done
            if d1 == tantivy::TERMINATED || d2 == tantivy::TERMINATED {
                self.current_doc = tantivy::TERMINATED;
                return self.current_doc;
            }
            
            // Align to same doc using seek() (uses skip lists)
            if d1 < d2 {
                self.constraint_postings.seek(d2);
                continue;
            } else if d2 < d1 {
                self.edge_postings.seek(d1);
                continue;
            }
            
            // Same doc - compute position intersection
            self.constraint_buf.clear();
            self.edge_buf.clear();
            self.constraint_postings.positions(&mut self.constraint_buf);
            self.edge_postings.positions(&mut self.edge_buf);
            
            self.intersection.clear();
            intersect_sorted_into(&self.constraint_buf, &self.edge_buf, &mut self.intersection);
            
            let doc = d1;
            
            // Advance both for next iteration
            self.constraint_postings.advance();
            self.edge_postings.advance();
            
            if !self.intersection.is_empty() {
                self.current_doc = doc;
                log::debug!(
                    "CombinedPositionDriver: doc {} has {} overlapping positions",
                    doc, self.intersection.len()
                );
                return self.current_doc;
            }
            // No position overlap - continue to next doc
        }
    }
}

impl CandidateDriver for CombinedPositionDriver {
    fn doc(&self) -> DocId {
        self.current_doc
    }
    
    fn advance(&mut self) -> DocId {
        self.advance_to_next_match()
    }
    
    fn matching_positions(&self) -> Option<&[u32]> {
        Some(&self.intersection)
    }
}

/// Two-pointer merge of sorted position lists into output vector
fn intersect_sorted_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
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
}

// =============================================================================
// Regex Support: Term Expansion and Union Position Iteration
// =============================================================================

/// Helper function to expand terms using a cached regex automaton.
/// Extracted to avoid code duplication and enable caching.
fn expand_with_automaton(
    term_dict: &tantivy::termdict::TermDictionary,
    automaton: &Regex,
    field: Field,
    inverted_index: &tantivy::InvertedIndexReader,
    max_expansions: usize,
) -> Option<Vec<SegmentPostings>> {
    let mut stream = match term_dict.search(automaton).into_stream() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Failed to search term dict: {:?}", e);
            return None;
        }
    };
    
    let mut postings_list = Vec::new();
    let mut count = 0;
    
    while stream.advance() {
        if count >= max_expansions {
            log::warn!(
                "collapse regex disabled: expanded_terms={} > max={}",
                count, max_expansions
            );
            return None;
        }
        
        let term_bytes = stream.key();
        let term = Term::from_field_bytes(field, term_bytes);
        if let Ok(Some(postings)) = inverted_index
            .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
        {
            postings_list.push(postings);
        }
        count += 1;
    }
    
    if postings_list.is_empty() {
        log::debug!("Regex matched 0 terms in segment");
        return None;
    }
    
    log::debug!("Regex expanded to {} terms", postings_list.len());
    Some(postings_list)
}

/// Get or compile a regex automaton from the cache (thread-safe).
fn get_or_compile_regex(
    cache: &Arc<RwLock<HashMap<String, Arc<Regex>>>>,
    pattern: &str,
) -> Option<Arc<Regex>> {
    // Fast path: read lock
    {
        let read_guard = cache.read().ok()?;
        if let Some(regex) = read_guard.get(pattern) {
            return Some(Arc::clone(regex));
        }
    } // Read lock released here
    
    // Slow path: write lock
    let mut write_guard = cache.write().ok()?;
    
    // Double-check after acquiring write lock (another thread might have compiled it)
    if let Some(regex) = write_guard.get(pattern) {
        return Some(Arc::clone(regex));
    }
    
    // Compile and cache
    let regex = match Regex::new(pattern) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            log::warn!("Invalid regex pattern '{}': {}", pattern, e);
            return None;
        }
    };
    write_guard.insert(pattern.to_string(), Arc::clone(&regex));
    Some(regex)
}

/// Expand a CollapsedMatcher to Vec<SegmentPostings>.
/// Returns None if:
/// - Field doesn't support positions
/// - Regex exceeds max_expansions (logs warning and bails out)
/// - No matching terms in segment
fn expand_matcher(
    reader: &SegmentReader,
    field: Field,
    matcher: &CollapsedMatcher,
    max_expansions: usize,
    regex_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
) -> Option<Vec<SegmentPostings>> {
    let inverted_index = reader.inverted_index(field).ok()?;
    
    match matcher {
        CollapsedMatcher::Exact(term_str) => {
            // Single term lookup (fast path)
            let term = Term::from_field_text(field, term_str);
            let postings = inverted_index
                .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
                .ok()??;
            Some(vec![postings])
        }
        CollapsedMatcher::RegexPattern(pattern) => {
            // Automaton-based term enumeration
            let term_dict = inverted_index.terms();
            
            // Get or compile regex automaton from cache (thread-safe)
            let automaton = get_or_compile_regex(&regex_cache, pattern)?;
            
            // Expand terms using the cached automaton
            expand_with_automaton(&term_dict, automaton.as_ref(), field, &inverted_index, max_expansions)
        }
    }
}

/// Entry in the min-heap for k-way merge of postings
struct PostingsEntry {
    doc: DocId,
    idx: usize,  // Index into postings vec
}

impl Ord for PostingsEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller doc first (reverse comparison)
        other.doc.cmp(&self.doc)
    }
}

impl PartialOrd for PostingsEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PostingsEntry {}

impl PartialEq for PostingsEntry {
    fn eq(&self, other: &Self) -> bool {
        self.doc == other.doc
    }
}

/// Union iterator over multiple postings lists.
/// Yields (doc_id, merged_positions) for each doc that appears in any postings.
/// Uses heap-based k-way merge for efficient iteration.
struct UnionPositionsIterator {
    postings: Vec<SegmentPostings>,
    heap: BinaryHeap<PostingsEntry>,
    current_doc: DocId,
    merged_positions: Vec<u32>,
    position_buf: Vec<u32>,
}

impl UnionPositionsIterator {
    fn new(postings: Vec<SegmentPostings>) -> Self {
        let mut heap = BinaryHeap::with_capacity(postings.len());
        
        // Initialize heap with first doc from each postings
        for (idx, p) in postings.iter().enumerate() {
            let doc = p.doc();
            if doc != tantivy::TERMINATED {
                heap.push(PostingsEntry { doc, idx });
            }
        }
        
        let mut iter = Self {
            postings,
            heap,
            current_doc: tantivy::TERMINATED,
            merged_positions: Vec::with_capacity(32),
            position_buf: Vec::with_capacity(16),
        };
        
        // Advance to first doc
        iter.advance_to_next_doc();
        iter
    }
    
    fn doc(&self) -> DocId {
        self.current_doc
    }
    
    fn positions(&self) -> &[u32] {
        &self.merged_positions
    }
    
    fn advance_to_next_doc(&mut self) -> DocId {
        self.merged_positions.clear();
        
        if self.heap.is_empty() {
            self.current_doc = tantivy::TERMINATED;
            return self.current_doc;
        }
        
        // Get minimum doc
        let min_doc = self.heap.peek().unwrap().doc;
        self.current_doc = min_doc;
        
        // Collect positions from all postings at min_doc directly into merged_positions
        while let Some(entry) = self.heap.peek() {
            if entry.doc != min_doc {
                break;
            }
            
            let entry = self.heap.pop().unwrap();
            let postings = &mut self.postings[entry.idx];
            
            // Get positions into temp buffer, then extend merged_positions
            self.position_buf.clear();
            postings.positions(&mut self.position_buf);
            self.merged_positions.extend_from_slice(&self.position_buf);
            
            // Advance this postings and re-insert if not exhausted
            let next_doc = postings.advance();
            if next_doc != tantivy::TERMINATED {
                self.heap.push(PostingsEntry { doc: next_doc, idx: entry.idx });
            }
        }
        
        // Sort and dedup the merged positions
        if !self.merged_positions.is_empty() {
            self.merged_positions.sort_unstable();
            self.merged_positions.dedup();
        }
        
        self.current_doc
    }
}

/// Driver that combines two UnionPositionsIterators with position intersection.
/// This is the regex-capable version of CombinedPositionDriver.
/// Implements the Lucene "SpanMultiTermQueryWrapper + SpanNear/AND" pattern.
struct UnionAndIntersectDriver {
    lhs: UnionPositionsIterator,  // Constraint side
    rhs: UnionPositionsIterator,  // Edge side
    current_doc: DocId,
    intersection: Vec<u32>,
}

impl UnionAndIntersectDriver {
    fn new(lhs: UnionPositionsIterator, rhs: UnionPositionsIterator) -> Self {
        let mut driver = Self {
            lhs,
            rhs,
            current_doc: tantivy::TERMINATED,
            intersection: Vec::with_capacity(16),
        };
        driver.advance_to_next_match();
        driver
    }
    
    fn advance_to_next_match(&mut self) -> DocId {
        loop {
            let d1 = self.lhs.doc();
            let d2 = self.rhs.doc();
            
            if d1 == tantivy::TERMINATED || d2 == tantivy::TERMINATED {
                self.current_doc = tantivy::TERMINATED;
                return self.current_doc;
            }
            
            if d1 < d2 {
                self.lhs.advance_to_next_doc();
                continue;
            } else if d2 < d1 {
                self.rhs.advance_to_next_doc();
                continue;
            }
            
            // Same doc - intersect positions
            self.intersection.clear();
            intersect_sorted_into(
                self.lhs.positions(),
                self.rhs.positions(),
                &mut self.intersection
            );
            
            let doc = d1;
            
            // Advance both for next iteration
            self.lhs.advance_to_next_doc();
            self.rhs.advance_to_next_doc();
            
            if !self.intersection.is_empty() {
                self.current_doc = doc;
                return self.current_doc;
            }
            // No position overlap - continue
        }
    }
}

impl CandidateDriver for UnionAndIntersectDriver {
    fn doc(&self) -> DocId {
        self.current_doc
    }
    
    fn advance(&mut self) -> DocId {
        self.advance_to_next_match()
    }
    
    fn matching_positions(&self) -> Option<&[u32]> {
        Some(&self.intersection)
    }
}

// =============================================================================
// End CandidateDriver Implementation
// =============================================================================

// Optimized graph traversal query using Odinson-style collapsed query optimization.
// Candidate generation is driven exclusively by CombinedPositionDriver - no BooleanQuery pre-filtering.
#[derive(Debug)]
pub struct OptimizedGraphTraversalQuery {
    #[allow(dead_code)]
    default_field: Field,
    dependencies_binary_field: Field,
    incoming_edges_field: Field,
    outgoing_edges_field: Field,
    traversal: crate::compiler::ast::Traversal,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
    /// Collapse spec for src constraint (first) - enables CombinedPositionDriver
    src_collapse: Option<CollapsedSpec>,
    /// Collapse spec for dst constraint (last) - enables CombinedPositionDriver
    dst_collapse: Option<CollapsedSpec>,
}


impl OptimizedGraphTraversalQuery {
    /// Create a new query using only collapse specs for candidate generation.
    /// No BooleanQuery pre-filtering - relies exclusively on CombinedPositionDriver.
    ///
    /// If neither src nor dst can be collapsed, EmptyDriver is used (returns no results).
    pub fn collapsed_only(
        default_field: Field,
        dependencies_binary_field: Field,
        incoming_edges_field: Field,
        outgoing_edges_field: Field,
        traversal: crate::compiler::ast::Traversal,
        src_pattern: crate::compiler::ast::Pattern,
        dst_pattern: crate::compiler::ast::Pattern,
        src_collapse: Option<CollapsedSpec>,
        dst_collapse: Option<CollapsedSpec>,
    ) -> Self {
        Self {
            default_field,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
            traversal,
            src_pattern,
            dst_pattern,
            src_collapse,
            dst_collapse,
        }
    }
}

impl Query for OptimizedGraphTraversalQuery {

    fn weight(&self, _scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        // Odinson-style: No BooleanQuery weights - use CombinedPositionDriver exclusively
        log::info!(
            "Creating collapsed-only weight: src_collapse={}, dst_collapse={}",
            self.src_collapse.is_some(),
            self.dst_collapse.is_some()
        );

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
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
            traversal: self.traversal.clone(),
            dependencies_binary_field: self.dependencies_binary_field,
            incoming_edges_field: self.incoming_edges_field,
            outgoing_edges_field: self.outgoing_edges_field,
            flat_steps,
            prefilter_plan,
            src_collapse: self.src_collapse.clone(),
            dst_collapse: self.dst_collapse.clone(),
            regex_cache: Arc::new(RwLock::new(HashMap::<String, Arc<Regex>>::new())),
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
            traversal: self.traversal.clone(),
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
            src_collapse: self.src_collapse.clone(),
            dst_collapse: self.dst_collapse.clone(),
        })
    }
}

/// Optimized weight for graph traversal queries using Odinson-style collapsed optimization.
/// No BooleanQuery weights - candidate generation driven exclusively by CombinedPositionDriver.
struct OptimizedGraphTraversalWeight {
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
    /// Collapse spec for src constraint - required for CombinedPositionDriver
    src_collapse: Option<CollapsedSpec>,
    /// Collapse spec for dst constraint - required for CombinedPositionDriver
    dst_collapse: Option<CollapsedSpec>,
    /// Cached compiled regex automata (shared across segments, thread-safe)
    regex_cache: Arc<RwLock<HashMap<String, Arc<Regex>>>>,
}

impl CollapsedMatcher {
    /// Format matcher for logging
    pub fn display(&self) -> String {
        match self {
            CollapsedMatcher::Exact(s) => format!("'{}'", s),
            CollapsedMatcher::RegexPattern(p) => format!("/{}/", p),
        }
    }
}

impl OptimizedGraphTraversalWeight {
    /// Build a CandidateDriver from a CollapsedSpec.
    /// Handles both exact matches (fast path) and regex patterns (term enumeration).
    /// 
    /// Returns None if:
    /// - Postings don't exist (term not in segment)
    /// - Regex expansion exceeds max_expansions limit
    /// - No matching terms for regex pattern
    fn build_combined_driver(
        &self,
        reader: &SegmentReader,
        spec: &CollapsedSpec,
    ) -> Option<Box<dyn CandidateDriver>> {
        // Expand constraint matcher
        let constraint_postings = expand_matcher(
            reader,
            spec.constraint_field,
            &spec.constraint_matcher,
            DEFAULT_MAX_TERM_EXPANSIONS,
            self.regex_cache.clone(),
        )?;
        
        // Expand edge matcher
        let edge_postings = expand_matcher(
            reader,
            spec.edge_field,
            &spec.edge_matcher,
            DEFAULT_MAX_TERM_EXPANSIONS,
            self.regex_cache.clone(),
        )?;
        
        // Fast path: both exact (single postings each) - use CombinedPositionDriver
        if constraint_postings.len() == 1 && edge_postings.len() == 1 {
            log::info!(
                "Built CombinedPositionDriver (fast path) for constraint={} edge={}",
                spec.constraint_matcher.display(), spec.edge_matcher.display()
            );
            return Some(Box::new(CombinedPositionDriver::new(
                constraint_postings.into_iter().next().unwrap(),
                edge_postings.into_iter().next().unwrap(),
            )));
        }
        
        // Regex path: use UnionAndIntersectDriver
        log::info!(
            "Built UnionAndIntersectDriver for constraint={} ({} postings) edge={} ({} postings)",
            spec.constraint_matcher.display(), constraint_postings.len(),
            spec.edge_matcher.display(), edge_postings.len()
        );
        
        let lhs = UnionPositionsIterator::new(constraint_postings);
        let rhs = UnionPositionsIterator::new(edge_postings);
        
        Some(Box::new(UnionAndIntersectDriver::new(lhs, rhs)))
    }
}

impl Weight for OptimizedGraphTraversalWeight {

    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        // Odinson-style: Build drivers exclusively from collapse specs
        // No GenericDriver fallback - use EmptyDriver when postings unavailable

        let src_driver: Box<dyn CandidateDriver> = if let Some(ref spec) = self.src_collapse {
            if let Some(driver) = self.build_combined_driver(reader, spec) {
                log::info!("Using driver for src (constraint={} edge={})",
                    spec.constraint_matcher.display(), spec.edge_matcher.display());
                driver
            } else {
                log::info!("Using EmptyDriver for src (postings unavailable in segment)");
                Box::new(EmptyDriver)
            }
        } else {
            log::info!("Using EmptyDriver for src (no collapse spec - pattern not collapsible)");
            Box::new(EmptyDriver)
        };

        let dst_driver: Box<dyn CandidateDriver> = if let Some(ref spec) = self.dst_collapse {
            if let Some(driver) = self.build_combined_driver(reader, spec) {
                log::info!("Using driver for dst (constraint={} edge={})",
                    spec.constraint_matcher.display(), spec.edge_matcher.display());
                driver
            } else {
                log::info!("Using EmptyDriver for dst (postings unavailable in segment)");
                Box::new(EmptyDriver)
            }
        } else {
            log::info!("Using EmptyDriver for dst (no collapse spec - pattern not collapsible)");
            Box::new(EmptyDriver)
        };

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
            src_driver,
            dst_driver,
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
            // Store collapse specs for position handoff
            src_collapse: self.src_collapse.clone(),
            dst_collapse: self.dst_collapse.clone(),
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
/// Uses CandidateDriver abstraction for Odinson-style collapsed query optimization
pub struct OptimizedGraphTraversalScorer {
    /// Source candidate driver (may be CombinedPositionDriver or GenericDriver)
    src_driver: Box<dyn CandidateDriver>,
    /// Destination candidate driver (may be CombinedPositionDriver or GenericDriver)
    dst_driver: Box<dyn CandidateDriver>,
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
    /// Optional collapse spec for src (for position handoff)
    src_collapse: Option<CollapsedSpec>,
    /// Optional collapse spec for dst (for position handoff)
    dst_collapse: Option<CollapsedSpec>,
}

impl OptimizedGraphTraversalScorer {
    /// Helper method to run traversal with any GraphTraversal<G: GraphAccess>
    /// This allows us to use ZeroCopyGraph directly without conversion
    fn run_traversal_with_engine<G: crate::digraph::graph_trait::GraphAccess>(
        &mut self,
        traversal_engine: &crate::digraph::traversal::GraphTraversal<G>,
        flat_steps: &[FlatPatternStep],
        src_positions: &[usize],
        lazy_tokens: &mut LazyConstraintTokens,
        allowed_positions_hashset: &[Option<std::collections::HashSet<u32>>],
        constraint_exact_flags: &[bool],
    ) -> bool {
        // Extract constraint_field_names to avoid borrowing self
        let constraint_field_names = &self.constraint_field_names;
        
        // Create closure for token access (wraps lazy_tokens)
        let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
            lazy_tokens.get(constraint_idx, position)
        };
        
        let mut all_matches = Vec::new();
        
        for &src_pos in src_positions {
            let all_paths = traversal_engine.automaton_query_paths(
                flat_steps, 
                &[src_pos], 
                constraint_field_names, 
                &mut get_token,
                allowed_positions_hashset,
                constraint_exact_flags,
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
                    all_matches.push(
                        crate::types::SpanWithCaptures::with_captures(
                            crate::types::Span { start: min_pos, end: max_pos + 1 },
                            captures
                        )
                    );
                }
            }
            
            if !all_paths.is_empty() {
                // Now we can mutate self after the closure is dropped
                self.current_doc_matches.extend(all_matches);
                return true;
            }
        }
        
        false
    }

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
            // Use drivers instead of scorers (Odinson-style optimization)
            let src_doc = self.src_driver.doc();
            let dst_doc = self.dst_driver.doc();

            // If either driver is exhausted, we're done
            if src_doc == tantivy::TERMINATED || dst_doc == tantivy::TERMINATED {
                self.current_doc = None;
                debug!("advance() terminated: src_doc = {}, dst_doc = {}", src_doc, dst_doc);

                // Log final stats when driver is exhausted (using module-level statics)
                let call_num = CALL_COUNT.load(Ordering::Relaxed);
                if call_num == 0 {
                    log::warn!(
                        "NO CANDIDATES FOUND! Drivers returned 0 matching documents. \
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
                    
                    let src_driver_docs = SRC_DRIVER_DOCS.load(Ordering::Relaxed);
                    let dst_driver_docs = DST_DRIVER_DOCS.load(Ordering::Relaxed);
                    let alignment_docs = DRIVER_ALIGNMENT_DOCS.load(Ordering::Relaxed);
                    let intersection_sum = DRIVER_INTERSECTION_SUM.load(Ordering::Relaxed);
                    let intersection_count = DRIVER_INTERSECTION_COUNT.load(Ordering::Relaxed);
                    let avg_intersection = if intersection_count > 0 {
                        intersection_sum as f64 / intersection_count as f64
                    } else {
                        0.0
                    };
                    
                    let prefilter_skipped_collapsed = PREFILTER_SKIPPED_ALL_COLLAPSED.load(Ordering::Relaxed);
                    let token_extraction_skipped = TOKEN_EXTRACTION_SKIPPED.load(Ordering::Relaxed);
                    
                    log::info!(
                        "FINAL Graph traversal stats: {} candidates checked, {} graphs deserialized ({} skipped, {:.1}% skip rate)",
                        call_num, deser_count, skipped_count, skip_rate
                    );
                    log::info!(
                        "FINAL Prefilter stats: {} docs checked, {} killed by prefilter ({:.1}% kill rate), avg allowed positions per constraint: {:.1}",
                        prefilter_docs, prefilter_killed, prefilter_kill_rate, avg_allowed_pos
                    );
                    log::info!(
                        "FINAL Optimization: {} prefilter calls skipped (all-collapsed), {} token extractions skipped",
                        prefilter_skipped_collapsed, token_extraction_skipped
                    );
                    log::info!(
                        "FINAL Odinson driver stats: src_driver={} docs, dst_driver={} docs, aligned={} docs, avg intersection size={:.1}",
                        src_driver_docs, dst_driver_docs, alignment_docs, avg_intersection
                    );
                }
                
                return tantivy::TERMINATED;
            }
            debug!("advance() considering src_doc = {}, dst_doc = {}", src_doc, dst_doc);
            if src_doc < dst_doc {
                SRC_DRIVER_DOCS.fetch_add(1, Ordering::Relaxed);
                self.src_driver.advance();
            } else if dst_doc < src_doc {
                DST_DRIVER_DOCS.fetch_add(1, Ordering::Relaxed);
                self.dst_driver.advance();
            } else {
                // src_doc == dst_doc: both drivers have matches in this doc
                let doc_id = src_doc;
                DRIVER_ALIGNMENT_DOCS.fetch_add(1, Ordering::Relaxed);
                
                // Track intersection sizes for metrics
                if let Some(src_pos) = self.src_driver.matching_positions() {
                    DRIVER_INTERSECTION_SUM.fetch_add(src_pos.len(), Ordering::Relaxed);
                    DRIVER_INTERSECTION_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(dst_pos) = self.dst_driver.matching_positions() {
                    DRIVER_INTERSECTION_SUM.fetch_add(dst_pos.len(), Ordering::Relaxed);
                    DRIVER_INTERSECTION_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                
                debug!("advance() found candidate doc_id = {}", doc_id);
                if self.check_graph_traversal(doc_id) {
                    debug!("advance() doc_id {} MATCHED graph traversal", doc_id);
                    self.current_doc = Some(doc_id);
                    // Compute Odinson-style score based on span widths and match count
                    let score = self.compute_odinson_score();
                    debug!("Odinson-style score for doc_id {}: {}", doc_id, score);
                    self.current_matches.push((doc_id, score));
                    self.match_index = self.current_matches.len() - 1;
                    // Advance both drivers for next call
                    self.src_driver.advance();
                    self.dst_driver.advance();
                    return doc_id;
                } else {
                    debug!("advance() doc_id {} did NOT match graph traversal", doc_id);
                    // No match, advance both drivers
                    self.src_driver.advance();
                    self.dst_driver.advance();
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

/// Lazy constraint token loader - parses constraint fields only when first accessed
/// This avoids parsing constraint fields that are never reached during graph traversal
struct LazyConstraintTokens<'a> {
    doc: &'a tantivy::schema::TantivyDocument,
    constraint_field_names: &'a [String],
    schema: &'a tantivy::schema::Schema,
    cache: Vec<Option<Vec<String>>>,  // One Option per constraint index
}

impl<'a> LazyConstraintTokens<'a> {
    /// Create a new lazy token loader
    fn new(
        doc: &'a tantivy::schema::TantivyDocument,
        constraint_field_names: &'a [String],
        schema: &'a tantivy::schema::Schema,
    ) -> Self {
        let cache = vec![None; constraint_field_names.len()];
        Self {
            doc,
            constraint_field_names,
            schema,
            cache,
        }
    }

    /// Ensure tokens for a constraint are loaded (parses on first access)
    fn ensure_loaded(&mut self, constraint_idx: usize) {
        if constraint_idx >= self.cache.len() {
            return;
        }
        
        if self.cache[constraint_idx].is_none() {
            let field_name = &self.constraint_field_names[constraint_idx];
            let tokens = self.extract_tokens_from_field(field_name);
            self.cache[constraint_idx] = Some(tokens);
        }
    }


    /// Get token at a specific position for a constraint (returns owned String)
    /// Parses the constraint field on first access
    fn get(&mut self, constraint_idx: usize, position: usize) -> Option<String> {
        self.ensure_loaded(constraint_idx);
        self.cache[constraint_idx]
            .as_ref()
            .and_then(|tokens| tokens.get(position))
            .cloned()
    }

    /// Get all tokens for a constraint (returns reference after ensuring loaded)
    /// Used for position calculation that needs full token list
    fn get_all_tokens(&mut self, constraint_idx: usize) -> Option<&[String]> {
        self.ensure_loaded(constraint_idx);
        self.cache[constraint_idx].as_ref().map(|tokens| tokens.as_slice())
    }

    /// Extract tokens from a field (reuses existing logic)
    fn extract_tokens_from_field(&self, field_name: &str) -> Vec<String> {
        crate::tantivy_integration::utils::extract_field_values(self.schema, self.doc, field_name)
    }
}

/// Check if a constraint is exact and can skip matches() when prefilter confirms
/// Only simple Field { Matcher::String } constraints are skippable
fn is_exact_skippable(constraint: &crate::compiler::ast::Constraint) -> bool {
    match constraint {
        crate::compiler::ast::Constraint::Field { 
            matcher: crate::compiler::ast::Matcher::String(_), 
            .. 
        } => true,
        crate::compiler::ast::Constraint::Field {
            matcher: crate::compiler::ast::Matcher::Regex { .. },
            ..
        } => false,  // Regex constraints need actual matching
        crate::compiler::ast::Constraint::Negated(_) 
        | crate::compiler::ast::Constraint::Conjunctive(_) 
        | crate::compiler::ast::Constraint::Disjunctive(_) 
        | crate::compiler::ast::Constraint::Wildcard 
        | crate::compiler::ast::Constraint::Fuzzy { .. } => false,
    }
}

impl OptimizedGraphTraversalScorer {
    /// Build allowed positions combining src_driver, dst_driver, and prefilter positions
    /// Converts all to HashSet<u32> for O(1) lookup
    fn build_allowed_positions(
        &self,
        src_driver_positions: Option<&[u32]>,
        dst_driver_positions: Option<&[u32]>,
        prefilter_positions: &[Option<Vec<u32>>],
        num_constraints: usize,
    ) -> Vec<Option<std::collections::HashSet<u32>>> {
        use std::collections::HashSet;
        
        let mut result: Vec<Option<HashSet<u32>>> = vec![None; num_constraints];
        
        // First constraint: from src_driver (if available)
        if let Some(src_positions) = src_driver_positions {
            if num_constraints > 0 {
                result[0] = Some(src_positions.iter().copied().collect());
            }
        }
        
        // Last constraint: from dst_driver (if available)
        if let Some(dst_positions) = dst_driver_positions {
            let last_idx = num_constraints.saturating_sub(1);
            if last_idx > 0 && last_idx < num_constraints {
                result[last_idx] = Some(dst_positions.iter().copied().collect());
            }
        }
        
        // Middle constraints: from prefilter
        for (idx, prefilter) in prefilter_positions.iter().enumerate() {
            if idx < num_constraints {
                // Only set if not already set by driver (first/last)
                if result[idx].is_none() {
                    if let Some(positions) = prefilter {
                        result[idx] = Some(positions.iter().copied().collect());
                    }
                }
            }
        }
        
        result
    }

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
    /// Compute allowed positions for each constraint, using driver positions for collapsed constraints
    /// to avoid duplicate work (Odinson optimization).
    /// 
    /// For collapsed constraints (0 and last), uses driver positions directly instead of
    /// recomputing constraint+edge intersection.
    fn compute_allowed_positions(
        &mut self,
        doc_id: DocId,
        src_driver_positions: Option<&[u32]>,
        dst_driver_positions: Option<&[u32]>,
    ) -> Option<Vec<Option<Vec<u32>>>> {
        // Start as "no restriction" for each constraint
        let mut allowed: Vec<Option<Vec<u32>>> = vec![None; self.prefilter_plan.num_constraints];
        
        // Determine which constraints are collapsed (to skip duplicate work)
        let src_collapsed_idx = self.src_collapse.as_ref().map(|s| s.constraint_idx);
        let dst_collapsed_idx = self.dst_collapse.as_ref().map(|s| s.constraint_idx);
        let last_constraint_idx = self.prefilter_plan.num_constraints.saturating_sub(1);
        let dst_is_last = dst_collapsed_idx.map(|idx| idx == last_constraint_idx).unwrap_or(false);

        // Use driver positions directly for collapsed constraints (avoid duplicate work)
        if let Some(src_positions) = src_driver_positions {
            if let Some(idx) = src_collapsed_idx {
                allowed[idx] = Some(src_positions.to_vec());
                log::debug!("Using src_driver positions for constraint {} (skipping recomputation)", idx);
            }
        }
        if let Some(dst_positions) = dst_driver_positions {
            if dst_is_last {
                if let Some(idx) = dst_collapsed_idx {
                    allowed[idx] = Some(dst_positions.to_vec());
                    log::debug!("Using dst_driver positions for constraint {} (skipping recomputation)", idx);
                }
            }
        }

        let mut buf: Vec<u32> = Vec::with_capacity(32);

        // Phase 1: Process edge requirements (skip edges for collapsed constraints)
        for (req_idx, req) in self.prefilter_plan.edge_reqs.iter().enumerate() {
            // Skip if this edge requirement is for a collapsed constraint
            if Some(req.constraint_idx) == src_collapsed_idx || Some(req.constraint_idx) == dst_collapsed_idx {
                log::debug!("Skipping edge requirement for collapsed constraint {}", req.constraint_idx);
                continue;
            }
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
                    // First restriction: take positions (move instead of clone)
                    allowed[req.constraint_idx] = Some(std::mem::take(&mut buf));
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
        // Skip collapsed constraints - they already have driver positions set
        for (req_idx, req) in self.constraint_reqs.iter().enumerate() {
            // Skip if this constraint is collapsed (already has driver positions)
            if Some(req.constraint_idx) == src_collapsed_idx || Some(req.constraint_idx) == dst_collapsed_idx {
                log::debug!("Skipping constraint requirement for collapsed constraint {}", req.constraint_idx);
                continue;
            }
            
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
                    // No edge restriction yet - take constraint positions (move instead of clone)
                    allowed[req.constraint_idx] = Some(std::mem::take(&mut buf));
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
    /// 5. Uses driver matching_positions for first/last constraints (Odinson-style position handoff)
    fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {
        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        self.current_doc_matches.clear();


        // Get driver positions for first/last constraints (Odinson-style position handoff)
        // These are already filtered for position overlap between constraint and edge
        // Clone positions to avoid borrow checker issues (they're small Vec<u32>)
        let src_driver_positions = self.src_driver.matching_positions().map(|p| p.to_vec());
        let dst_driver_positions = self.dst_driver.matching_positions().map(|p| p.to_vec());
        
        if src_driver_positions.is_some() {
            log::debug!("Using src_driver positions for constraint 0 (Odinson position handoff)");
        }
        if dst_driver_positions.is_some() {
            log::debug!("Using dst_driver positions for last constraint (Odinson position handoff)");
        }

        // OPTIMIZATION: Detect if ALL constraints are collapsed (2-constraint pattern)
        // When both src and dst are collapsed, compute_allowed_positions is redundant
        // because driver positions already contain filtered constraint+edge intersections
        let num_constraints = self.prefilter_plan.num_constraints;
        let all_collapsed = num_constraints == 2 
            && self.src_collapse.is_some() 
            && self.dst_collapse.is_some()
            && src_driver_positions.is_some()
            && dst_driver_positions.is_some();

        // Phase 0: Postings prefilter (before any store access)
        // Skip entirely for all-collapsed patterns (positions already computed by drivers)
        PREFILTER_DOCS.fetch_add(1, Ordering::Relaxed);
        let allowed_positions = if all_collapsed {
            // OPTIMIZATION: Skip compute_allowed_positions entirely
            // Driver positions are already filtered for constraint+edge intersection
            PREFILTER_SKIPPED_ALL_COLLAPSED.fetch_add(1, Ordering::Relaxed);
            log::debug!("Skipping prefilter: all {} constraints collapsed", num_constraints);
            
            let mut allowed: Vec<Option<Vec<u32>>> = vec![None; num_constraints];
            if let Some(ref positions) = src_driver_positions {
                allowed[0] = Some(positions.clone());
            }
            if let Some(ref positions) = dst_driver_positions {
                allowed[num_constraints - 1] = Some(positions.clone());
            }
            allowed
        } else {
            // Standard path: use prefilter for middle constraints or non-collapsed patterns
            match self.compute_allowed_positions(
                doc_id,
                src_driver_positions.as_deref(),
                dst_driver_positions.as_deref(),
            ) {
                Some(ap) => ap,
                None => {
                    PREFILTER_KILLED.fetch_add(1, Ordering::Relaxed);
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
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

        // Count total constraints to identify last constraint index
        let total_constraints = flat_steps.iter()
            .filter(|s| matches!(s, FlatPatternStep::Constraint(_)))
            .count();

        // Phase 1: Load stored document (only survivors reach here)
        let doc = match self.store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => return false,
        };

        // Phase 2: Create lazy token loader (parses on first access per constraint)
        // OPTIMIZATION: Avoids parsing constraint fields that are never reached during traversal
        let mut lazy_tokens = LazyConstraintTokens::new(
            &doc,
            &self.constraint_field_names,
            self.reader.schema(),
        );

        // Build allowed positions as HashSet for O(1) lookup
        let allowed_positions_hashset = self.build_allowed_positions(
            src_driver_positions.as_deref(),
            dst_driver_positions.as_deref(),
            &allowed_positions,
            num_constraints,
        );

        // Build constraint_exact_flags by analyzing flat_steps
        let mut constraint_idx_counter = 0;
        let constraint_exact_flags: Vec<bool> = flat_steps.iter()
            .filter_map(|step| {
                if let FlatPatternStep::Constraint(constraint_pat) = step {
                    let unwrapped = self.unwrap_constraint_pattern(constraint_pat);
                    if let Pattern::Constraint(constraint) = unwrapped {
                        let is_exact = is_exact_skippable(constraint);
                        constraint_idx_counter += 1;
                        Some(is_exact)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Phase 3: Check constraints with position restrictions
        // Odinson position handoff: use driver positions for first (constraint 0) and last constraint
        // NOTE: We still need to compute cached_positions for src_positions used in Phase 5
        // But we can use lazy_tokens for the actual traversal
        let mut constraint_count = 0;
        let mut cached_positions: Vec<Vec<usize>> = Vec::new();

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(constraint_pat) = step {
                if constraint_count >= self.constraint_field_names.len() {
                    GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                    return false;
                }

                // For cached_positions, we still need tokens (used for src_positions)
                // But the actual traversal will use lazy loading via closure
                let tokens = match lazy_tokens.get_all_tokens(constraint_count) {
                    Some(t) => t,
                    None => {
                        GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                };
                
                let unwrapped = self.unwrap_constraint_pattern(constraint_pat);
                
                let is_wildcard = matches!(
                    unwrapped,
                    Pattern::Constraint(crate::compiler::ast::Constraint::Wildcard)
                );

                // Odinson position handoff: check if we can use driver positions from allowed_positions
                // (which were set directly from drivers, avoiding duplicate work)
                let is_first_constraint = constraint_count == 0;
                let is_last_constraint = constraint_count == total_constraints - 1;
                
                let positions = if is_first_constraint && self.src_collapse.is_some() {
                    // Use allowed_positions[0] which was set from src_driver (already filtered by constraint + edge)
                    // This skips token scanning entirely for constraint 0
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        log::debug!("Skipping token scan for constraint 0, using {} driver positions", allowed.len());
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Fallback: shouldn't happen if driver worked correctly
                        self.find_positions_in_tokens(tokens, constraint_pat)
                    }
                } else if is_last_constraint && self.dst_collapse.is_some() {
                    // Use allowed_positions[last] which was set from dst_driver (already filtered by constraint + edge)
                    // This skips token scanning entirely for last constraint
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
                        log::debug!("Skipping token scan for last constraint, using {} driver positions", allowed.len());
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Fallback: shouldn't happen if driver worked correctly
                        self.find_positions_in_tokens(tokens, constraint_pat)
                    }
                } else if is_wildcard {
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
        
        // Phase 5: Run traversal
        let src_positions: &[usize] = cached_positions.get(0).map(|v| v.as_slice()).unwrap_or(&[]);
        if constraint_count > 0 && src_positions.is_empty() {
            return false;
        }

        // Extract values we need before mutable borrow
        let constraint_field_names = &self.constraint_field_names;

        // Use zero-copy format only - no fallback to legacy format
        if !ZeroCopyGraph::is_valid_format(binary_data) {
            log::error!("Query-time: Invalid graph format - expected zero-copy format (magic number mismatch). Skipping document.");
            GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let traversal_result = match ZeroCopyGraph::from_bytes(binary_data) {
            Ok(zc_graph) => {
                // TRUE ZERO-COPY: Use ZeroCopyGraph directly without conversion
                use crate::digraph::graph_trait::GraphAccess;
                log::debug!("ZeroCopyGraph loaded successfully: {} nodes, {} labels", 
                    zc_graph.node_count(), zc_graph.label_count());
                let traversal_engine = crate::digraph::traversal::GraphTraversal::new(zc_graph);
                // Create closure for token access (wraps lazy_tokens)
                let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
                    lazy_tokens.get(constraint_idx, position)
                };
                
                let mut all_matches = Vec::new();
                
                for &src_pos in src_positions {
                    let all_paths = traversal_engine.automaton_query_paths(
                        flat_steps, 
                        &[src_pos], 
                        constraint_field_names, 
                        &mut get_token,
                        &allowed_positions_hashset,
                        &constraint_exact_flags,
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
                            all_matches.push(
                                crate::types::SpanWithCaptures::with_captures(
                                    crate::types::Span { start: min_pos, end: max_pos + 1 },
                                    captures
                                )
                            );
                        }
                    }
                    
                    if !all_paths.is_empty() {
                        // Now we can mutate self after the closure is dropped
                        self.current_doc_matches.extend(all_matches);
                        return true;
                    }
                }
                
                false
            }
            Err(e) => {
                log::error!("Query-time: ZeroCopyGraph::from_bytes failed: {:?}. Skipping document.", e);
                GRAPH_DESER_SKIPPED.fetch_add(1, Ordering::Relaxed);
                false
            }
        };

        // Log stats periodically
        if call_num > 0 && call_num % 100 == 0 {
            let deser = GRAPH_DESER_COUNT.load(Ordering::Relaxed);
            let skipped = GRAPH_DESER_SKIPPED.load(Ordering::Relaxed);
            let pf_docs = PREFILTER_DOCS.load(Ordering::Relaxed);
            let pf_killed = PREFILTER_KILLED.load(Ordering::Relaxed);
            let pos_sum = PREFILTER_ALLOWED_POS_SUM.load(Ordering::Relaxed);
            let pos_count = PREFILTER_ALLOWED_POS_COUNT.load(Ordering::Relaxed);
            let pf_skip_collapsed = PREFILTER_SKIPPED_ALL_COLLAPSED.load(Ordering::Relaxed);
            let token_skip = TOKEN_EXTRACTION_SKIPPED.load(Ordering::Relaxed);
            
            log::info!(
                "Stats: calls={} deser={} skipped={} prefilter_killed={}/{} ({:.1}%) avg_positions={:.1} collapsed_skip={} token_skip={}",
                call_num, deser, skipped,
                pf_killed, pf_docs,
                if pf_docs > 0 { pf_killed as f64 / pf_docs as f64 * 100.0 } else { 0.0 },
                if pos_count > 0 { pos_sum as f64 / pos_count as f64 } else { 0.0 },
                pf_skip_collapsed, token_skip
            );
        }

        traversal_result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect_sorted_into_empty() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![1, 2, 3];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_intersect_sorted_into_no_overlap() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_intersect_sorted_into_full_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn test_intersect_sorted_into_partial_overlap() {
        let a = vec![1, 3, 5, 7, 9];
        let b = vec![2, 3, 4, 5, 6];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![3, 5]);
    }

    #[test]
    fn test_intersect_sorted_into_single_element_overlap() {
        let a = vec![5];
        let b = vec![1, 2, 5, 10];
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![5]);
    }

    #[test]
    fn test_collapsed_spec_clone() {
        // Test that CollapsedSpec can be cloned (required for passing through Weight)
        let spec = CollapsedSpec {
            constraint_field: Field::from_field_id(0),
            constraint_matcher: CollapsedMatcher::Exact("test".to_string()),
            edge_field: Field::from_field_id(1),
            edge_matcher: CollapsedMatcher::Exact("nsubj".to_string()),
            constraint_idx: 0,
        };
        let cloned = spec.clone();
        assert!(matches!(cloned.constraint_matcher, CollapsedMatcher::Exact(ref s) if s == "test"));
        assert!(matches!(cloned.edge_matcher, CollapsedMatcher::Exact(ref s) if s == "nsubj"));
        assert_eq!(cloned.constraint_idx, 0);
    }

    #[test]
    fn test_collapsed_spec_regex() {
        // Test CollapsedSpec with regex patterns
        let spec = CollapsedSpec {
            constraint_field: Field::from_field_id(0),
            constraint_matcher: CollapsedMatcher::RegexPattern("protein.*".to_string()),
            edge_field: Field::from_field_id(1),
            edge_matcher: CollapsedMatcher::RegexPattern("nmod_.*".to_string()),
            constraint_idx: 0,
        };
        let cloned = spec.clone();
        assert!(matches!(cloned.constraint_matcher, CollapsedMatcher::RegexPattern(ref s) if s == "protein.*"));
        assert!(matches!(cloned.edge_matcher, CollapsedMatcher::RegexPattern(ref s) if s == "nmod_.*"));
    }

    #[test]
    fn test_collapsed_matcher_display() {
        // Test display formatting for logging
        let exact = CollapsedMatcher::Exact("hello".to_string());
        assert_eq!(exact.display(), "'hello'");
        
        let regex = CollapsedMatcher::RegexPattern("nmod_.*".to_string());
        assert_eq!(regex.display(), "/nmod_.*/");
    }

    #[test]
    fn test_generic_driver_no_positions() {
        // GenericDriver should always return None for matching_positions
        // We can't easily create a Box<dyn Scorer> without an index,
        // so we just verify the type exists and trait is implemented
        // The actual behavior is tested via integration tests
    }

    #[test]
    fn test_intersect_sorted_into_large_lists() {
        // Test with larger lists to verify O(n+m) behavior
        let a: Vec<u32> = (0..1000).step_by(2).collect(); // even numbers
        let b: Vec<u32> = (0..1000).step_by(3).collect(); // multiples of 3
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        // Intersection should be multiples of 6 (LCM of 2 and 3)
        let expected: Vec<u32> = (0..1000).step_by(6).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_intersect_sorted_into_appends_to_buffer() {
        // Verify the function appends to the output buffer (caller must clear if needed)
        // This matches how CombinedPositionDriver uses it: it clears before calling
        let mut out = vec![99, 98, 97]; // Pre-populated
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        intersect_sorted_into(&a, &b, &mut out);
        // Function appends, doesn't clear
        assert_eq!(out, vec![99, 98, 97, 2, 3]);
        
        // Typical usage pattern (as in CombinedPositionDriver):
        out.clear();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![2, 3]);
    }
}
