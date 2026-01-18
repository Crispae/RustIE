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
use rayon::prelude::*;

use crate::compiler::ast::FlatPatternStep;
use crate::digraph::zero_copy::ZeroCopyGraph;
use crate::compiler::ast::{Pattern, Traversal, Matcher, Constraint};
use crate::digraph::traversal::PARALLEL_START_POSITIONS_THRESHOLD;

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
// Regex expansion statistics
static REGEX_EXPANSION_COUNT: AtomicUsize = AtomicUsize::new(0);
static REGEX_EXPANSION_TERMS: AtomicUsize = AtomicUsize::new(0);

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

/// Unified requirement for position prefiltering (edge or constraint)
/// Groups all requirements by constraint_idx for combined filtering (Odinson-style)
#[derive(Clone, Debug)]
enum PositionRequirement {
    Edge {
        req_idx: usize,  // Index into self.edge_postings
        field: Field,
        label: String,
    },
    Constraint {
        req_idx: usize,  // Index into self.constraint_postings
        field: Field,
        term: String,
    },
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
/// Perform galloping search (exponential search) for target in arr starting at start_idx.
/// Returns the index where target is found, or where it would be inserted.
/// Guaranteed to return a value >= start_idx.
fn galloping_search(arr: &[u32], target: u32, start_idx: usize) -> usize {
    if start_idx >= arr.len() {
        return arr.len();
    }
    
    // Check first element
    if arr[start_idx] >= target {
        return start_idx;
    }

    let mut step = 1;
    let mut current = start_idx;
    
    // Gallop forward: 1, 2, 4, 8...
    while current + step < arr.len() && arr[current + step] < target {
        current += step;
        step *= 2;
    }
    
    // Binary search in the identified range [current + 1, min(current + step + 1, len)]
    // We strictly know arr[current] < target, so we search after it.
    let upper_bound = std::cmp::min(current + step + 1, arr.len());
    let range = &arr[current + 1..upper_bound];
    match range.binary_search(&target) {
        Ok(idx) => current + 1 + idx,
        Err(idx) => current + 1 + idx,
    }
}

/// Helper for linear two-pointer intersection (fast for similar list sizes)
fn intersect_linear(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
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

/// Helper for skewed intersection (one short list, one long list)
fn intersect_skewed(short: &[u32], long: &[u32], out: &mut Vec<u32>) {
    let mut long_idx = 0;
    
    for &target in short {
        // Find position of target in long list (or Next Greater Element)
        long_idx = galloping_search(long, target, long_idx);
        
        if long_idx >= long.len() {
            break;
        }
        
        if long[long_idx] == target {
            out.push(target);
            long_idx += 1; // Advance past the match
        }
    }
}

/// Intersection with hybrid Linear/Galloping strategy
fn intersect_sorted_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    if a.is_empty() || b.is_empty() {
        return;
    }

    // Heuristic: If size ratio > 10, use galloping search.
    // Galloping is O(N log M) which beats O(N+M) when M >> N.
    // Linear is faster for dense/similar lists due to CPU cache locality.
    const SKEW_RATIO: usize = 10;
    
    if a.len() > b.len() * SKEW_RATIO {
        intersect_skewed(b, a, out); // b is short, a is long
    } else if b.len() > a.len() * SKEW_RATIO {
        intersect_skewed(a, b, out); // a is short, b is long
    } else {
        intersect_linear(a, b, out); // Standard two-pointer match
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
        return None;
    }
    
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
        // Logging removed for performance

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
            return Some(Box::new(CombinedPositionDriver::new(
                constraint_postings.into_iter().next().unwrap(),
                edge_postings.into_iter().next().unwrap(),
            )));
        }
        
        // Regex path: use UnionAndIntersectDriver
        // Logging removed for performance
        
        let lhs = UnionPositionsIterator::new(constraint_postings);
        let rhs = UnionPositionsIterator::new(edge_postings);
        
        Some(Box::new(UnionAndIntersectDriver::new(lhs, rhs)))
    }

    /// Expand regex patterns in constraints to their matching terms for prefiltering
    /// Returns additional ConstraintTermReq entries for regex patterns
    fn expand_regex_constraints(
        &self,
        reader: &SegmentReader,
        flat_steps: &[FlatPatternStep],
        schema: &Schema,
    ) -> Vec<ConstraintTermReq> {
        let mut expanded_reqs = Vec::new();
        let mut constraint_idx = 0;

        for step in flat_steps.iter() {
            if let FlatPatternStep::Constraint(pat) = step {
                let inner = unwrap_constraint_pattern_static(pat);
                
                if let Pattern::Constraint(Constraint::Field { name, matcher }) = inner {
                    // Only expand regex patterns (exact strings are already handled by build_constraint_requirements)
                    if let Matcher::Regex { pattern, .. } = matcher {
                        if let Ok(field) = schema.get_field(name) {
                            // Check if field is indexed with positions
                            let field_entry = schema.get_field_entry(field);
                            let has_positions = field_entry.field_type().get_index_record_option()
                                .map(|opt| opt.has_positions())
                                .unwrap_or(false);
                            
                            if has_positions {
                                // Strip /.../ delimiters
                                let clean_pattern = pattern.trim_start_matches('/').trim_end_matches('/');
                                
                                // Expand regex using FST automaton
                                if let Some(automaton) = get_or_compile_regex(&self.regex_cache, clean_pattern) {
                                    if let Ok(inverted_index) = reader.inverted_index(field) {
                                        let term_dict = inverted_index.terms();
                                        
                                        if let Some(mut stream) = term_dict.search(automaton.as_ref()).into_stream().ok() {
                                            let mut count = 0;
                                            let mut sample_terms = Vec::new();
                                            while stream.advance() && count < DEFAULT_MAX_TERM_EXPANSIONS {
                                                let term_bytes = stream.key();
                                                let term_str = String::from_utf8_lossy(term_bytes);
                                                
                                                if count < 5 {
                                                    sample_terms.push(term_str.to_string());
                                                }
                                                
                                                expanded_reqs.push(ConstraintTermReq {
                                                    field,
                                                    term: term_str.to_string(),
                                                    constraint_idx,
                                                });
                                                count += 1;
                                            }
                                            
                                            if count > 0 {
                                                REGEX_EXPANSION_COUNT.fetch_add(1, Ordering::Relaxed);
                                                REGEX_EXPANSION_TERMS.fetch_add(count, Ordering::Relaxed);
                                                // Logging removed for performance
                                            } else {
                                                log::warn!(
                                                    "Regex constraint '{}' (constraint_idx={}, field='{}', clean_pattern='{}') expanded to 0 terms - pattern may not match any terms in segment",
                                                    pattern, constraint_idx, name, clean_pattern
                                                );
                                            }
                                        } else {
                                            log::warn!(
                                                "Failed to create stream for regex pattern '{}' (constraint_idx={}, field='{}', clean_pattern='{}')",
                                                pattern, constraint_idx, name, clean_pattern
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                constraint_idx += 1;
            }
        }

        expanded_reqs
    }
}

impl Weight for OptimizedGraphTraversalWeight {

    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        // Odinson-style: Build drivers exclusively from collapse specs
        // No GenericDriver fallback - use EmptyDriver when postings unavailable

        let src_driver: Box<dyn CandidateDriver> = if let Some(ref spec) = self.src_collapse {
            if let Some(driver) = self.build_combined_driver(reader, spec) {
                driver
            } else {
                Box::new(EmptyDriver)
            }
        } else {
            Box::new(EmptyDriver)
        };

        let dst_driver: Box<dyn CandidateDriver> = if let Some(ref spec) = self.dst_collapse {
            if let Some(driver) = self.build_combined_driver(reader, spec) {
                driver
            } else {
                Box::new(EmptyDriver)
            }
        } else {
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
        let mut constraint_reqs = build_constraint_requirements(&self.flat_steps, schema);
        
        // Expand regex patterns for constraint prefiltering (per-segment expansion)
        let expanded_reqs = self.expand_regex_constraints(reader, &self.flat_steps, schema);
        constraint_reqs.extend(expanded_reqs);
        
        // Log prefilter plan info (once per query)
        // Logging removed for performance

        // Log which constraint fields are being used for prefiltering (only those with positions)
        if !constraint_reqs.is_empty() {
            // Logging removed for performance
        } else {
            log::warn!("⚠️  CONSTRAINT PREFILTERING DISABLED: No constraint fields indexed with positions!");
            log::warn!("   This means regex constraints (like /N.*/ or /V.*/) are NOT being prefiltered.");
            log::warn!("   Only edge requirements (nsubj, dobj) are being used for prefiltering.");
            log::warn!("   This significantly reduces prefilter effectiveness!");
            
            // Diagnostic: Check what constraints we have
            let mut constraint_idx = 0;
            for step in &self.flat_steps {
                if let FlatPatternStep::Constraint(pat) = step {
                    let inner = unwrap_constraint_pattern_static(pat);
                    if let Pattern::Constraint(Constraint::Field { name, matcher }) = inner {
                        match matcher {
                            Matcher::String(_) => {
                                log::warn!("   Constraint {}: field='{}' uses exact match (should be prefiltered)", 
                                    constraint_idx, name);
                            }
                            Matcher::Regex { pattern, .. } => {
                                log::warn!("   Constraint {}: field='{}' uses regex '{}' (NOT prefiltered - this is the problem!)", 
                                    constraint_idx, name, pattern);
                            }
                        }
                    }
                    constraint_idx += 1;
                }
            }
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
                    
                    // Logging removed for performance
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

/// Immutable token accessor for thread-safe parallel processing
/// Provides read-only access to pre-loaded tokens from LazyConstraintTokens
struct ImmutableTokenAccessor<'a> {
    tokens: Vec<Option<&'a [String]>>,  // References to pre-loaded tokens
}

impl<'a> ImmutableTokenAccessor<'a> {
    /// Create a new immutable token accessor from a LazyConstraintTokens instance
    /// All tokens must be pre-loaded before calling this
    fn new(lazy_tokens: &'a LazyConstraintTokens) -> Self {
        let tokens: Vec<Option<&'a [String]>> = lazy_tokens
            .cache
            .iter()
            .map(|opt_tokens| opt_tokens.as_ref().map(|tokens| tokens.as_slice()))
            .collect();
        Self { tokens }
    }
    
    /// Get token at a specific position for a constraint (returns owned String)
    /// Returns None if constraint not loaded or position out of bounds
    fn get(&self, constraint_idx: usize, position: usize) -> Option<String> {
        self.tokens
            .get(constraint_idx)?
            .and_then(|tokens| tokens.get(position))
            .cloned()
    }
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

/// Process a single start position and return all matching paths as spans
/// This is used for parallel processing where each thread processes one start position
/// graph_bytes: The raw bytes of the graph (zero-copy, can be shared across threads)
fn process_single_start_position(
    graph_bytes: &[u8],
    flat_steps: &[FlatPatternStep],
    start_pos: usize,
    constraint_field_names: &[String],
    token_accessor: &ImmutableTokenAccessor,
    allowed_positions: &[Option<std::collections::HashSet<u32>>],
    constraint_exact_flags: &[bool],
) -> Vec<crate::types::SpanWithCaptures> {
    // Recreate ZeroCopyGraph from bytes for this thread (zero-copy, no allocation)
    let graph = match ZeroCopyGraph::from_bytes(graph_bytes) {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    
    use crate::digraph::graph_trait::GraphAccess;
    let traversal_engine = crate::digraph::traversal::GraphTraversal::new(graph);
    
    // Create mutable closure for token access (wraps immutable accessor)
    let mut get_token = |constraint_idx: usize, position: usize| -> Option<String> {
        token_accessor.get(constraint_idx, position)
    };
    
    let all_paths = traversal_engine.automaton_query_paths(
        flat_steps,
        &[start_pos],
        constraint_field_names,
        &mut get_token,
        allowed_positions,
        constraint_exact_flags,
    );
    
    let mut matches = Vec::new();
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
            matches.push(
                crate::types::SpanWithCaptures::with_captures(
                    crate::types::Span { start: min_pos, end: max_pos + 1 },
                    captures
                )
            );
        }
    }
    matches
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

    /// Intersect two sorted slices into a new vector (avoids cloning)
    /// O(n+m) time using two-pointer merge
    fn intersect_sorted_slices(a: &[u32], b: &[u32]) -> Vec<u32> {
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
        out
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
            }
        }
        if let Some(dst_positions) = dst_driver_positions {
            if dst_is_last {
                if let Some(idx) = dst_collapsed_idx {
                    allowed[idx] = Some(dst_positions.to_vec());
                }
            }
        }

        // Odinson-style combined filtering: Group ALL requirements (edges + constraints) by constraint_idx
        use std::collections::{HashMap, HashSet};
        let mut requirements_by_constraint: HashMap<usize, Vec<PositionRequirement>> = HashMap::new();
        
        // Add edge requirements (skip collapsed constraints)
        for (req_idx, req) in self.prefilter_plan.edge_reqs.iter().enumerate() {
            // Skip if this edge requirement is for a collapsed constraint
            if Some(req.constraint_idx) == src_collapsed_idx || Some(req.constraint_idx) == dst_collapsed_idx {
                continue;
            }
            
            requirements_by_constraint
                .entry(req.constraint_idx)
                .or_insert_with(Vec::new)
                .push(PositionRequirement::Edge {
                    req_idx,
                    field: req.field,
                    label: req.label.clone(),
                });
        }
        
        // Add constraint requirements (skip collapsed constraints)
        // Track which constraint indices have requirements for validation
        let mut constraint_indices_with_reqs: HashSet<usize> = HashSet::new();
        
        for (req_idx, req) in self.constraint_reqs.iter().enumerate() {
            // Skip if this constraint is collapsed (already has driver positions)
            if Some(req.constraint_idx) == src_collapsed_idx || Some(req.constraint_idx) == dst_collapsed_idx {
                continue;
            }
            
            constraint_indices_with_reqs.insert(req.constraint_idx);
            
            requirements_by_constraint
                .entry(req.constraint_idx)
                .or_insert_with(Vec::new)
                .push(PositionRequirement::Constraint {
                    req_idx,
                    field: req.field,
                    term: req.term.clone(),
                });
        }

        let mut buf: Vec<u32> = Vec::with_capacity(32);

        // Process each constraint_idx with ALL its requirements together (Odinson-style)
        for (constraint_idx, mut requirements) in requirements_by_constraint {
            // OPTIMIZATION: Process edges FIRST, then constraints
            // This ensures edge_intersection is computed before processing constraints,
            // allowing immediate intersection and keeping intermediate sets small
            requirements.sort_by_key(|req| match req {
                PositionRequirement::Edge { .. } => 0,  // Process edges first
                PositionRequirement::Constraint { .. } => 1,  // Process constraints second
            });
            
            // For each constraint_idx, we need to:
            // 1. Intersect all edge positions together (processed first)
            // 2. Union constraint term positions, intersecting immediately with edge_intersection when available
            // 3. Final intersection of constraint union with edge intersection (if not already done)
            
            let mut edge_intersection: Option<Vec<u32>> = None;
            // OPTIMIZATION: Pre-allocate to avoid multiple reallocations
            let mut constraint_union: Vec<u32> = Vec::with_capacity(64);
            let mut has_constraint_reqs = false;
            let mut edge_count = 0;
            let mut constraint_count = 0;
            // Track if we've been intersecting immediately with edge_intersection
            let mut intersected_immediately = false;

            for req in requirements {
                buf.clear();
                
                match req {
                    PositionRequirement::Edge { req_idx, .. } => {
                        edge_count += 1;
                        let postings_opt = self.edge_postings.get_mut(req_idx).and_then(|p| p.as_mut());

                        // If postings is None, the term doesn't exist in this segment at all => cannot match
                        let postings = match postings_opt {
                            Some(p) => p,
                            None => {
                                log::warn!("EdgeReq[{}] has no postings in segment", req_idx);
                                return None;
                            }
                        };

                        // Only seek if we're not already at or past the target document
                        if postings.doc() < doc_id {
                            postings.seek(doc_id);
                        }

                        // Term not present in this doc => cannot match
                        if postings.doc() != doc_id {
                            return None;
                        }

                        postings.positions(&mut buf);

                        if buf.is_empty() {
                            log::warn!("EdgeReq[{}] doc_id={} has term but ZERO positions!", req_idx, doc_id);
                            return None;
                        }

                        // Intersect with existing edge positions
                        match &mut edge_intersection {
                            None => {
                                edge_intersection = Some(std::mem::take(&mut buf));
                            }
                            Some(existing) => {
                                Self::intersect_sorted_in_place(existing, &buf);
                                // OPTIMIZATION: Early exit when edge_intersection becomes empty
                                if existing.is_empty() {
                                    return None;  // Fail immediately
                                }
                            }
                        }
                    }
                    PositionRequirement::Constraint { req_idx, .. } => {
                        constraint_count += 1;
                        has_constraint_reqs = true;
                        
                        let postings_opt = self.constraint_postings.get_mut(req_idx).and_then(|p| p.as_mut());

                        // If postings is None, skip this term (other terms for same constraint_idx might match)
                        let postings = match postings_opt {
                            Some(p) => p,
                            None => {
                                continue;
                            }
                        };

                        // Only seek if we're not already at or past the target document
                        if postings.doc() < doc_id {
                            postings.seek(doc_id);
                        }

                        // Term not present in this doc => skip this term
                        if postings.doc() != doc_id {
                            continue;
                        }

                        postings.positions(&mut buf);

                        if !buf.is_empty() {
                            // OPTIMIZATION: If edge_intersection exists, intersect immediately to keep sets small
                            // This prevents building large intermediate constraint_union sets
                            if let Some(ref edge_positions) = edge_intersection {
                                // OPTIMIZATION: Use intersect_sorted_slices to avoid expensive clone
                                let filtered = Self::intersect_sorted_slices(&buf, edge_positions);
                                if !filtered.is_empty() {
                                    constraint_union.extend_from_slice(&filtered);
                                    intersected_immediately = true;
                                }
                            } else {
                                // No edge restrictions yet - add all positions to union
                                constraint_union.extend_from_slice(&buf);
                            }
                        }
                    }
                }
            }

            // Check: if a constraint has requirements but no matching terms, document cannot match
            if constraint_indices_with_reqs.contains(&constraint_idx) && constraint_union.is_empty() {
                // This constraint has regex expansion but no terms matched in this document
                return None;
            }

            // Compute final intersection for this constraint_idx
            let final_positions = if has_constraint_reqs {
                // OPTIMIZATION: Sort and dedup the unioned constraint positions
                // Even though filtered results are sorted, the union of multiple sorted slices needs sorting
                if !constraint_union.is_empty() {
                    constraint_union.sort_unstable();
                    constraint_union.dedup();
                }
                
                if constraint_union.is_empty() {
                    return None;
                }

                // Intersect constraint union with edge intersection
                match edge_intersection {
                    None => {
                        // No edge restrictions - use constraint positions
                        constraint_union
                    }
                    Some(mut edge_positions) => {
                        // Intersect: only positions that have BOTH the edge AND at least one constraint term
                        Self::intersect_sorted_in_place(&mut edge_positions, &constraint_union);
                        if edge_positions.is_empty() {
                            return None;  // Fail immediately
                        }
                        edge_positions
                    }
                }
            } else {
                // No constraint requirements - use edge intersection directly
                match edge_intersection {
                    None => {
                        // No requirements at all for this constraint_idx (shouldn't happen, but handle gracefully)
                        continue;
                    }
                    Some(edge_positions) => edge_positions,
                }
            };

            allowed[constraint_idx] = Some(final_positions);
        }

        // Validate: all constraints with requirements must have positions
        for constraint_idx in &constraint_indices_with_reqs {
            if Some(*constraint_idx) != src_collapsed_idx && Some(*constraint_idx) != dst_collapsed_idx {
                if allowed[*constraint_idx].is_none() || allowed[*constraint_idx].as_ref().unwrap().is_empty() {
                    return None;
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
        
        // Driver positions are used for first/last constraints (Odinson position handoff)

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
                        allowed.iter().map(|&p| p as usize).collect()
                    } else {
                        // Fallback: shouldn't happen if driver worked correctly
                        self.find_positions_in_tokens(tokens, constraint_pat)
                    }
                } else if is_last_constraint && self.dst_collapse.is_some() {
                    // Use allowed_positions[last] which was set from dst_driver (already filtered by constraint + edge)
                    // This skips token scanning entirely for last constraint
                    if let Some(ref allowed) = allowed_positions[constraint_count] {
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
                // Check if we should use parallel processing
                if src_positions.len() >= PARALLEL_START_POSITIONS_THRESHOLD {
                    // Pre-load all tokens for thread-safety before parallel processing
                    for constraint_idx in 0..constraint_field_names.len() {
                        lazy_tokens.ensure_loaded(constraint_idx);
                    }
                    
                    // Create immutable token accessor for parallel processing
                    let token_accessor = ImmutableTokenAccessor::new(&lazy_tokens);
                    
                    // Process start positions in parallel
                    // Pass binary_data (bytes) instead of graph, so each thread can create its own ZeroCopyGraph
                    // This is safe because ZeroCopyGraph is zero-copy - it just holds references to the bytes
                    let all_matches: Vec<crate::types::SpanWithCaptures> = src_positions
                        .par_iter()
                        .flat_map(|&src_pos| {
                            process_single_start_position(
                                binary_data,
                                flat_steps,
                                src_pos,
                                constraint_field_names,
                                &token_accessor,
                                &allowed_positions_hashset,
                                &constraint_exact_flags,
                            )
                        })
                        .collect();
                    
                    if !all_matches.is_empty() {
                        self.current_doc_matches.extend(all_matches);
                        return true;
                    }
                } else {
                    // Sequential processing for small number of start positions (preserves lazy loading)
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
            
            // Logging removed for performance
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
                            // Constraint prefilter added
                        } else {
                            // Constraint prefilter skipped (field not indexed with positions)
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

/// Public function to retrieve profiling statistics
/// Returns a struct with all performance metrics
#[derive(Debug, Clone)]
pub struct GraphTraversalStats {
    pub call_count: usize,
    pub graph_deser_count: usize,
    pub graph_deser_skipped: usize,
    pub prefilter_docs: usize,
    pub prefilter_killed: usize,
    pub prefilter_allowed_pos_sum: usize,
    pub prefilter_allowed_pos_count: usize,
    pub src_driver_docs: usize,
    pub dst_driver_docs: usize,
    pub driver_alignment_docs: usize,
    pub driver_intersection_sum: usize,
    pub driver_intersection_count: usize,
    pub prefilter_skipped_all_collapsed: usize,
    pub token_extraction_skipped: usize,
    pub regex_expansion_count: usize,
    pub regex_expansion_terms: usize,
}

impl GraphTraversalStats {
    pub fn get() -> Self {
        Self {
            call_count: CALL_COUNT.load(Ordering::Relaxed),
            graph_deser_count: GRAPH_DESER_COUNT.load(Ordering::Relaxed),
            graph_deser_skipped: GRAPH_DESER_SKIPPED.load(Ordering::Relaxed),
            prefilter_docs: PREFILTER_DOCS.load(Ordering::Relaxed),
            prefilter_killed: PREFILTER_KILLED.load(Ordering::Relaxed),
            prefilter_allowed_pos_sum: PREFILTER_ALLOWED_POS_SUM.load(Ordering::Relaxed),
            prefilter_allowed_pos_count: PREFILTER_ALLOWED_POS_COUNT.load(Ordering::Relaxed),
            src_driver_docs: SRC_DRIVER_DOCS.load(Ordering::Relaxed),
            dst_driver_docs: DST_DRIVER_DOCS.load(Ordering::Relaxed),
            driver_alignment_docs: DRIVER_ALIGNMENT_DOCS.load(Ordering::Relaxed),
            driver_intersection_sum: DRIVER_INTERSECTION_SUM.load(Ordering::Relaxed),
            driver_intersection_count: DRIVER_INTERSECTION_COUNT.load(Ordering::Relaxed),
            prefilter_skipped_all_collapsed: PREFILTER_SKIPPED_ALL_COLLAPSED.load(Ordering::Relaxed),
            token_extraction_skipped: TOKEN_EXTRACTION_SKIPPED.load(Ordering::Relaxed),
            regex_expansion_count: REGEX_EXPANSION_COUNT.load(Ordering::Relaxed),
            regex_expansion_terms: REGEX_EXPANSION_TERMS.load(Ordering::Relaxed),
        }
    }

    pub fn reset() {
        CALL_COUNT.store(0, Ordering::Relaxed);
        GRAPH_DESER_COUNT.store(0, Ordering::Relaxed);
        GRAPH_DESER_SKIPPED.store(0, Ordering::Relaxed);
        PREFILTER_DOCS.store(0, Ordering::Relaxed);
        PREFILTER_KILLED.store(0, Ordering::Relaxed);
        PREFILTER_ALLOWED_POS_SUM.store(0, Ordering::Relaxed);
        PREFILTER_ALLOWED_POS_COUNT.store(0, Ordering::Relaxed);
        SRC_DRIVER_DOCS.store(0, Ordering::Relaxed);
        DST_DRIVER_DOCS.store(0, Ordering::Relaxed);
        DRIVER_ALIGNMENT_DOCS.store(0, Ordering::Relaxed);
        DRIVER_INTERSECTION_SUM.store(0, Ordering::Relaxed);
        DRIVER_INTERSECTION_COUNT.store(0, Ordering::Relaxed);
        PREFILTER_SKIPPED_ALL_COLLAPSED.store(0, Ordering::Relaxed);
        TOKEN_EXTRACTION_SKIPPED.store(0, Ordering::Relaxed);
        REGEX_EXPANSION_COUNT.store(0, Ordering::Relaxed);
        REGEX_EXPANSION_TERMS.store(0, Ordering::Relaxed);
    }

    pub fn print_summary(&self) {
        println!("\n=== Graph Traversal Performance Statistics ===");
        
        let skip_rate = if self.call_count > 0 {
            (self.graph_deser_skipped as f64 / self.call_count as f64) * 100.0
        } else {
            0.0
        };
        
        let prefilter_kill_rate = if self.prefilter_docs > 0 {
            (self.prefilter_killed as f64 / self.prefilter_docs as f64) * 100.0
        } else {
            0.0
        };
        
        let avg_allowed_pos = if self.prefilter_allowed_pos_count > 0 {
            self.prefilter_allowed_pos_sum as f64 / self.prefilter_allowed_pos_count as f64
        } else {
            0.0
        };
        
        let avg_intersection = if self.driver_intersection_count > 0 {
            self.driver_intersection_sum as f64 / self.driver_intersection_count as f64
        } else {
            0.0
        };
        
        println!("Document Processing:");
        println!("  Candidates checked: {}", self.call_count);
        println!("  Graphs deserialized: {} ({} skipped, {:.1}% skip rate)", 
                 self.graph_deser_count, self.graph_deser_skipped, skip_rate);
        
        println!("\nPrefilter Performance:");
        println!("  Documents checked: {}", self.prefilter_docs);
        println!("  Killed by prefilter: {} ({:.1}% kill rate)", 
                 self.prefilter_killed, prefilter_kill_rate);
        println!("  Avg allowed positions per constraint: {:.1}", avg_allowed_pos);
        println!("  Prefilter skipped (all-collapsed): {}", self.prefilter_skipped_all_collapsed);
        
        println!("\nOdinson Driver Performance:");
        println!("  Source driver docs: {}", self.src_driver_docs);
        println!("  Destination driver docs: {}", self.dst_driver_docs);
        println!("  Aligned docs: {}", self.driver_alignment_docs);
        println!("  Avg intersection size: {:.1}", avg_intersection);
        
        println!("\nOptimizations:");
        println!("  Token extractions skipped: {}", self.token_extraction_skipped);
        
        println!("\nRegex Expansion:");
        println!("  Regex patterns expanded: {}", self.regex_expansion_count);
        println!("  Total terms from expansion: {}", self.regex_expansion_terms);
        if self.regex_expansion_count > 0 {
            let avg_terms = self.regex_expansion_terms as f64 / self.regex_expansion_count as f64;
            println!("  Average terms per pattern: {:.1}", avg_terms);
        }
        
        // Calculate efficiency metrics
        if self.call_count > 0 {
            let deser_rate = (self.graph_deser_count as f64 / self.call_count as f64) * 100.0;
            println!("\nEfficiency Metrics:");
            println!("  Graph deserialization rate: {:.1}% (lower is better)", deser_rate);
            println!("  Prefilter effectiveness: {:.1}% documents filtered out", prefilter_kill_rate);
        }
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
    #[test]
    fn test_intersect_sorted_skewed_galloping() {
        // Test highly skewed lists to trigger galloping search
        // a: long list (0..1000)
        let a: Vec<u32> = (0..1000).collect();
        // b: short list (sparse)
        let b: Vec<u32> = vec![5, 50, 500, 999];
        
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert_eq!(out, vec![5, 50, 500, 999], "Should find all matches in skewed intersection");
        
        // Reverse order (short vs long)
        out.clear();
        intersect_sorted_into(&b, &a, &mut out);
        assert_eq!(out, vec![5, 50, 500, 999], "Should handle reversed arguments correctly");
    }

    #[test]
    fn test_intersect_sorted_skewed_no_match() {
        let a: Vec<u32> = (0..1000).collect();
        let b: Vec<u32> = vec![1001, 2000]; // Out of bounds
        
        let mut out = Vec::new();
        intersect_sorted_into(&a, &b, &mut out);
        assert!(out.is_empty(), "Should be empty");
    }
}
