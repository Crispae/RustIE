//! Type definitions and constants for graph traversal queries.

use std::sync::atomic::AtomicUsize;
use tantivy::schema::Field;

// ---------------------------------------------------------------------------
// Profiling counters: gated behind `profiling` feature to eliminate
// atomic contention (LOCK XADD / cache-line bouncing) in production builds.
// ---------------------------------------------------------------------------

/// Increment a profiling counter. Compiles to a no-op unless `--features profiling`.
#[cfg(feature = "profiling")]
macro_rules! prof_inc {
    ($counter:expr) => {
        $counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    };
    ($counter:expr, $val:expr) => {
        $counter.fetch_add($val, std::sync::atomic::Ordering::Relaxed)
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! prof_inc {
    ($counter:expr) => { 0usize };
    ($counter:expr, $val:expr) => { 0usize };
}

// Re-export macro for use in sibling modules
pub(crate) use prof_inc;

// Counters are always defined (so code referencing them compiles),
// but only mutated when the `profiling` feature is active.
pub(crate) static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_DESER_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_DESER_SKIPPED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_KILLED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_ALLOWED_POS_SUM: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_ALLOWED_POS_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(crate) static SRC_DRIVER_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DST_DRIVER_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DRIVER_ALIGNMENT_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DRIVER_INTERSECTION_SUM: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DRIVER_INTERSECTION_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_SKIPPED_ALL_COLLAPSED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static TOKEN_EXTRACTION_SKIPPED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static REGEX_EXPANSION_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static REGEX_EXPANSION_TERMS: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Capture name pre-allocation: avoids format!("c{}", N) heap allocations
// and the global CAPTURE_COUNTER atomic contention on every match.
// ---------------------------------------------------------------------------

/// Pre-computed capture names for indices 0..99. Avoids heap allocation for the
/// overwhelming majority of captures (most patterns have < 10 constraints).
const PRECOMPUTED_CAPTURE_NAMES: &[&str] = &[
    "c0","c1","c2","c3","c4","c5","c6","c7","c8","c9",
    "c10","c11","c12","c13","c14","c15","c16","c17","c18","c19",
    "c20","c21","c22","c23","c24","c25","c26","c27","c28","c29",
    "c30","c31","c32","c33","c34","c35","c36","c37","c38","c39",
    "c40","c41","c42","c43","c44","c45","c46","c47","c48","c49",
    "c50","c51","c52","c53","c54","c55","c56","c57","c58","c59",
    "c60","c61","c62","c63","c64","c65","c66","c67","c68","c69",
    "c70","c71","c72","c73","c74","c75","c76","c77","c78","c79",
    "c80","c81","c82","c83","c84","c85","c86","c87","c88","c89",
    "c90","c91","c92","c93","c94","c95","c96","c97","c98","c99",
];

/// Get a capture name for the given index. Uses pre-computed strings for 0..99,
/// falls back to format! for larger indices (extremely rare in practice).
#[inline]
pub(crate) fn capture_name(idx: usize) -> String {
    if idx < PRECOMPUTED_CAPTURE_NAMES.len() {
        PRECOMPUTED_CAPTURE_NAMES[idx].to_string()
    } else {
        format!("c{}", idx)
    }
}

/// Default maximum number of terms to expand for regex patterns.
/// Prevents runaway memory/time on broad patterns like `.*`
pub const DEFAULT_MAX_TERM_EXPANSIONS: usize = 50;

/// Edge term requirement for position prefiltering
#[derive(Clone, Debug)]
pub(crate) struct EdgeTermReq {
    pub field: Field,           // incoming_edges_field or outgoing_edges_field
    pub label: String,          // exact term, e.g. "nsubj"
    pub constraint_idx: usize,  // which constraint this restricts
}

/// Constraint term requirement (for exact string matches)
#[derive(Clone, Debug)]
pub(crate) struct ConstraintTermReq {
    pub field: Field,           // constraint field, e.g. "entity"
    pub term: String,           // exact term value, e.g. "B-Gene"
    pub constraint_idx: usize,  // which constraint this restricts
}

/// Position prefilter plan computed from flattened pattern steps
/// NOTE: constraint_reqs are built separately in the scorer (need schema access)
#[derive(Clone, Debug, Default)]
pub(crate) struct PositionPrefilterPlan {
    pub edge_reqs: Vec<EdgeTermReq>,
    pub num_constraints: usize,
}

/// Unified requirement for position prefiltering (edge or constraint)
/// Groups all requirements by constraint_idx for combined filtering (Odinson-style)
#[derive(Clone, Debug)]
pub(crate) enum PositionRequirement {
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

impl CollapsedMatcher {
    /// Format matcher for logging
    pub fn display(&self) -> String {
        match self {
            CollapsedMatcher::Exact(s) => format!("'{}'", s),
            CollapsedMatcher::RegexPattern(p) => format!("/{}/", p),
        }
    }
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
