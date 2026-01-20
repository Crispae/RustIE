//! Type definitions and constants for graph traversal queries.

use std::sync::atomic::AtomicUsize;
use tantivy::schema::Field;

// Global counter for generating unique capture names (much faster than rand)
pub(crate) static CAPTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Module-level counters for profiling (shared across all instances)
pub(crate) static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_DESER_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static GRAPH_DESER_SKIPPED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_KILLED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_ALLOWED_POS_SUM: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PREFILTER_ALLOWED_POS_COUNT: AtomicUsize = AtomicUsize::new(0);

// Odinson-style collapsed query metrics
pub(crate) static SRC_DRIVER_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DST_DRIVER_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DRIVER_ALIGNMENT_DOCS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DRIVER_INTERSECTION_SUM: AtomicUsize = AtomicUsize::new(0);
pub(crate) static DRIVER_INTERSECTION_COUNT: AtomicUsize = AtomicUsize::new(0);
// Optimization: Skip prefilter when all constraints are collapsed (2-constraint patterns)
pub(crate) static PREFILTER_SKIPPED_ALL_COLLAPSED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static TOKEN_EXTRACTION_SKIPPED: AtomicUsize = AtomicUsize::new(0);
// Regex expansion statistics
pub(crate) static REGEX_EXPANSION_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static REGEX_EXPANSION_TERMS: AtomicUsize = AtomicUsize::new(0);

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
