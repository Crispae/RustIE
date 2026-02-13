//! Label matcher optimization for efficient graph traversal.
//!
//! This module provides pre-computed label matchers that avoid repeated
//! vocabulary lookups during traversal, improving performance significantly.

use std::sync::Arc;
use regex::Regex;

use crate::query::ast::{Traversal, Matcher, FlatPatternStep};
use crate::digraph::graph_trait::GraphAccess;
use super::GraphTraversal;

/// Pre-computed label matcher for a traversal step.
///
/// OPTIMIZATION: Avoids repeated vocabulary lookups during traversal.
#[derive(Clone)]
pub(crate) enum ResolvedTraversalMatcher {
    /// Matches any label
    Wildcard,
    /// Pre-resolved label ID for O(1) comparison
    ExactLabel(usize),
    /// Pre-compiled regex pattern
    RegexLabel(Arc<Regex>),
    /// Label not in vocabulary - will never match
    NotFound,
}

impl<G: GraphAccess> GraphTraversal<G> {
    /// Pre-compute all label matchers for the pattern ONCE before traversal.
    /// Returns a Vec indexed by step_idx containing the resolved matcher (if any).
    pub(crate) fn precompute_matchers(&self, pattern: &[FlatPatternStep]) -> Vec<Option<ResolvedTraversalMatcher>> {
        pattern.iter().map(|step| {
            match step {
                FlatPatternStep::Traversal(traversal) => {
                    Some(self.resolve_traversal_matcher(traversal))
                }
                FlatPatternStep::Constraint(_) => None,
            }
        }).collect()
    }

    /// Resolve a traversal to its matcher.
    pub(crate) fn resolve_traversal_matcher(&self, traversal: &Traversal) -> ResolvedTraversalMatcher {
        match traversal {
            Traversal::OutgoingWildcard | Traversal::IncomingWildcard => {
                ResolvedTraversalMatcher::Wildcard
            }
            Traversal::Outgoing(matcher) | Traversal::Incoming(matcher) => {
                self.resolve_matcher(matcher)
            }
            Traversal::Optional(inner) => {
                self.resolve_traversal_matcher(inner)
            }
            _ => ResolvedTraversalMatcher::Wildcard,
        }
    }

    /// Resolve an AST Matcher to a ResolvedTraversalMatcher.
    pub(crate) fn resolve_matcher(&self, matcher: &Matcher) -> ResolvedTraversalMatcher {
        match matcher {
            Matcher::String(s) => {
                if let Some(id) = self.graph.get_label_id(s) {
                    ResolvedTraversalMatcher::ExactLabel(id)
                } else {
                    ResolvedTraversalMatcher::NotFound
                }
            }
            Matcher::Regex { regex, .. } => {
                ResolvedTraversalMatcher::RegexLabel(regex.clone())
            }
        }
    }

    /// Fast label matching using pre-resolved matcher.
    #[inline(always)]
    pub(crate) fn matches_label(&self, resolved: &ResolvedTraversalMatcher, label_id: usize) -> bool {
        match resolved {
            ResolvedTraversalMatcher::Wildcard => true,
            ResolvedTraversalMatcher::ExactLabel(expected_id) => label_id == *expected_id,
            ResolvedTraversalMatcher::RegexLabel(regex) => {
                if let Some(term) = self.graph.get_label(label_id) {
                    regex.is_match(term)
                } else {
                    false
                }
            }
            ResolvedTraversalMatcher::NotFound => false,
        }
    }
}
