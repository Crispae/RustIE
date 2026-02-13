//! OptimizedGraphTraversalQuery implementation.
//!
//! This module provides the query type for graph traversal using

use tantivy::{
    query::{Query, Weight, EnableScoring},
    schema::Field,
    Result as TantivyResult,
};

use crate::query::ast::Pattern;
use super::types::CollapsedSpec;
use super::weight::OptimizedGraphTraversalWeight;
use super::pattern_utils::{flatten_graph_traversal_pattern, build_position_prefilter_plan};

/// Optimized graph traversal query using Odinson-style collapsed query optimization.
/// Candidate generation is driven exclusively by CombinedPositionDriver - no BooleanQuery pre-filtering.
#[derive(Debug)]
pub struct OptimizedGraphTraversalQuery {
    #[allow(dead_code)]
    pub(crate) default_field: Field,
    pub(crate) dependencies_binary_field: Field,
    pub(crate) incoming_edges_field: Field,
    pub(crate) outgoing_edges_field: Field,
    pub(crate) traversal: crate::query::ast::Traversal,
    pub(crate) src_pattern: crate::query::ast::Pattern,
    pub(crate) dst_pattern: crate::query::ast::Pattern,
    /// Collapse spec for src constraint (first) - enables CombinedPositionDriver
    pub(crate) src_collapse: Option<CollapsedSpec>,
    /// Collapse spec for dst constraint (last) - enables CombinedPositionDriver
    pub(crate) dst_collapse: Option<CollapsedSpec>,
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
        traversal: crate::query::ast::Traversal,
        src_pattern: crate::query::ast::Pattern,
        dst_pattern: crate::query::ast::Pattern,
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

    /// Returns the concrete weight without boxing. Used by execution to share one weight across segments.
    /// `_searcher` is accepted for API symmetry with `Query::weight()` but is currently
    /// unused because graph traversal scoring does not depend on corpus statistics.
    pub(crate) fn concrete_weight(
        &self,
        _searcher: &tantivy::Searcher,
    ) -> TantivyResult<OptimizedGraphTraversalWeight> {
        let full_pattern = Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        };
        let mut flat_steps = Vec::new();
        flatten_graph_traversal_pattern(&full_pattern, &mut flat_steps);
        let prefilter_plan = build_position_prefilter_plan(
            &flat_steps,
            self.incoming_edges_field,
            self.outgoing_edges_field,
        );
        Ok(OptimizedGraphTraversalWeight::new(
            self.dependencies_binary_field,
            flat_steps,
            prefilter_plan,
            self.src_collapse.clone(),
            self.dst_collapse.clone(),
        ))
    }
}

impl Query for OptimizedGraphTraversalQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        // Single source of truth: delegate to concrete_weight when searcher is available.
        if let EnableScoring::Enabled { searcher, .. } = scoring {
            return Ok(Box::new(self.concrete_weight(searcher)?));
        }
        // Disabled scoring: no searcher; build weight inline (same logic as concrete_weight).
        let full_pattern = Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        };
        let mut flat_steps = Vec::new();
        flatten_graph_traversal_pattern(&full_pattern, &mut flat_steps);
        let prefilter_plan = build_position_prefilter_plan(
            &flat_steps,
            self.incoming_edges_field,
            self.outgoing_edges_field,
        );
        Ok(Box::new(OptimizedGraphTraversalWeight::new(
            self.dependencies_binary_field,
            flat_steps,
            prefilter_plan,
            self.src_collapse.clone(),
            self.dst_collapse.clone(),
        )))
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
