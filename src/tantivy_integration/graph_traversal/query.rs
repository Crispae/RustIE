//! OptimizedGraphTraversalQuery implementation.
//!
//! This module provides the query type for graph traversal using

use tantivy::{
    query::{Query, Weight, EnableScoring, Bm25Weight},
    schema::{Field, Schema},
    Result as TantivyResult,
    Term,
};

use std::sync::{Arc, RwLock};

use crate::query::ast::{Pattern, Constraint, Matcher};
use super::types::CollapsedSpec;
use super::weight::{OptimizedGraphTraversalWeight, RegexCache, RegexCacheInner};
use super::pattern_utils::{flatten_graph_traversal_pattern, build_position_prefilter_plan};

/// Collect Tantivy terms from a pattern for BM25 IDF. Only exact string constraints
/// contribute a term; regex/wildcard are skipped.
fn extract_terms_from_pattern(pattern: &Pattern, schema: &Schema, terms: &mut Vec<Term>) {
    match pattern {
        Pattern::Constraint(c) => extract_terms_from_constraint(c, schema, terms),
        Pattern::NamedCapture { pattern: inner, .. } => extract_terms_from_pattern(inner, schema, terms),
        Pattern::Repetition { pattern: inner, .. } => extract_terms_from_pattern(inner, schema, terms),
        Pattern::GraphTraversal { src, dst, .. } => {
            extract_terms_from_pattern(src, schema, terms);
            extract_terms_from_pattern(dst, schema, terms);
        }
        Pattern::Disjunctive(patterns) => {
            for p in patterns {
                extract_terms_from_pattern(p, schema, terms);
            }
        }
        Pattern::Concatenated(patterns) => {
            for p in patterns {
                extract_terms_from_pattern(p, schema, terms);
            }
        }
        Pattern::Assertion(_) | Pattern::Mention { .. } => {}
    }
}

fn extract_terms_from_constraint(c: &Constraint, schema: &Schema, terms: &mut Vec<Term>) {
    match c {
        Constraint::Field { name, matcher } => {
            if let Matcher::String(s) = matcher {
                if let Ok(field) = schema.get_field(name) {
                    terms.push(Term::from_field_text(field, s));
                }
            }
        }
        Constraint::Conjunctive(cs) | Constraint::Disjunctive(cs) => {
            for c in cs {
                extract_terms_from_constraint(c, schema, terms);
            }
        }
        Constraint::Negated(inner) => extract_terms_from_constraint(inner, schema, terms),
        Constraint::Wildcard | Constraint::Fuzzy { .. } => {}
    }
}

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

    /// Build weight with a provided (shared) regex cache. Single source of truth for weight construction.
    pub(crate) fn concrete_weight_with_cache(
        &self,
        regex_cache: RegexCache,
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
            regex_cache,
            None,
            self.default_field,
        ))
    }

    /// Build weight with BM25 scoring when searcher and extractable terms are available.
    /// Uses IDF and field norms from Tantivy; falls back to Option B when terms are empty or on error.
    pub(crate) fn concrete_weight_with_bm25(
        &self,
        searcher: &tantivy::Searcher,
        regex_cache: RegexCache,
        score_field: Field,
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

        let schema = searcher.schema();
        let mut terms = Vec::new();
        extract_terms_from_pattern(&self.src_pattern, schema, &mut terms);
        extract_terms_from_pattern(&self.dst_pattern, schema, &mut terms);

        let bm25_weight = if terms.is_empty() {
            None
        } else {
            Some(Bm25Weight::for_terms(searcher, &terms)?)
        };

        Ok(OptimizedGraphTraversalWeight::new(
            self.dependencies_binary_field,
            flat_steps,
            prefilter_plan,
            self.src_collapse.clone(),
            self.dst_collapse.clone(),
            regex_cache,
            bm25_weight,
            score_field,
        ))
    }

    /// Build weight with a fresh (empty) regex cache. Used when the query runs outside the engine.
    pub(crate) fn concrete_weight(
        &self,
        _searcher: &tantivy::Searcher,
    ) -> TantivyResult<OptimizedGraphTraversalWeight> {
        let fresh_cache = Arc::new(RwLock::new(RegexCacheInner::new(256)));
        self.concrete_weight_with_cache(fresh_cache)
    }
}

impl Query for OptimizedGraphTraversalQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        if let EnableScoring::Enabled { searcher, .. } = scoring {
            return Ok(Box::new(self.concrete_weight(searcher)?));
        }
        let fresh_cache = Arc::new(RwLock::new(RegexCacheInner::new(256)));
        Ok(Box::new(self.concrete_weight_with_cache(fresh_cache)?))
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
