//! Pattern flattening and prefilter building utilities.
//!
//! This module provides functions for converting nested graph traversal patterns
//! into flat representations and building prefilter plans for position-based filtering.

use tantivy::schema::{Field, Schema};

use crate::query::ast::{Pattern, FlatPatternStep, Traversal, Matcher, Constraint};
use super::types::{EdgeTermReq, ConstraintTermReq, PositionPrefilterPlan};

/// Flatten a nested Pattern::GraphTraversal AST into a flat Vec<FlatPatternStep>
pub fn flatten_graph_traversal_pattern(pattern: &crate::query::ast::Pattern, steps: &mut Vec<FlatPatternStep>) {
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
        Pattern::NamedCapture { pattern: inner, .. } => {
            // Wrap the constraint step with the capture
            steps.push(FlatPatternStep::Constraint(pattern.clone()));
        }
        Pattern::Repetition { pattern: inner, .. } => {
            // For repetitions, add the constraint step
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
pub(crate) fn build_position_prefilter_plan(
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
pub(crate) fn build_constraint_requirements(flat_steps: &[FlatPatternStep], schema: &Schema) -> Vec<ConstraintTermReq> {
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
                        }
                    }
                }
                // For regex: we'd need term enumeration (more complex, handled separately)
            }
            constraint_idx += 1;
        }
    }

    constraint_reqs
}

/// Helper to unwrap NamedCapture/Repetition to get underlying constraint pattern
pub(crate) fn unwrap_constraint_pattern_static(pat: &Pattern) -> &Pattern {
    match pat {
        Pattern::NamedCapture { pattern, .. } => unwrap_constraint_pattern_static(pattern),
        Pattern::Repetition { pattern, .. } => unwrap_constraint_pattern_static(pattern),
        _ => pat,
    }
}
