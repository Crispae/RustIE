use tantivy::{
    query::Query,
    schema::{Field, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Traversal, FlatPatternStep};
use anyhow::{Result, anyhow};
use crate::tantivy_integration::graph_traversal::{OptimizedGraphTraversalQuery, flatten_graph_traversal_pattern, CollapsedSpec, CollapsedMatcher};

/// Compiler for graph traversal patterns
///
/// Uses Odinson-style collapsed query optimization exclusively:
/// - No BooleanQuery pre-filtering stage
/// - Relies on CombinedPositionDriver for index-level position intersection
/// - CollapsedSpec for first/last constraints merged with adjacent edges
pub struct GraphCompiler {
    schema: Schema,
}

impl GraphCompiler {
    pub fn new(schema: Schema) -> Self {
        Self { schema }
    }

    pub fn compile_pattern(&self, pattern: &Pattern) -> Result<Box<dyn Query>> {
        match pattern {
            Pattern::GraphTraversal { src, traversal, dst } => {
                self.compile_graph_traversal(src, traversal, dst)
            }
            _ => {
                Err(anyhow!("GraphCompiler only handles GraphTraversal patterns"))
            }
        }
    }

    /// Try to build a CollapsedSpec for Odinson-style index-level position filtering.
    /// Returns None if the constraint or traversal is not suitable for collapsing.
    /// 
    /// Rules:
    /// - Constraint must be exact string or regex matcher (not wildcard)
    /// - Traversal must be simple Incoming/Outgoing with exact or regex label, 
    ///   disjunctive traversal (same direction), or concatenated traversal (not wildcard)
    /// - For first constraint: use the first traversal step
    /// - For last constraint: use the last traversal step
    fn try_build_collapse_spec(
        &self,
        flat_steps: &[FlatPatternStep],
        is_first: bool,
        incoming_edges_field: Field,
        outgoing_edges_field: Field,
    ) -> Option<CollapsedSpec> {
        // Count constraints to identify positions
        let constraint_indices: Vec<usize> = flat_steps.iter().enumerate()
            .filter(|(_, s)| matches!(s, FlatPatternStep::Constraint(_)))
            .map(|(i, _)| i)
            .collect();
        
        if constraint_indices.is_empty() {
            return None;
        }
        
        let total_constraints = constraint_indices.len();
        let (constraint_step_idx, traversal_step_idx, constraint_idx) = if is_first {
            // First constraint and the traversal after it
            let c_idx = constraint_indices[0];
            let t_idx = c_idx + 1; // Traversal is right after first constraint
            (c_idx, t_idx, 0usize)
        } else {
            // Last constraint and the traversal before it
            let c_idx = *constraint_indices.last()?;
            let t_idx = c_idx.checked_sub(1)?; // Traversal is right before last constraint
            (c_idx, t_idx, total_constraints - 1)
        };
        
        // Helper to extract CollapsedMatcher from a Matcher
        fn matcher_to_collapsed(matcher: &Matcher) -> Option<CollapsedMatcher> {
            match matcher {
                Matcher::String(term) => Some(CollapsedMatcher::Exact(term.clone())),
                Matcher::Regex { pattern, .. } => Some(CollapsedMatcher::RegexPattern(pattern.clone())),
            }
        }
        
        // Get the constraint pattern - supports exact string and regex matchers
        let constraint_step = flat_steps.get(constraint_step_idx)?;
        let (constraint_field_name, constraint_matcher) = match constraint_step {
            FlatPatternStep::Constraint(Pattern::Constraint(
                Constraint::Field { name, matcher }
            )) => (name.clone(), matcher_to_collapsed(matcher)?),
            FlatPatternStep::Constraint(Pattern::NamedCapture { pattern, .. }) => {
                // Unwrap named capture to get underlying constraint
                match pattern.as_ref() {
                    Pattern::Constraint(Constraint::Field { name, matcher }) => {
                        (name.clone(), matcher_to_collapsed(matcher)?)
                    }
                    _ => return None, // Can't collapse wildcard
                }
            }
            _ => return None, // Can't collapse wildcard
        };
        
        // Get the traversal - supports exact and regex edge labels
        // 
        // Edge field mapping correctness (critical for avoiding false negatives):
        // - For first constraint:
        //   * `<nsubj` (Incoming) → incoming_edges_field (first node receives incoming edge)
        //   * `>nsubj` (Outgoing) → outgoing_edges_field (first node has outgoing edge)
        // - For last constraint:
        //   * `<nsubj` (Incoming) → incoming_edges_field (last node receives incoming edge)
        //   * `>dobj` (Outgoing) → incoming_edges_field (last node is TARGET of outgoing edge, so it has incoming edge)
        //
        // This mapping ensures we check the correct edge field at the constraint position.
        let traversal_step = flat_steps.get(traversal_step_idx)?;
        let (edge_field, edge_matcher) = match traversal_step {
            FlatPatternStep::Traversal(Traversal::Incoming(matcher)) => {
                // Incoming edge: constraint node must have incoming edge (works for both first and last)
                (incoming_edges_field, matcher_to_collapsed(matcher)?)
            }
            FlatPatternStep::Traversal(Traversal::Outgoing(matcher)) => {
                if is_first {
                    // First constraint with outgoing: constraint node needs outgoing edge
                    (outgoing_edges_field, matcher_to_collapsed(matcher)?)
                } else {
                    // Last constraint with outgoing (going TO it): constraint node needs incoming edge
                    // Example: [word=thirsty] >xcomp [word=pretzels]
                    //          The "pretzels" node is the TARGET of >xcomp, so it has an incoming edge
                    (incoming_edges_field, matcher_to_collapsed(matcher)?)
                }
            }
            FlatPatternStep::Traversal(Traversal::Concatenated(traversals)) => {
                // For concatenated traversals, extract the outermost edge
                let relevant_trav = if is_first {
                    traversals.first()
                } else {
                    traversals.last()
                };
                match relevant_trav {
                    Some(Traversal::Incoming(matcher)) => {
                        (incoming_edges_field, matcher_to_collapsed(matcher)?)
                    }
                    Some(Traversal::Outgoing(matcher)) => {
                        if is_first {
                            (outgoing_edges_field, matcher_to_collapsed(matcher)?)
                        } else {
                            (incoming_edges_field, matcher_to_collapsed(matcher)?)
                        }
                    }
                    _ => return None, // Can't collapse
                }
            }
            FlatPatternStep::Traversal(Traversal::Optional(_)) => {
                // Optional traversal: can't collapse (would drop "no-edge" matches)
                return None;
            }
            FlatPatternStep::Traversal(Traversal::Disjunctive(traversals)) => {
                // Disjunctive traversal: convert labels to regex pattern
                log::info!(
                    "Processing DISJUNCTIVE traversal with {} alternatives for {} constraint",
                    traversals.len(),
                    if is_first { "first" } else { "last" }
                );
                
                // Extract labels and verify all are same direction
                let mut labels = Vec::new();
                let mut all_outgoing = true;
                let mut all_incoming = true;
                
                for trav in traversals {
                    match trav {
                        Traversal::Outgoing(matcher) => {
                            all_incoming = false;
                            match matcher {
                                Matcher::String(s) => {
                                    log::info!("  - Outgoing edge label (exact): '{}'", s);
                                    labels.push(regex::escape(s));
                                }
                                Matcher::Regex { pattern, .. } => {
                                    log::info!("  - Outgoing edge label (regex): '{}'", pattern);
                                    labels.push(pattern.clone());
                                }
                            }
                        }
                        Traversal::Incoming(matcher) => {
                            all_outgoing = false;
                            match matcher {
                                Matcher::String(s) => {
                                    log::info!("  - Incoming edge label (exact): '{}'", s);
                                    labels.push(regex::escape(s));
                                }
                                Matcher::Regex { pattern, .. } => {
                                    log::info!("  - Incoming edge label (regex): '{}'", pattern);
                                    labels.push(pattern.clone());
                                }
                            }
                        }
                        _ => {
                            log::warn!("  - Nested complex traversal not supported, falling back");
                            return None;
                        }
                    }
                }
                
                if labels.is_empty() || (!all_outgoing && !all_incoming) {
                    log::warn!(
                        "Disjunctive traversal cannot be collapsed: empty={} mixed_directions={}",
                        labels.is_empty(),
                        !all_outgoing && !all_incoming
                    );
                    return None; // Mixed directions or empty
                }
                
                // Build regex pattern: "(label1|label2|label3)"
                // Note: FST regex doesn't support anchors (^ and $), but since we match
                // against complete terms in the dictionary, anchors aren't needed
                let regex_pattern = format!("({})", labels.join("|"));
                log::info!(
                    "Disjunctive traversal collapsed to regex: '{}' (direction: {})",
                    regex_pattern,
                    if all_outgoing { "outgoing" } else { "incoming" }
                );
                let edge_matcher = CollapsedMatcher::RegexPattern(regex_pattern);
                
                // Determine edge field based on direction and position
                let edge_field = if all_outgoing {
                    if is_first { outgoing_edges_field } else { incoming_edges_field }
                } else {
                    incoming_edges_field
                };
                
                (edge_field, edge_matcher)
            }
            FlatPatternStep::Traversal(Traversal::KleeneStar(_)) => {
                // Kleene star: can't collapse (variable length)
                return None;
            }
            _ => return None, // Can't collapse wildcards/other complex traversals
        };
        
        // Get constraint field
        let constraint_field = self.schema.get_field(&constraint_field_name).ok()?;
        
        log::info!(
            "Built CollapsedSpec for {} constraint: field='{}' constraint={} edge={}",
            if is_first { "first" } else { "last" },
            constraint_field_name,
            constraint_matcher.display(),
            edge_matcher.display()
        );
        
        Some(CollapsedSpec {
            constraint_field,
            constraint_matcher,
            edge_field,
            edge_matcher,
            constraint_idx,
        })
    }

    /// Compile a graph traversal pattern using Odinson-style collapsed query optimization.
    ///
    /// This method does NOT use BooleanQuery pre-filtering. Instead, it relies exclusively
    /// on CombinedPositionDriver for index-level position intersection. Candidate generation
    /// is driven by collapse specs that intersect constraint terms with edge labels.
    ///
    /// If collapse specs cannot be built (e.g., regex constraints, wildcard edges),
    /// the query uses EmptyDriver and returns no results for that segment.
    fn compile_graph_traversal(&self, src: &Pattern, traversal: &Traversal, dst: &Pattern) -> Result<Box<dyn Query>> {
        // Build the full pattern
        let full_pattern = Pattern::GraphTraversal {
            src: Box::new(src.clone()),
            traversal: traversal.clone(),
            dst: Box::new(dst.clone()),
        };

        // Flatten pattern steps - needed for collapse spec building
        let mut flat_steps = Vec::new();
        flatten_graph_traversal_pattern(&full_pattern, &mut flat_steps);

        // Get edge label fields from schema (needed for collapse specs)
        let incoming_edges_field = self.schema.get_field("incoming_edges")
            .map_err(|_| anyhow!("Incoming edges field not found in schema"))?;
        let outgoing_edges_field = self.schema.get_field("outgoing_edges")
            .map_err(|_| anyhow!("Outgoing edges field not found in schema"))?;

        // Build collapse specs for first and last constraints
        // These enable CombinedPositionDriver for index-level position intersection
        let src_collapse = self.try_build_collapse_spec(
            &flat_steps,
            true,  // first constraint
            incoming_edges_field,
            outgoing_edges_field,
        );
        let dst_collapse = self.try_build_collapse_spec(
            &flat_steps,
            false, // last constraint
            incoming_edges_field,
            outgoing_edges_field,
        );

        // Log collapse spec status
        log::info!(
            "Odinson-style collapsed query (no BooleanQuery): src_collapse={}, dst_collapse={}",
            src_collapse.is_some(),
            dst_collapse.is_some()
        );

        if src_collapse.is_none() && dst_collapse.is_none() {
            log::warn!(
                "Neither src nor dst could be collapsed. Query will use EmptyDriver \
                and may return no results. Consider using exact string constraints \
                with simple edge traversals for optimal performance."
            );
        }

        // Get remaining schema fields
        let dependencies_binary_field = self.schema.get_field("dependencies_binary")
            .map_err(|_| anyhow!("Dependencies binary field not found in schema"))?;
        let default_field = self.schema.get_field("word")
            .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;

        // Create the query with collapse specs only - no BooleanQuery
        Ok(Box::new(OptimizedGraphTraversalQuery::collapsed_only(
            default_field,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
            traversal.clone(),
            src.clone(),
            dst.clone(),
            src_collapse,
            dst_collapse,
        )))
    }
} 