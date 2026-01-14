use tantivy::{
    query::{Query, BooleanQuery, Occur, TermQuery, RegexQuery},
    schema::{Term, Field, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Traversal, FlatPatternStep};
use anyhow::{Result, anyhow};
use crate::tantivy_integration::graph_traversal::{OptimizedGraphTraversalQuery, flatten_graph_traversal_pattern, CollapsedSpec};
use crate::compiler::compiler::basic_compiler::BasicCompiler;

/// Compiler for graph traversal patterns
pub struct GraphCompiler {
    basic_compiler: BasicCompiler,
    schema: Schema,
}

impl GraphCompiler {
    pub fn new(schema: Schema) -> Self {
        Self {
            basic_compiler: BasicCompiler::new(schema.clone()),
            schema,
        }
    }

    pub fn compile_pattern(&self, pattern: &Pattern) -> Result<Box<dyn Query>> {
        match pattern {
            Pattern::GraphTraversal { src, traversal, dst } => {
                self.compile_graph_traversal(src, traversal, dst)
            }
            _ => {
                // For non-traversal patterns, delegate to basic compiler
                self.basic_compiler.compile_pattern(pattern)
            }
        }
    }

    /// Extract all constraint patterns from a (possibly nested) GraphTraversal pattern
    fn extract_all_constraints(&self, pattern: &Pattern) -> Vec<Pattern> {
        let mut constraints = Vec::new();
        self.collect_constraints(pattern, &mut constraints);
        constraints
    }

    fn collect_constraints(&self, pattern: &Pattern, constraints: &mut Vec<Pattern>) {
        match pattern {
            Pattern::GraphTraversal { src, dst, .. } => {
                // Recursively collect from nested traversals
                self.collect_constraints(src, constraints);
                self.collect_constraints(dst, constraints);
            }
            Pattern::Constraint(_) => {
                constraints.push(pattern.clone());
            }
            _ => {
                // For other patterns (named captures, etc.), add as-is
                constraints.push(pattern.clone());
            }
        }
    }

    /// Try to build a CollapsedSpec for Odinson-style index-level position filtering.
    /// Returns None if the constraint or traversal is not suitable for collapsing.
    /// 
    /// Rules:
    /// - Constraint must be exact string matcher (not regex, not wildcard)
    /// - Traversal must be simple Incoming/Outgoing with exact label (not wildcard, disjunction)
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
        
        // Get the constraint pattern - must be exact string matcher
        let constraint_step = flat_steps.get(constraint_step_idx)?;
        let (constraint_field_name, constraint_term) = match constraint_step {
            FlatPatternStep::Constraint(Pattern::Constraint(
                Constraint::Field { name, matcher: Matcher::String(term) }
            )) => (name.clone(), term.clone()),
            FlatPatternStep::Constraint(Pattern::NamedCapture { pattern, .. }) => {
                // Unwrap named capture to get underlying constraint
                match pattern.as_ref() {
                    Pattern::Constraint(Constraint::Field { name, matcher: Matcher::String(term) }) => {
                        (name.clone(), term.clone())
                    }
                    _ => return None, // Can't collapse
                }
            }
            _ => return None, // Can't collapse regex/wildcard
        };
        
        // Get the traversal - must be simple Incoming/Outgoing with exact label
        let traversal_step = flat_steps.get(traversal_step_idx)?;
        let (edge_field, edge_label) = match traversal_step {
            FlatPatternStep::Traversal(Traversal::Incoming(Matcher::String(label))) => {
                // Incoming edge: constraint node must have incoming edge
                (incoming_edges_field, label.clone())
            }
            FlatPatternStep::Traversal(Traversal::Outgoing(Matcher::String(label))) => {
                if is_first {
                    // First constraint with outgoing: constraint node needs outgoing edge
                    (outgoing_edges_field, label.clone())
                } else {
                    // Last constraint with outgoing (going TO it): constraint node needs incoming edge
                    (incoming_edges_field, label.clone())
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
                    Some(Traversal::Incoming(Matcher::String(label))) => {
                        (incoming_edges_field, label.clone())
                    }
                    Some(Traversal::Outgoing(Matcher::String(label))) => {
                        if is_first {
                            (outgoing_edges_field, label.clone())
                        } else {
                            (incoming_edges_field, label.clone())
                        }
                    }
                    _ => return None, // Can't collapse
                }
            }
            _ => return None, // Can't collapse wildcards/disjunctions/optional
        };
        
        // Get constraint field
        let constraint_field = self.schema.get_field(&constraint_field_name).ok()?;
        
        log::info!(
            "Built CollapsedSpec for {} constraint: field='{}' term='{}' edge_label='{}'",
            if is_first { "first" } else { "last" },
            constraint_field_name,
            constraint_term,
            edge_label
        );
        
        Some(CollapsedSpec {
            constraint_field,
            constraint_term,
            edge_field,
            edge_label,
            constraint_idx,
        })
    }

    fn compile_graph_traversal(&self, src: &Pattern, traversal: &Traversal, dst: &Pattern) -> Result<Box<dyn Query>> {
        // Build the full pattern to extract all constraints and edge labels
        let full_pattern = Pattern::GraphTraversal {
            src: Box::new(src.clone()),
            traversal: traversal.clone(),
            dst: Box::new(dst.clone()),
        };

        // Extract ALL constraints from the entire pattern (flattened)
        let all_constraints = self.extract_all_constraints(&full_pattern);

        // Compile each constraint to a query and combine with AND (BooleanQuery)
        // This ensures we only check documents that have ALL required tokens
        let mut sub_queries: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for constraint in &all_constraints {
            if let Ok(query) = self.basic_compiler.compile_pattern(constraint) {
                sub_queries.push((Occur::Must, query));
            }
        }

        // Extract edge labels from the full pattern and add them to the AND query
        let mut flat_steps = Vec::new();
        flatten_graph_traversal_pattern(&full_pattern, &mut flat_steps);
        
        let mut required_incoming = std::collections::HashSet::new();
        let mut required_outgoing = std::collections::HashSet::new();
        
        for step in &flat_steps {
            if let FlatPatternStep::Traversal(trav) = step {
                match trav {
                    Traversal::Outgoing(matcher) => {
                        let label_str = matcher.pattern_str().to_string();
                        required_outgoing.insert(label_str);
                    },
                    Traversal::Incoming(matcher) => {
                        let label_str = matcher.pattern_str().to_string();
                        required_incoming.insert(label_str);
                    },
                    Traversal::Optional(inner) => {
                        // Extract from optional traversal
                        match &**inner {
                            Traversal::Outgoing(matcher) => {
                                let label_str = matcher.pattern_str().to_string();
                                required_outgoing.insert(label_str);
                            },
                            Traversal::Incoming(matcher) => {
                                let label_str = matcher.pattern_str().to_string();
                                required_incoming.insert(label_str);
                            },
                            _ => {}
                        }
                    },
                    _ => {} // Wildcards/Disjunctions - skip for filtering
                }
            }
        }

        // Get edge label fields from schema
        let incoming_edges_field = self.schema.get_field("incoming_edges")
            .map_err(|_| anyhow!("Incoming edges field not found in schema"))?;
        let outgoing_edges_field = self.schema.get_field("outgoing_edges")
            .map_err(|_| anyhow!("Outgoing edges field not found in schema"))?;

        // Add edge label filters to the AND query
        for label in &required_outgoing {
            let term = Term::from_field_text(outgoing_edges_field, &label);
            let query = Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::WithFreqsAndPositions));
            sub_queries.push((Occur::Must, query));
            log::info!("Added outgoing edge filter: term='{}' in field outgoing_edges", label);
        }

        for label in &required_incoming {
            let term = Term::from_field_text(incoming_edges_field, &label);
            let query = Box::new(TermQuery::new(term, tantivy::schema::IndexRecordOption::WithFreqsAndPositions));
            sub_queries.push((Occur::Must, query));
            log::info!("Added incoming edge filter: term='{}' in field incoming_edges", label);
        }

        log::info!("Combined query includes {} constraints + {} edge label filters", 
                   all_constraints.len(), required_incoming.len() + required_outgoing.len());

        // Create combined index query - documents must match ALL constraints AND edge labels
        let combined_query: Box<dyn Query> = if sub_queries.len() == 1 {
            sub_queries.pop().unwrap().1
        } else if sub_queries.is_empty() {
            // Fallback: match all documents (shouldn't happen with valid patterns)
            Box::new(tantivy::query::AllQuery)
        } else {
            Box::new(BooleanQuery::new(sub_queries))
        };

        // Get the dependencies fields from schema.
        let dependencies_binary_field = self.schema.get_field("dependencies_binary")
            .map_err(|_| anyhow!("Dependencies binary field not found in schema"))?;
        let incoming_edges_field = self.schema.get_field("incoming_edges")
            .map_err(|_| anyhow!("Incoming edges field not found in schema"))?;
        let outgoing_edges_field = self.schema.get_field("outgoing_edges")
            .map_err(|_| anyhow!("Outgoing edges field not found in schema"))?;
        let default_field = self.schema.get_field("word")
            .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;

        // Try to build collapse specs for Odinson-style index-level position filtering
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
        
        if src_collapse.is_some() || dst_collapse.is_some() {
            log::info!(
                "Odinson-style collapsed query optimization enabled: src={}, dst={}",
                src_collapse.is_some(),
                dst_collapse.is_some()
            );
        }

        // Pass the combined query as BOTH src_query and dst_query
        // The OptimizedGraphTraversalQuery will use combined_query for initial filtering
        // Then the full pattern is used for graph traversal verification
        Ok(Box::new(OptimizedGraphTraversalQuery::with_collapse_specs(
            default_field,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
            combined_query.box_clone(),  // Use combined query for src filtering
            traversal.clone(),
            combined_query,              // Use same combined query for dst filtering
            src.clone(),
            dst.clone(),
            src_collapse,
            dst_collapse,
        )))
    }
} 