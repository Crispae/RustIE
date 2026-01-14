use tantivy::{
    query::{Query, BooleanQuery, Occur, TermQuery, RegexQuery},
    schema::{Term, Field, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Traversal, FlatPatternStep};
use anyhow::{Result, anyhow};
use crate::tantivy_integration::graph_traversal::{OptimizedGraphTraversalQuery, flatten_graph_traversal_pattern};
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

        // Pass the combined query as BOTH src_query and dst_query
        // The OptimizedGraphTraversalQuery will use combined_query for initial filtering
        // Then the full pattern is used for graph traversal verification
        Ok(Box::new(OptimizedGraphTraversalQuery::new(
            default_field,
            dependencies_binary_field,
            incoming_edges_field,
            outgoing_edges_field,
            combined_query.box_clone(),  // Use combined query for src filtering
            traversal.clone(),
            combined_query,              // Use same combined query for dst filtering
            src.clone(),
            dst.clone(),
        )))
    }
} 