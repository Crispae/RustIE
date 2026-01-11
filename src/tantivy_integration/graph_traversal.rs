use tantivy::{
    query::{Query, Weight, EnableScoring, Scorer},
    schema::{Field, Value},
    DocId, Score, SegmentReader,
    Result as TantivyResult,
    DocSet,
};

use crate::compiler::ast::FlatPatternStep;
use crate::digraph::graph::DirectedGraph;
use crate::compiler::ast::{Pattern, Constraint, Traversal};
use rand::{distributions::Alphanumeric, Rng};



// Optimized graph traversal query that first finds documents containing both source and destination tokens
#[derive(Debug)]
pub struct OptimizedGraphTraversalQuery {
    default_field: Field,
    dependencies_binary_field: Field,
    src_query: Box<dyn Query>,
    traversal: crate::compiler::ast::Traversal,
    dst_query: Box<dyn Query>,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
}


impl OptimizedGraphTraversalQuery {
    pub fn new(
        default_field: Field,
        dependencies_binary_field: Field,
        src_query: Box<dyn Query>,
        traversal: crate::compiler::ast::Traversal,
        dst_query: Box<dyn Query>,
        src_pattern: crate::compiler::ast::Pattern,
        dst_pattern: crate::compiler::ast::Pattern,
    ) -> Self {
        Self {
            default_field,
            dependencies_binary_field,
            src_query,
            traversal,
            dst_query,
            src_pattern,
            dst_pattern,
        }
    }
}

impl Query for OptimizedGraphTraversalQuery {
    
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let src_weight = self.src_query.weight(scoring)?;
        let dst_weight = self.dst_query.weight(scoring)?;
        
        

        Ok(Box::new(OptimizedGraphTraversalWeight {
            src_weight,
            dst_weight,
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
            traversal: self.traversal.clone(),
            dependencies_binary_field: self.dependencies_binary_field,
        }))
    }
}

impl tantivy::query::QueryClone for OptimizedGraphTraversalQuery {
    fn box_clone(&self) -> Box<dyn Query> {
        Box::new(OptimizedGraphTraversalQuery {
            default_field: self.default_field,
            dependencies_binary_field: self.dependencies_binary_field,
            src_query: self.src_query.box_clone(),
            traversal: self.traversal.clone(),
            dst_query: self.dst_query.box_clone(),
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
        })
    }
}

/// Optimized weight for graph traversal queries
struct OptimizedGraphTraversalWeight {
    src_weight: Box<dyn Weight>,
    dst_weight: Box<dyn Weight>,
    traversal: crate::compiler::ast::Traversal,
    dependencies_binary_field: Field,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
}

impl Weight for OptimizedGraphTraversalWeight {
    
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        // Create fresh scorers for finding documents
        let src_scorer = self.src_weight.scorer(reader, boost)?;
        let dst_scorer = self.dst_weight.scorer(reader, boost)?;
        
        let mut scorer = OptimizedGraphTraversalScorer {
            src_scorer,
            dst_scorer,
            traversal: self.traversal.clone(),
            dependencies_binary_field: self.dependencies_binary_field,
            reader: reader.clone(),
            current_doc: None,
            current_matches: Vec::new(),
            match_index: 0,
            src_pattern: self.src_pattern.clone(),
            dst_pattern: self.dst_pattern.clone(),
       
            current_doc_matches: Vec::new(),
        };
        

        
        // Create fresh scorers for traversal (since the previous ones were consumed)
        let src_scorer_for_traversal = self.src_weight.scorer(reader, boost)?;
        let dst_scorer_for_traversal = self.dst_weight.scorer(reader, boost)?;
        
        // Replace the consumed scorers with fresh ones
        scorer.src_scorer = src_scorer_for_traversal;
        scorer.dst_scorer = dst_scorer_for_traversal;
        
        // Advance to the first document
        let first_doc = scorer.advance();
        
        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("OptimizedGraphTraversalQuery", Score::default()))
    }
}

/// Optimized scorer for graph traversal queries
pub struct OptimizedGraphTraversalScorer {
    
    src_scorer: Box<dyn Scorer>,
    dst_scorer: Box<dyn Scorer>,
    traversal: crate::compiler::ast::Traversal,
    dependencies_binary_field: Field,
    reader: SegmentReader,
    current_doc: Option<DocId>,
    current_matches: Vec<(DocId, Score)>,
    match_index: usize,
    src_pattern: crate::compiler::ast::Pattern,
    dst_pattern: crate::compiler::ast::Pattern,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
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

            let src_doc = self.src_scorer.doc();
            let dst_doc = self.dst_scorer.doc();

            // If either scorer is exhausted, we're done
            if src_doc == tantivy::TERMINATED || dst_doc == tantivy::TERMINATED {
                self.current_doc = None;
                println!("DEBUG: advance() terminated: src_doc = {}, dst_doc = {}", src_doc, dst_doc);
                return tantivy::TERMINATED;
            }
            println!("DEBUG: advance() considering src_doc = {}, dst_doc = {}", src_doc, dst_doc);
            if src_doc < dst_doc {
                self.src_scorer.advance();
            } else if dst_doc < src_doc {
                self.dst_scorer.advance();
            } else {
                // src_doc == dst_doc: both have matches in this doc
                let doc_id = src_doc;
                println!("DEBUG: advance() found candidate doc_id = {}", doc_id);
                if self.check_graph_traversal(doc_id) {
                    println!("DEBUG: advance() doc_id {} MATCHED graph traversal", doc_id);
                    self.current_doc = Some(doc_id);
                    // Advance both scorers for next call
                    self.src_scorer.advance();
                    self.dst_scorer.advance();
                    return doc_id;
                } else {
                    println!("DEBUG: advance() doc_id {} did NOT match graph traversal", doc_id);
                    // No match, advance both scorers
                    self.src_scorer.advance();
                    self.dst_scorer.advance();
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

impl OptimizedGraphTraversalScorer {
    

    /// Check if a document has valid graph traversal from source to destination
    fn check_graph_traversal(&mut self, doc_id: DocId) -> bool {        
        // Clear matches from previous document
        self.current_doc_matches.clear();
        
        // Phase 1: Get document
        let store_reader = match self.reader.get_store_reader(1) {
            Ok(reader) => reader,
            Err(_) => {
                return false;
            }
        };
        let doc = match store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => {
                return false;
            }
        };

        // Phase 2: Extract tokens for each constraint step
        let mut flat_steps = Vec::new();
        flatten_graph_traversal_pattern(&self.traversal_to_pattern(), &mut flat_steps);

        // For each constraint step, extract the field name and tokens
        let mut constraint_fields_and_tokens = Vec::new();
        for step in &flat_steps {
            if let FlatPatternStep::Constraint(pat) = step {
                let field_name = match pat {
                    crate::compiler::ast::Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, .. }) => name.as_str(),
                    _ => "word",
                };
                let tokens = self.extract_tokens_from_field(&doc, field_name);
                constraint_fields_and_tokens.push((field_name.to_string(), tokens));
            }
        }

        // Find the first constraint in the flat_steps
        let first_constraint = flat_steps.iter().find_map(|step| {
            if let FlatPatternStep::Constraint(pat) = step {
                Some(pat)
            } else {
                None
            }
        });

        // Use the tokens for the first constraint to find source positions
        let src_positions = match first_constraint {
            Some(pat) => {
                if let Some((_, tokens)) = constraint_fields_and_tokens.get(0) {
                    self.find_positions_in_tokens(tokens, pat)
                } else {
                    vec![]
                }
            },
            None => vec![],
        };

        if src_positions.is_empty() {
            println!("DEBUG: src_positions empty");
            return false;
        }
        
        // Phase 4: Get binary dependency graph
        let binary_data = match doc.get_first(self.dependencies_binary_field).and_then(|v| v.as_bytes()) {
            Some(data) => data,
            None => {
                println!("DEBUG: No binary dependency graph");
                return false;
            }
        };
        
        // Phase 5: Deserialize the graph
        let graph = match DirectedGraph::from_bytes(binary_data) {
            Ok(graph) => graph,
            Err(_) => {
                println!("DEBUG: Failed to deserialize graph");
                return false;
            }
        };
        
        // Phase 7: Run automaton traversal for each src_pos
        let traversal_engine = crate::digraph::traversal::GraphTraversal::new(graph);
        let mut found = false;
        
        for &src_pos in &src_positions {
            let all_paths = traversal_engine.automaton_query_paths(&flat_steps, &[src_pos], &constraint_fields_and_tokens);
            for path in &all_paths {
                // Collect captures for each constraint step
                println!("DEBUG: flat_steps = {:?}", flat_steps);
                println!("DEBUG: path = {:?}", path);
                let mut captures = Vec::new();
                let mut constraint_idx = 0;
                for (step_idx, step) in flat_steps.iter().enumerate() {
                    if let FlatPatternStep::Constraint(ref pat) = step {
                        if let Some(&node_idx) = path.get(constraint_idx) {
                            let span = crate::types::Span { start: node_idx, end: node_idx + 1 };
                            let name = match pat {
                                crate::compiler::ast::Pattern::NamedCapture { name, .. } => name.clone(),
                                _ => {
                                    let rand_name: String = rand::thread_rng()
                                        .sample_iter(&rand::distributions::Alphanumeric)
                                        .take(8)
                                        .map(char::from)
                                        .collect();
                                    rand_name
                                }
                            };
                            captures.push(crate::types::NamedCapture::new(name, span));
                        }
                        constraint_idx += 1;
                    }
                }
                if !path.is_empty() {
                    let min_pos = *path.iter().min().unwrap();
                    let max_pos = *path.iter().max().unwrap();
                    let span = crate::types::Span { start: min_pos, end: max_pos + 1 };
                    println!("DEBUG: Adding SpanWithCaptures: span = {:?}, captures = {:?}", span, captures);
                    self.current_doc_matches.push(crate::types::SpanWithCaptures::with_captures(span, captures));
                }
            }
            if !all_paths.is_empty() {
                found = true;
            }
        }
        found
    }

    /// Extract the field name from a pattern
    fn get_field_name_from_pattern<'a>(&self, pattern: &'a crate::compiler::ast::Pattern) -> &'a str {
        match pattern {
            crate::compiler::ast::Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, .. }) => {
                name.as_str()
            }
            _ => "word", // default to word field
        }
    }
    
    /// Extract tokens from a specific field in the document
    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument, field_name: &str) -> Vec<String> {
        if let Ok(field) = self.reader.schema().get_field(field_name) {
            let tokens: Vec<String> = doc.get_all(field)
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            tokens
        } else {
            vec![]
        }
    }
    
    /// Find positions that match a pattern (for backward compatibility)
    fn find_positions_matching_pattern(&self, tokens: &[String], pattern: &crate::compiler::ast::Pattern) -> Vec<usize> {
        self.find_positions_in_tokens(tokens, pattern)
    }

    // NEW: expose matches for the current doc
    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        println!("DEBUG: get_current_doc_matches called, current_doc_matches.len() = {}", self.current_doc_matches.len());
        println!("DEBUG: current_doc_matches = {:?}", self.current_doc_matches);
        &self.current_doc_matches
    }

    /// Helper: Convert traversal AST to Pattern (for now, just wrap in GraphTraversal)
    fn traversal_to_pattern(&self) -> Pattern {
        Pattern::GraphTraversal {
            src: Box::new(self.src_pattern.clone()),
            traversal: self.traversal.clone(),
            dst: Box::new(self.dst_pattern.clone()),
        }
    }

    /// Find positions in tokens that match a given pattern (string, regex, or wildcard for any field)
    fn find_positions_in_tokens(&self, tokens: &[String], pattern: &crate::compiler::ast::Pattern) -> Vec<usize> {
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
                    Matcher::Regex { pattern, .. } => {
                        let re = regex::Regex::new(pattern).unwrap();
                        for (i, token) in tokens.iter().enumerate() {
                            if re.is_match(token) {
                                positions.push(i);
                            }
                        }
                    }
                }
            }
            Pattern::Constraint(Constraint::Wildcard) => {
                for i in 0..tokens.len() {
                    positions.push(i);
                }
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
