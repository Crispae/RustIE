use tantivy::{
    query::{Query, Weight, EnableScoring, Scorer},
    schema::{Field, Value},
    DocId, Score, SegmentReader,
    Result as TantivyResult,
    DocSet,
};
use crate::digraph::graph::DirectedGraph;
use crate::compiler::ast::{Pattern, Constraint, Traversal};
use rand::{distributions::Alphanumeric, Rng};

/// Optimized pattern matching query that provides position-aware matching for concatenated patterns
#[derive(Debug)]
pub struct OptimizedPatternMatchingQuery {
    default_field: Field,
    pattern: crate::compiler::ast::Pattern,
    sub_queries: Vec<Box<dyn Query>>,
}

impl OptimizedPatternMatchingQuery {
    pub fn new(
        default_field: Field,
        pattern: crate::compiler::ast::Pattern,
        sub_queries: Vec<Box<dyn Query>>,
    ) -> Self {
        Self {
            default_field,
            pattern,
            sub_queries,
        }
    }
}

impl Query for OptimizedPatternMatchingQuery {
    
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let sub_weights: Vec<Box<dyn Weight>> = self.sub_queries
            .iter()
            .map(|q| q.weight(scoring.clone()))
            .collect::<TantivyResult<Vec<_>>>()?;

        Ok(Box::new(OptimizedPatternMatchingWeight {
            sub_weights,
            pattern: self.pattern.clone(),
            default_field: self.default_field,
        }))
    }
}

impl tantivy::query::QueryClone for OptimizedPatternMatchingQuery {
    fn box_clone(&self) -> Box<dyn Query> {
        Box::new(OptimizedPatternMatchingQuery {
            default_field: self.default_field,
            pattern: self.pattern.clone(),
            sub_queries: self.sub_queries.iter().map(|q| q.box_clone()).collect(),
        })
    }
}

/// Optimized weight for pattern matching queries
struct OptimizedPatternMatchingWeight {
    sub_weights: Vec<Box<dyn Weight>>,
    pattern: crate::compiler::ast::Pattern,
    default_field: Field,
}

impl Weight for OptimizedPatternMatchingWeight {
    
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        // Create scorers for all sub-queries
        let sub_scorers: Vec<Box<dyn Scorer>> = self.sub_weights
            .iter()
            .map(|w| w.scorer(reader, boost))
            .collect::<TantivyResult<Vec<_>>>()?;
        
        let mut scorer = OptimizedPatternMatchingScorer {
            sub_scorers,
            pattern: self.pattern.clone(),
            default_field: self.default_field,
            reader: reader.clone(),
            current_doc: None,
            current_matches: Vec::new(),
            match_index: 0,
            current_doc_matches: Vec::new(),
        };
        
        // Advance to the first document
        let first_doc = scorer.advance();
        
        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("OptimizedPatternMatchingQuery", Score::default()))
    }
}

/// Optimized scorer for pattern matching queries with position awareness
pub struct OptimizedPatternMatchingScorer {
    sub_scorers: Vec<Box<dyn Scorer>>,
    pattern: crate::compiler::ast::Pattern,
    default_field: Field,
    reader: SegmentReader,
    current_doc: Option<DocId>,
    current_matches: Vec<(DocId, Score)>,
    match_index: usize,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
}

impl Scorer for OptimizedPatternMatchingScorer {
    
    fn score(&mut self) -> Score {
        if let Some((_, score)) = self.current_matches.get(self.match_index) {
            *score
        } else {
            Score::default()
        }
    }
}

impl tantivy::DocSet for OptimizedPatternMatchingScorer {
    fn advance(&mut self) -> DocId {
        // Odinson-style: only advance the first sub-scorer, check all others for the same doc
        if self.sub_scorers.is_empty() {
            self.current_doc = None;
            return tantivy::TERMINATED;
        }
        loop {
            let doc = self.sub_scorers[0].advance();
            if doc == tantivy::TERMINATED {
                self.current_doc = None;
                return tantivy::TERMINATED;
            }
            // Check if all sub-scorers are on the same doc
            let mut all_match = true;
            for scorer in &mut self.sub_scorers[1..] {
                while scorer.doc() < doc {
                    let next = scorer.advance();
                    if next == tantivy::TERMINATED {
                        all_match = false;
                        break;
                    }
                }
                if scorer.doc() != doc {
                    all_match = false;
                    break;
                }
            }
            if all_match {
                // Now check for valid sequence/position match in this doc
                if self.check_pattern_matching(doc) {
                    self.current_doc = Some(doc);
                    return doc;
                }
            }
            // Otherwise, continue advancing the first scorer
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc.unwrap_or(tantivy::TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl OptimizedPatternMatchingScorer {
    /// Check if a document has valid pattern matches with position awareness (Odinson-style)
    fn check_pattern_matching(&mut self, doc_id: DocId) -> bool {
        println!("DEBUG: check_pattern_matching for doc_id = {}", doc_id);
        self.current_doc_matches.clear();
        // Get document
        let store_reader = match self.reader.get_store_reader(1) {
            Ok(reader) => reader,
            Err(_) => return false,
        };
        let doc = match store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => return false,
        };
        // Extract tokens from the default field
        let tokens = self.extract_tokens_from_field(&doc, "word");
        if tokens.is_empty() {
            println!("DEBUG: tokens empty");
            return false;
        }
        println!("DEBUG: Tokens in document: {:?}", tokens);
        // Find all valid spans matching the concatenated pattern
        let all_spans = find_constraint_spans_in_sequence(&self.pattern, &tokens);
        println!("DEBUG: Found {} valid spans for concatenated pattern", all_spans.len());
        for match_seq in all_spans {
            self.current_doc_matches.extend(match_seq);
        }
        println!("DEBUG: Pattern matches created: {:?}", self.current_doc_matches);
        !self.current_doc_matches.is_empty()
    }

    /// Extract tokens from a specific field in the document
    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument, field_name: &str) -> Vec<String> {
        if let Ok(field) = self.reader.schema().get_field(field_name) {
            doc.get_all(field)
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        } else {
            vec![]
        }
    }

    /// Expose matches for the current doc
    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        &self.current_doc_matches
    }
}

/// Find all valid spans in a token sequence for a concatenated pattern (Odinson-style)
pub fn find_constraint_spans_in_sequence(pattern: &crate::compiler::ast::Pattern, tokens: &[String]) -> Vec<Vec<crate::types::SpanWithCaptures>> {
    use crate::compiler::ast::Pattern;
    // Only handle Pattern::Concatenated for now
    if let Pattern::Concatenated(patterns) = pattern {
        let mut results = Vec::new();
        let n = patterns.len();
        let len = tokens.len();
        // For each possible start position, try to match the sequence
        for start in 0..len {
            let mut pos = start;
            let mut match_seq = Vec::new();
            let mut matched = true;
            for pat in patterns {
                match pat {
                    Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, matcher }) => {
                        // Only support "word" field for now
                        if name != "word" { matched = false; break; }
                        match matcher {
                            crate::compiler::ast::Matcher::String(s) => {
                                if pos < len && &tokens[pos] == s {
                                    let span = crate::types::Span { start: pos, end: pos + 1 };
                                    let capture = crate::types::NamedCapture::new(format!("c{}", pos), span.clone());
                                    match_seq.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                                    pos += 1;
                                } else {
                                    matched = false; break;
                                }
                            }
                            crate::compiler::ast::Matcher::Regex { pattern, .. } => {
                                if pos < len && regex::Regex::new(pattern).unwrap().is_match(&tokens[pos]) {
                                    let span = crate::types::Span { start: pos, end: pos + 1 };
                                    let capture = crate::types::NamedCapture::new(format!("c{}", pos), span.clone());
                                    match_seq.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                                    pos += 1;
                                } else {
                                    matched = false; break;
                                }
                            }
                        }
                    }
                    Pattern::Constraint(crate::compiler::ast::Constraint::Wildcard) => {
                        // Wildcard matches any single token
                        if pos < len {
                            let span = crate::types::Span { start: pos, end: pos + 1 };
                            let capture = crate::types::NamedCapture::new(format!("wildcard{}", pos), span.clone());
                            match_seq.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                            pos += 1;
                        } else {
                            matched = false; break;
                        }
                    }
                    _ => { matched = false; break; }
                }
            }
            if matched {
                results.push(match_seq);
            }
        }
        results
    } else {
        Vec::new()
    }
}
