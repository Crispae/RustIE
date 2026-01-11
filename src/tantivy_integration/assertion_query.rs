use tantivy::query::{Query, Weight, Scorer, EnableScoring};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet};
use tantivy::schema::{Field, Value};
use crate::compiler::ast::Pattern;

/// Lookahead assertion type
#[derive(Debug, Clone)]
pub enum LookaheadType {
    PositiveLookahead,
    NegativeLookahead,
    PositiveLookbehind,
    NegativeLookbehind,
}

/// Query wrapper for lookahead/lookbehind assertions
#[derive(Debug)]
pub struct LookaheadQuery {
    lookahead_type: LookaheadType,
    assertion_pattern: Pattern,
    default_field: Field,
}

impl Clone for LookaheadQuery {
    fn clone(&self) -> Self {
        LookaheadQuery {
            lookahead_type: self.lookahead_type.clone(),
            assertion_pattern: self.assertion_pattern.clone(),
            default_field: self.default_field,
        }
    }
}

impl LookaheadQuery {
    pub fn positive_lookahead(pattern: Pattern, default_field: Field) -> Self {
        Self {
            lookahead_type: LookaheadType::PositiveLookahead,
            assertion_pattern: pattern,
            default_field,
        }
    }
    
    pub fn negative_lookahead(pattern: Pattern, default_field: Field) -> Self {
        Self {
            lookahead_type: LookaheadType::NegativeLookahead,
            assertion_pattern: pattern,
            default_field,
        }
    }
    
    pub fn positive_lookbehind(pattern: Pattern, default_field: Field) -> Self {
        Self {
            lookahead_type: LookaheadType::PositiveLookbehind,
            assertion_pattern: pattern,
            default_field,
        }
    }
    
    pub fn negative_lookbehind(pattern: Pattern, default_field: Field) -> Self {
        Self {
            lookahead_type: LookaheadType::NegativeLookbehind,
            assertion_pattern: pattern,
            default_field,
        }
    }
}

impl Query for LookaheadQuery {
    fn weight(&self, _scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        Ok(Box::new(LookaheadWeight {
            lookahead_type: self.lookahead_type.clone(),
            assertion_pattern: self.assertion_pattern.clone(),
            default_field: self.default_field,
        }))
    }
}

// QueryClone is automatically implemented by Tantivy for Clone types

struct LookaheadWeight {
    lookahead_type: LookaheadType,
    assertion_pattern: Pattern,
    default_field: Field,
}

impl Weight for LookaheadWeight {
    fn scorer(&self, reader: &SegmentReader, _boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        let mut scorer = LookaheadScorer {
            lookahead_type: self.lookahead_type.clone(),
            assertion_pattern: self.assertion_pattern.clone(),
            default_field: self.default_field,
            reader: reader.clone(),
            current_doc: None,
            all_docs: reader.max_doc(),
            next_doc_id: 0,
        };
        let _ = scorer.advance();
        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("LookaheadQuery", Score::default()))
    }
}

pub struct LookaheadScorer {
    lookahead_type: LookaheadType,
    assertion_pattern: Pattern,
    default_field: Field,
    reader: SegmentReader,
    current_doc: Option<DocId>,
    all_docs: DocId,
    next_doc_id: DocId,
}

impl Scorer for LookaheadScorer {
    fn score(&mut self) -> Score {
        1.0
    }
}

impl DocSet for LookaheadScorer {
    fn advance(&mut self) -> DocId {
        // Iterate through documents, checking assertion at each position
        while self.next_doc_id < self.all_docs {
            let doc_id = self.next_doc_id;
            self.next_doc_id += 1;
            
            // For lookahead assertions, we need to verify that the assertion pattern
            // matches at the expected position. This is done by extracting tokens
            // and checking if the pattern matches.
            if self.check_assertion(doc_id) {
                self.current_doc = Some(doc_id);
                return doc_id;
            }
        }
        
        self.current_doc = None;
        tantivy::TERMINATED
    }

    fn doc(&self) -> DocId {
        self.current_doc.unwrap_or(tantivy::TERMINATED)
    }

    fn size_hint(&self) -> u32 {
        self.all_docs
    }
}

impl LookaheadScorer {
    fn check_assertion(&self, doc_id: DocId) -> bool {
        // Get the document
        let store_reader = match self.reader.get_store_reader(1) {
            Ok(reader) => reader,
            Err(_) => return false,
        };
        let doc = match store_reader.get(doc_id) {
            Ok(doc) => doc,
            Err(_) => return false,
        };
        
        // Extract tokens from the default field
        let tokens = self.extract_tokens_from_field(&doc);
        if tokens.is_empty() {
            return false;
        }
        
        // Find positions matching the assertion pattern
        let positions = self.find_positions_matching_pattern(&tokens, &self.assertion_pattern);
        
        // For lookahead assertions, just check if there are any matches
        // The actual position-relative filtering is done at a higher level
        match self.lookahead_type {
            LookaheadType::PositiveLookahead | LookaheadType::PositiveLookbehind => {
                !positions.is_empty()
            }
            LookaheadType::NegativeLookahead | LookaheadType::NegativeLookbehind => {
                positions.is_empty()
            }
        }
    }
    
    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument) -> Vec<String> {
        doc.get_all(self.default_field)
            .filter_map(|v| Value::as_str(&v).map(|s| s.to_string()))
            .collect()
    }
    
    fn find_positions_matching_pattern(&self, tokens: &[String], pattern: &Pattern) -> Vec<usize> {
        use crate::compiler::ast::{Constraint, Matcher};
        
        let mut positions = Vec::new();
        match pattern {
            Pattern::Constraint(Constraint::Field { name, matcher }) => {
                if name != "word" { return positions; }
                match matcher {
                    Matcher::String(s) => {
                        for (i, token) in tokens.iter().enumerate() {
                            if token == s {
                                positions.push(i);
                            }
                        }
                    }
                    Matcher::Regex { pattern, .. } => {
                        if let Ok(re) = regex::Regex::new(pattern) {
                            for (i, token) in tokens.iter().enumerate() {
                                if re.is_match(token) {
                                    positions.push(i);
                                }
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
