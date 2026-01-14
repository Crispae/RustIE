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
        
        // Extract and split tokens for all potential fields
        let mut field_cache = std::collections::HashMap::new();
        let field_names = ["word", "lemma", "pos", "tag", "chunk", "entity", "norm"];
        for name in field_names {
            let tokens = crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), &doc, name);
            if !tokens.is_empty() {
                field_cache.insert(name.to_string(), tokens);
            }
        }
        
        if field_cache.is_empty() {
            return false;
        }
        
        // Find positions matching the assertion pattern
        let positions = self.find_positions_matching_pattern(&field_cache, &self.assertion_pattern);
        
        // For lookahead assertions, just check if there are any matches
        // The actual position-relative filtering is done at a higher level
        // Negative assertions should not exclude documents at the pre-filtering stage
        // because the exclusion is position-relative.
        match self.lookahead_type {
            LookaheadType::PositiveLookahead | LookaheadType::PositiveLookbehind => {
                !positions.is_empty()
            }
            LookaheadType::NegativeLookahead | LookaheadType::NegativeLookbehind => {
                true 
            }
        }
    }
    
    fn extract_tokens_from_field(&self, _doc: &tantivy::schema::TantivyDocument) -> Vec<String> {
        // Deprecated: field extraction now happens in check_assertion using the cache
        vec![]
    }
    
    fn find_positions_matching_pattern(&self, field_cache: &std::collections::HashMap<String, Vec<String>>, pattern: &Pattern) -> Vec<usize> {
        use crate::compiler::ast::{Constraint, Matcher};
        
        let mut positions = Vec::new();
        
        // Use 'word' field to determine length
        let len = field_cache.get("word").or_else(|| field_cache.values().next()).map(|v| v.len()).unwrap_or(0);
        if len == 0 { return positions; }

        match pattern {
            Pattern::Constraint(constraint) => {
                for i in 0..len {
                    if matches_constraint_at_position(constraint, field_cache, i) {
                        positions.push(i);
                    }
                }
            }
            Pattern::NamedCapture { pattern, .. } => {
                return self.find_positions_matching_pattern(field_cache, pattern);
            }
            Pattern::Disjunctive(patterns) => {
                for pat in patterns {
                    positions.extend(self.find_positions_matching_pattern(field_cache, pat));
                }
                positions.sort();
                positions.dedup();
            }
            _ => {}
        }
        positions
    }
}

fn matches_constraint_at_position(
    constraint: &crate::compiler::ast::Constraint,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize
) -> bool {
    use crate::compiler::ast::Constraint;
    
    match constraint {
        Constraint::Wildcard => true,
        Constraint::Field { name, matcher } => {
            if let Some(tokens) = field_cache.get(name) {
                if pos < tokens.len() {
                    matcher.matches(&tokens[pos])
                } else {
                    false
                }
            } else {
                // Try fallback for 'tag' -> 'pos'
                if name == "tag" {
                    if let Some(tokens) = field_cache.get("pos") {
                        if pos < tokens.len() {
                            return matcher.matches(&tokens[pos]);
                        }
                    }
                }
                false
            }
        }
        Constraint::Fuzzy { name, matcher } => {
            if let Some(tokens) = field_cache.get(name) {
                if pos < tokens.len() {
                    tokens[pos].to_lowercase().contains(&matcher.to_lowercase())
                } else {
                    false
                }
            } else {
                false
            }
        }
        Constraint::Negated(inner) => !matches_constraint_at_position(inner, field_cache, pos),
        Constraint::Conjunctive(constraints) => {
            constraints.iter().all(|c| matches_constraint_at_position(c, field_cache, pos))
        }
        Constraint::Disjunctive(constraints) => {
            constraints.iter().any(|c| matches_constraint_at_position(c, field_cache, pos))
        }
    }
}
