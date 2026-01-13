use tantivy::query::{Query, Weight, Scorer, EnableScoring};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet};
use tantivy::schema::{Field, Value};
use crate::compiler::ast::Pattern;

#[derive(Debug)]

pub struct RustieConcatQuery {
    pub default_field: Field,
    pub pattern: Pattern,
    pub sub_queries: Vec<Box<dyn Query>>,
}

impl Clone for RustieConcatQuery {
    fn clone(&self) -> Self {
        RustieConcatQuery {
            default_field: self.default_field,
            pattern: self.pattern.clone(),
            sub_queries: self.sub_queries.iter().map(|q| q.box_clone()).collect(),
        }
    }
}

impl RustieConcatQuery {
    pub fn new(
        default_field: Field,
        pattern: Pattern,
        sub_queries: Vec<Box<dyn Query>>,
    ) -> Self {
        Self {
            default_field,
            pattern,
            sub_queries,
        }
    }
}

impl Query for RustieConcatQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let sub_weights: Vec<Box<dyn Weight>> = self.sub_queries
            .iter()
            .map(|q| q.weight(scoring.clone()))
            .collect::<TantivyResult<Vec<_>>>()?;

        Ok(Box::new(RustieConcatWeight {
            sub_weights,
            pattern: self.pattern.clone(),
            default_field: self.default_field,
        }))
    }
}

struct RustieConcatWeight {
    sub_weights: Vec<Box<dyn Weight>>,
    pattern: Pattern,
    default_field: Field,
}

impl Weight for RustieConcatWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        let sub_scorers: Vec<Box<dyn Scorer>> = self.sub_weights
            .iter()
            .map(|w| w.scorer(reader, boost))
            .collect::<TantivyResult<Vec<_>>>()?;

        let mut scorer = RustieConcatScorer {
            sub_scorers,
            pattern: self.pattern.clone(),
            default_field: self.default_field,
            reader: reader.clone(),
            current_doc: None,
            current_matches: Vec::new(),
            match_index: 0,
            current_doc_matches: Vec::new(),
            boost, // Pass boost for Odinson-style scoring
        };
        let _ = scorer.advance();
        Ok(Box::new(scorer))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        Ok(tantivy::query::Explanation::new("RustieConcatQuery", Score::default()))
    }
}

pub struct RustieConcatScorer {
    sub_scorers: Vec<Box<dyn Scorer>>,
    pattern: Pattern,
    default_field: Field,
    reader: SegmentReader,
    current_doc: Option<DocId>,
    current_matches: Vec<(DocId, Score)>,
    match_index: usize,
    current_doc_matches: Vec<crate::types::SpanWithCaptures>,
    /// Boost factor from weight creation
    boost: Score,
}

impl RustieConcatScorer {
    /// Compute sloppy frequency factor based on span width (Odinson-style)
    /// Uses the formula: 1.0 / (1.0 + distance) where distance = span width
    /// Shorter spans get higher scores
    fn compute_slop_factor(span_width: usize) -> Score {
        1.0 / (1.0 + span_width as f32)
    }

    /// Compute Odinson-style score for the current document
    /// Accumulates sloppy frequency for all matches, similar to Lucene/Odinson
    fn compute_odinson_score(&self) -> Score {
        if self.current_doc_matches.is_empty() {
            return 0.0;
        }

        // Accumulate sloppy frequency for all matches (Odinson approach)
        let mut acc_sloppy_freq: Score = 0.0;

        for span_match in &self.current_doc_matches {
            let span_width = span_match.span.end.saturating_sub(span_match.span.start);
            acc_sloppy_freq += Self::compute_slop_factor(span_width);
        }

        // Base score from pattern matching
        let base_score = 1.0;

        // Final score: base_score * accumulated_sloppy_freq * boost
        // This follows Odinson's: docScorer.score(docID(), accSloppyFreq)
        let final_score = base_score * acc_sloppy_freq * self.boost;

        // Ensure minimum score of 1.0 for any match (normalized)
        final_score.max(1.0)
    }
}

impl Scorer for RustieConcatScorer {
    fn score(&mut self) -> Score {
        if let Some((_, score)) = self.current_matches.get(self.match_index) {
            *score
        } else {
            Score::default()
        }
    }
}

impl DocSet for RustieConcatScorer {
    fn advance(&mut self) -> DocId {
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
                    // Compute Odinson-style score based on span widths and match count
                    let score = self.compute_odinson_score();
                    self.current_matches.push((doc, score));
                    self.match_index = self.current_matches.len() - 1;
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

impl RustieConcatScorer {
    fn check_pattern_matching(&mut self, doc_id: DocId) -> bool {
        println!("DEBUG: check_pattern_matching called for doc_id={} with pattern={:?}", doc_id, self.pattern);
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

        // Extract and split tokens for all potential fields mentioned in the pattern
        let mut field_cache = std::collections::HashMap::new();
        let field_names = ["word", "lemma", "pos", "tag", "chunk", "entity", "norm"];
        for name in field_names {
            let tokens = crate::tantivy_integration::utils::extract_field_values(self.reader.schema(), &doc, name);
            if !tokens.is_empty() {
                field_cache.insert(name.to_string(), tokens);
            }
        }

        // Find all valid spans matching the concatenated pattern
        let all_spans = find_constraint_spans_in_sequence(&self.pattern, &field_cache);
        self.current_doc_matches = all_spans;
        
        !self.current_doc_matches.is_empty()
    }

    fn extract_tokens_from_field(&self, _doc: &tantivy::schema::TantivyDocument, _field_name: &str) -> Vec<String> {
        // Deprecated: field extraction now happens in check_pattern_matching using the cache
        vec![]
    }

    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        &self.current_doc_matches
    }
}

pub fn find_constraint_spans_in_sequence(
    pattern: &crate::compiler::ast::Pattern, 
    field_cache: &std::collections::HashMap<String, Vec<String>>
) -> Vec<crate::types::SpanWithCaptures> {
    use crate::compiler::ast::Pattern;
    
    if let Pattern::Concatenated(patterns) = pattern {
        let mut results = Vec::new();
        
        // Use 'word' field to determine sentence length as it's the most reliable
        let len = field_cache.get("word").or_else(|| field_cache.values().next()).map(|v| v.len()).unwrap_or(0);
        if len == 0 { return results; }

        for start in 0..len {
            let mut pos = start;
            let mut all_captures = Vec::new();
            let mut matched = true;
            
            for (idx, pat) in patterns.iter().enumerate() {
                if pos >= len {
                    matched = false;
                    break;
                }
                
                if let Some(span_with_caps) = matches_pattern_at_position(pat, field_cache, pos) {
                    pos = span_with_caps.span.end;
                    all_captures.extend(span_with_caps.captures);
                } else {
                    // Log failure for debugging
                    if patterns.len() < 4 { // Only for relatively simple patterns
                         let p_strs: Vec<String> = patterns.iter().map(|p| format!("{:?}", p)).collect();
                         let is_target = p_strs.iter().any(|s| s.contains("JJ") || s.contains("NNP") || s.contains("VBZ"));
                         if is_target {
                             println!("DEBUG: Sequence match fail. Patterns: {:?} | start: {}, failed_at_idx: {}, failed_at_pos: {}", p_strs, start, idx, pos);
                         }
                    }
                    matched = false;
                    break;
                }
            }
            
            if matched {
                let full_span = crate::types::Span { start, end: pos };
                results.push(crate::types::SpanWithCaptures::with_captures(full_span, all_captures));
            }
        }
        results
    } else {
        Vec::new()
    }
}

fn matches_pattern_at_position(
    pattern: &crate::compiler::ast::Pattern,
    field_cache: &std::collections::HashMap<String, Vec<String>>,
    pos: usize
) -> Option<crate::types::SpanWithCaptures> {
    use crate::compiler::ast::{Pattern, Constraint};
    
    match pattern {
        Pattern::Constraint(constraint) => {
            if matches_constraint_at_position(constraint, field_cache, pos) {
                let span = crate::types::Span { start: pos, end: pos + 1 };
                let capture = crate::types::NamedCapture::new(format!("c{}", pos), span.clone());
                Some(crate::types::SpanWithCaptures::with_captures(span, vec![capture]))
            } else {
                None
            }
        }
        Pattern::NamedCapture { name, pattern } => {
            if let Some(mut m) = matches_pattern_at_position(pattern, field_cache, pos) {
                // Add the named capture
                let capture = crate::types::NamedCapture::new(name.clone(), m.span.clone());
                m.captures.push(capture);
                Some(m)
            } else {
                None
            }
        }
        Pattern::Disjunctive(patterns) => {
            for pat in patterns {
                if let Some(m) = matches_pattern_at_position(pat, field_cache, pos) {
                    return Some(m);
                }
            }
            None
        }
        Pattern::Repetition { pattern, min, max } => {
            // Very simple greedy repetition for matching at current position
            // This is just to support basic concatenation of repetitions
            let mut current_pos = pos;
            let mut count = 0;
            let mut total_captures = Vec::new();
            
            while let Some(m) = matches_pattern_at_position(pattern, field_cache, current_pos) {
                current_pos = m.span.end;
                total_captures.extend(m.captures);
                count += 1;
                if let Some(max_val) = max {
                    if count >= *max_val { break; }
                }
            }
            
            if count >= *min {
                let span = crate::types::Span { start: pos, end: current_pos };
                Some(crate::types::SpanWithCaptures::with_captures(span, total_captures))
            } else {
                None
            }
        }
        Pattern::Assertion(assertion) => {
            use crate::compiler::ast::Assertion;
            match assertion {
                Assertion::PositiveLookahead(child) => {
                    if matches_pattern_at_position(child, field_cache, pos).is_some() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
                Assertion::NegativeLookahead(child) => {
                    if matches_pattern_at_position(child, field_cache, pos).is_none() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
                Assertion::PositiveLookbehind(child) => {
                    if pos > 0 && matches_pattern_at_position(child, field_cache, pos - 1).is_some() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
                Assertion::NegativeLookbehind(child) => {
                    if pos == 0 || matches_pattern_at_position(child, field_cache, pos - 1).is_none() {
                        Some(crate::types::SpanWithCaptures::new(crate::types::Span { start: pos, end: pos }))
                    } else {
                        None
                    }
                }
            }
        }
        Pattern::Concatenated(nested_patterns) => {
            // Support nested concatenations
            let mut current_pos = pos;
            let mut all_captures = Vec::new();
            for pat in nested_patterns {
                if let Some(m) = matches_pattern_at_position(pat, field_cache, current_pos) {
                    current_pos = m.span.end;
                    all_captures.extend(m.captures);
                } else {
                    return None;
                }
            }
            Some(crate::types::SpanWithCaptures::with_captures(crate::types::Span { start: pos, end: current_pos }, all_captures))
        }
        _ => None,
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
                    let result = matcher.matches(&tokens[pos]);
                    if !result && (name == "tag" || name == "word") {
                        println!("DEBUG: Match failed: field={}, pos={}, token='{}', matcher={:?}", name, pos, tokens[pos], matcher);
                    }
                    result
                } else {
                    false
                }
            } else {
                // If it's looking for 'tag' but we only have 'pos' in cache, try fallback
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
