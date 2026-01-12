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
            return false;
        }
        // Find all valid spans matching the concatenated pattern
        let all_spans = find_constraint_spans_in_sequence(&self.pattern, &tokens);
        for match_seq in all_spans {
            self.current_doc_matches.extend(match_seq);
        }
        !self.current_doc_matches.is_empty()
    }

    fn extract_tokens_from_field(&self, doc: &tantivy::schema::TantivyDocument, field_name: &str) -> Vec<String> {
        if let Ok(field) = self.reader.schema().get_field(field_name) {
            doc.get_all(field)
                .filter_map(|v| Value::as_str(&v).map(|s| s.to_string()))  // Fixed: added & before v
                .collect()
        } else {
            vec![]
        }
    }

    pub fn get_current_doc_matches(&self) -> &[crate::types::SpanWithCaptures] {
        &self.current_doc_matches
    }
}

// Consider renaming this function or moving it to avoid the duplicate export warning
pub fn find_constraint_spans_in_sequence(pattern: &crate::compiler::ast::Pattern, tokens: &[String]) -> Vec<Vec<crate::types::SpanWithCaptures>> {
    use crate::compiler::ast::Pattern;
    if let Pattern::Concatenated(patterns) = pattern {
        let mut results = Vec::new();
        let len = tokens.len();
        for start in 0..len {
            let mut pos = start;
            let mut match_seq = Vec::new();
            let mut matched = true;
            for pat in patterns {
                match pat {
                    Pattern::Constraint(crate::compiler::ast::Constraint::Field { name, matcher }) => {
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
                            // Use pre-compiled regex from Matcher for performance
                            crate::compiler::ast::Matcher::Regex { regex, .. } => {
                                if pos < len && regex.is_match(&tokens[pos]) {
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