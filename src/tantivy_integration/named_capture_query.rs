use tantivy::query::{Query, Weight, Scorer, EnableScoring};
use tantivy::{DocId, Score, SegmentReader, Result as TantivyResult, DocSet};
use tantivy::schema::Field;
use crate::query::ast::Pattern;

/// Query that wraps an inner query and tags its matches with a capture name.
///
/// Following the Odinson approach, captures are computed lazily at result-collection
/// time rather than eagerly during scorer iteration.  The scorer is a thin
/// pass-through to the inner scorer — it only finds matching documents and
/// delegates scoring.  Capture extraction happens in the execution layer using
/// already-loaded document data, eliminating redundant document-store reads.
#[derive(Debug)]
pub struct RustieNamedCaptureQuery {
    pub inner_query: Box<dyn Query>,
    pub capture_name: String,
    pub pattern: Pattern,
    pub default_field: Field,
}

impl Clone for RustieNamedCaptureQuery {
    fn clone(&self) -> Self {
        RustieNamedCaptureQuery {
            inner_query: self.inner_query.box_clone(),
            capture_name: self.capture_name.clone(),
            pattern: self.pattern.clone(),
            default_field: self.default_field,
        }
    }
}

impl RustieNamedCaptureQuery {
    pub fn new(
        inner_query: Box<dyn Query>,
        capture_name: String,
        pattern: Pattern,
        default_field: Field,
    ) -> Self {
        Self {
            inner_query,
            capture_name,
            pattern,
            default_field,
        }
    }

    /// Returns the concrete weight without boxing. Used by execution to call concrete_scorer per segment.
    pub(crate) fn concrete_weight(
        &self,
        searcher: &tantivy::Searcher,
    ) -> TantivyResult<RustieNamedCaptureWeight> {
        let scoring = EnableScoring::Enabled {
            searcher,
            statistics_provider: searcher,
        };
        let inner_weight = self.inner_query.weight(scoring)?;
        Ok(RustieNamedCaptureWeight {
            inner_weight,
            default_field: self.default_field,
        })
    }

    /// Compute named captures for a document given its already-loaded token data.
    /// This avoids redundant document-store reads since the execution layer already
    /// loads the document for `extract_sentence_result`.
    pub fn compute_captures(
        capture_name: &str,
        pattern: &Pattern,
        field_name: &str,
        tokens: &[String],
    ) -> Vec<crate::types::SpanWithCaptures> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let match_positions = pattern.extract_matching_positions(field_name, tokens);

        match_positions
            .into_iter()
            .map(|pos| {
                let span = crate::types::Span { start: pos, end: pos + 1 };
                let capture = crate::types::NamedCapture::new(
                    capture_name.to_string(),
                    span.clone(),
                );
                crate::types::SpanWithCaptures::with_captures(span, vec![capture])
            })
            .collect()
    }
}

impl Query for RustieNamedCaptureQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> TantivyResult<Box<dyn Weight>> {
        let inner_weight = self.inner_query.weight(scoring)?;

        Ok(Box::new(RustieNamedCaptureWeight {
            inner_weight,
            default_field: self.default_field,
        }))
    }
}

// QueryClone is automatically implemented by Tantivy for Clone types

pub(crate) struct RustieNamedCaptureWeight {
    inner_weight: Box<dyn Weight>,
    default_field: Field,
}

impl RustieNamedCaptureWeight {
    /// Build the concrete scorer for this segment.
    /// The scorer is a thin pass-through — no document-store access, no eager match computation.
    pub(crate) fn concrete_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> TantivyResult<RustieNamedCaptureScorer> {
        let inner_scorer = self.inner_weight.scorer(reader, boost)?;
        Ok(RustieNamedCaptureScorer {
            inner_scorer,
        })
    }

    pub fn default_field(&self) -> Field {
        self.default_field
    }
}

impl Weight for RustieNamedCaptureWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> TantivyResult<Box<dyn Scorer>> {
        Ok(Box::new(self.concrete_scorer(reader, boost)?))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> TantivyResult<tantivy::query::Explanation> {
        self.inner_weight.explain(reader, doc)
    }
}

/// Thin pass-through scorer: delegates document iteration and scoring to the
/// inner scorer.  No document-store access, no eager match computation.
/// Captures are computed lazily in the execution layer via
/// `RustieNamedCaptureQuery::compute_captures`.
pub struct RustieNamedCaptureScorer {
    inner_scorer: Box<dyn Scorer>,
}

impl Scorer for RustieNamedCaptureScorer {
    fn score(&mut self) -> Score {
        self.inner_scorer.score()
    }
}

impl DocSet for RustieNamedCaptureScorer {
    fn advance(&mut self) -> DocId {
        self.inner_scorer.advance()
    }

    fn doc(&self) -> DocId {
        self.inner_scorer.doc()
    }

    fn size_hint(&self) -> u32 {
        self.inner_scorer.size_hint()
    }
}
