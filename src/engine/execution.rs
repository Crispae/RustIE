//! Query execution methods for ExtractorEngine

use crate::engine::constants::*;
use crate::engine::core::ExtractorEngine;
use crate::engine::pagination_driver::{PaginationDriver, Paginatable};
use crate::query::compiler::CompiledQuery;
use crate::results::rustie_results::SentenceResult;
use crate::tantivy_integration::concat_query::{RustieConcatQuery, RustieConcatWeight};
use crate::tantivy_integration::graph_traversal::{
    OptimizedGraphTraversalQuery, OptimizedGraphTraversalWeight,
};
use crate::tantivy_integration::named_capture_query::{
    RustieNamedCaptureQuery, RustieNamedCaptureWeight,
};
use crate::tantivy_integration::paging_collector::{
    PaginatedSearchResult, PagingCollector, ScoredDoc, SimpleCollector,
};
use crate::types::SearchCursor;
use anyhow::Result;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;
use tantivy::{query::{Query, Scorer}, DocAddress, DocSet, Score};

/// Result of a paginated search: total hits, one page of sentence results, optional next cursor.
pub struct PaginatedResult {
    pub total_hits: usize,
    pub sentence_results: Vec<SentenceResult>,
    pub next_cursor: Option<SearchCursor>,
}

/// A sentence result with its DocAddress and score, sortable in the same global order
/// as ScoredDoc (score DESC, segment ASC, doc ASC). Used by execute_graph_traversal_paginated
/// to merge, dedup, skip, and page graph traversal results while preserving cursor compatibility.
#[derive(Clone)]
struct ScoredSentence {
    score: Score,
    address: DocAddress,
    sentence_result: SentenceResult,
}

impl ScoredSentence {
    fn new(score: Score, address: DocAddress, sentence_result: SentenceResult) -> Self {
        Self {
            score,
            address,
            sentence_result,
        }
    }

    fn as_scored_doc(&self) -> ScoredDoc {
        ScoredDoc {
            score: self.score,
            address: self.address,
        }
    }

    fn dedup_key(&self) -> (Arc<str>, Arc<str>) {
        (
            self.sentence_result.document_id.clone(),
            self.sentence_result.sentence_id.clone(),
        )
    }
}

impl PartialEq for ScoredSentence {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.address == other.address
    }
}

impl Eq for ScoredSentence {}

impl PartialOrd for ScoredSentence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredSentence {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.address.segment_ord.cmp(&other.address.segment_ord))
            .then_with(|| self.address.doc_id.cmp(&other.address.doc_id))
    }
}

impl Paginatable for ScoredSentence {
    fn score(&self) -> f32 {
        self.score
    }

    fn doc_address(&self) -> DocAddress {
        self.address
    }

    fn dedup_key(&self) -> Option<(String, String)> {
        Some((
            self.sentence_result.document_id.to_string(),
            self.sentence_result.sentence_id.to_string(),
        ))
    }
}

impl ExtractorEngine {
    /// Execute a query with Odinson-style cursor-based pagination.
    /// Supports all pattern types: basic constraints, graph traversal, concatenated, and named captures.
    pub fn query_paginated(
        &self,
        query: &str,
        page_size: usize,
        after: Option<SearchCursor>,
    ) -> Result<PaginatedResult> {
        let pattern = self.parser().parse_query(query)?;
        let compiled = self.compiler().compile_pattern(&pattern)?;

        match &compiled {
            CompiledQuery::Graph(gq) => {
                self.execute_graph_traversal_paginated(gq, page_size, after)
            }
            CompiledQuery::Concat(cq) => {
                self.execute_concat_paginated(cq, page_size, after)
            }
            CompiledQuery::NamedCapture(nq) => {
                self.execute_named_capture_paginated(nq, page_size, after)
            }
            CompiledQuery::Basic(bq) => {
                self.execute_paginated(bq.as_ref(), page_size, after, &pattern)
            }
        }
    }

    /// Execute a compiled query with pagination: use paging collector, load only one page of docs.
    fn execute_paginated(
        &self,
        query: &dyn Query,
        page_size: usize,
        after: Option<SearchCursor>,
        pattern: &crate::query::ast::Pattern,
    ) -> Result<PaginatedResult> {
        let searcher = self.reader.searcher();
        let search_result: PaginatedSearchResult = match &after {
            None => searcher
                .search(query, &SimpleCollector::new(page_size))
                .map_err(anyhow::Error::from)?,
            Some(cursor) => searcher
                .search(query, &PagingCollector::new(page_size, cursor.clone()))
                .map_err(anyhow::Error::from)?,
        };

        let needs_word_positions = pattern.references_field(FIELD_WORD);
        let mut sentence_results = Vec::with_capacity(search_result.scored_docs.len());

        for scored_doc in &search_result.scored_docs {
            let doc = self.doc(scored_doc.address)?;
            let mut result = self.extract_sentence_result(&doc, scored_doc.score)?;
            if needs_word_positions {
                let tokens = self.extract_field_values(&doc, FIELD_WORD);
                let match_positions = pattern.extract_matching_positions(FIELD_WORD, &tokens);
                let mut matches = Vec::new();
                for (i, &start) in match_positions.iter().enumerate() {
                    let span = crate::types::Span {
                        start,
                        end: start + 1,
                    };
                    let capture = crate::types::NamedCapture::new(format!("c{}", i), span.clone());
                    matches.push(crate::types::SpanWithCaptures::with_captures(span, vec![capture]));
                }
                result.matches = matches;
            }
            sentence_results.push(result);
        }

        let next_cursor = search_result
            .scored_docs
            .last()
            .and_then(|last| {
                PaginationDriver::next_cursor_for_page(
                    last,
                    search_result.total_hits,
                    page_size,
                    search_result.scored_docs.len(),
                )
            });

        Ok(PaginatedResult {
            total_hits: search_result.total_hits,
            sentence_results,
            next_cursor,
        })
    }

    /// Best-effort paginated execution for graph traversal queries.
    /// Reuses the parallel segment loop from execute_graph_traversal, then sorts by
    /// (score DESC, segment_ord ASC, doc_id ASC), deduplicates by (document_id, sentence_id),
    /// records total_hits, skips past cursor, takes page_size, and builds next_cursor.
    fn execute_graph_traversal_paginated(
        &self,
        graph_query: &OptimizedGraphTraversalQuery,
        page_size: usize,
        after: Option<SearchCursor>,
    ) -> Result<PaginatedResult> {
        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        let score_field = self.default_field();
        let weight: Arc<OptimizedGraphTraversalWeight> = Arc::new(
            graph_query
                .concrete_weight_with_bm25(&searcher, self.regex_cache.clone(), score_field)
                .or_else(|_| graph_query.concrete_weight_with_cache(self.regex_cache.clone()))
                .map_err(anyhow::Error::from)?,
        );

        let segment_results: Vec<Vec<ScoredSentence>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);
                let mut scorer = match weight.concrete_scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!(
                            "graph traversal paginated: segment {segment_ord} scorer creation failed: {e}"
                        );
                        return None;
                    }
                };

                let mut segment_scored = Vec::new();

                loop {
                    let doc_id = scorer.doc();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);
                    let matches = scorer.take_current_doc_matches();

                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) =
                            self.extract_sentence_result(&doc, score)
                        {
                            sentence_result.matches = matches;
                            segment_scored.push(ScoredSentence::new(
                                score,
                                doc_address,
                                sentence_result,
                            ));
                        }
                    }

                    if scorer.advance() == tantivy::TERMINATED {
                        break;
                    }
                }

                Some(segment_scored)
            })
            .collect();

        let mut all_scored: Vec<ScoredSentence> = segment_results
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect();

        all_scored.sort();

        let driver = PaginationDriver::new(page_size, after);
        let page = driver.paginate(all_scored);

        let sentence_results = page
            .items
            .into_iter()
            .map(|ss| ss.sentence_result)
            .collect();

        Ok(PaginatedResult {
            total_hits: page.total_hits,
            sentence_results,
            next_cursor: page.next_cursor,
        })
    }

    /// Paginated execution for concatenated pattern queries.
    /// Same parallel-segment approach as the non-paginated variant, then sort + paginate.
    fn execute_concat_paginated(
        &self,
        pattern_query: &RustieConcatQuery,
        page_size: usize,
        after: Option<SearchCursor>,
    ) -> Result<PaginatedResult> {
        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        let weight: Arc<RustieConcatWeight> = Arc::new(
            pattern_query.concrete_weight(&searcher, self.regex_cache.clone()).map_err(anyhow::Error::from)?,
        );

        let segment_results: Vec<Vec<ScoredSentence>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);
                let mut scorer = match weight.concrete_scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!(
                            "concat paginated: segment {segment_ord} scorer creation failed: {e}"
                        );
                        return None;
                    }
                };

                let mut segment_scored = Vec::new();

                loop {
                    let doc_id = scorer.advance();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);
                    let matches = scorer.take_current_doc_matches();

                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) =
                            self.extract_sentence_result(&doc, score)
                        {
                            sentence_result.matches = matches;
                            segment_scored.push(ScoredSentence::new(
                                score,
                                doc_address,
                                sentence_result,
                            ));
                        }
                    }
                }

                Some(segment_scored)
            })
            .collect();

        let mut all_scored: Vec<ScoredSentence> = segment_results
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect();

        all_scored.sort();

        let driver = PaginationDriver::new(page_size, after);
        let page = driver.paginate(all_scored);

        let sentence_results = page
            .items
            .into_iter()
            .map(|ss| ss.sentence_result)
            .collect();

        Ok(PaginatedResult {
            total_hits: page.total_hits,
            sentence_results,
            next_cursor: page.next_cursor,
        })
    }

    /// Paginated execution for named capture queries.
    /// Same parallel-segment approach as the non-paginated variant, then sort + paginate.
    fn execute_named_capture_paginated(
        &self,
        named_query: &RustieNamedCaptureQuery,
        page_size: usize,
        after: Option<SearchCursor>,
    ) -> Result<PaginatedResult> {
        let searcher = self.reader.searcher();
        let num_segments = searcher.segment_readers().len();

        let weight: Arc<RustieNamedCaptureWeight> = Arc::new(
            named_query.concrete_weight(&searcher).map_err(anyhow::Error::from)?,
        );

        let capture_name = named_query.capture_name.clone();
        let pattern = named_query.pattern.clone();

        let segment_results: Vec<Vec<ScoredSentence>> = (0..num_segments)
            .into_par_iter()
            .filter_map(|segment_ord| {
                let segment_reader = searcher.segment_reader(segment_ord as u32);
                let mut scorer = match weight.concrete_scorer(segment_reader, 1.0) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!(
                            "named capture paginated: segment {segment_ord} scorer creation failed: {e}"
                        );
                        return None;
                    }
                };

                let field_name = searcher
                    .schema()
                    .get_field_name(weight.default_field())
                    .to_string();

                let mut segment_scored = Vec::new();

                loop {
                    let doc_id = scorer.advance();
                    if doc_id == tantivy::TERMINATED {
                        break;
                    }

                    let score = scorer.score();
                    let doc_address = DocAddress::new(segment_ord as u32, doc_id);

                    if let Ok(doc) = searcher.doc(doc_address) {
                        if let Ok(mut sentence_result) =
                            self.extract_sentence_result(&doc, score)
                        {
                            let tokens = self.extract_field_values(&doc, &field_name);
                            sentence_result.matches = RustieNamedCaptureQuery::compute_captures(
                                &capture_name,
                                &pattern,
                                &field_name,
                                &tokens,
                            );
                            segment_scored.push(ScoredSentence::new(
                                score,
                                doc_address,
                                sentence_result,
                            ));
                        }
                    }
                }

                Some(segment_scored)
            })
            .collect();

        let mut all_scored: Vec<ScoredSentence> = segment_results
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect();

        all_scored.sort();

        let driver = PaginationDriver::new(page_size, after);
        let page = driver.paginate(all_scored);

        let sentence_results = page
            .items
            .into_iter()
            .map(|ss| ss.sentence_result)
            .collect();

        Ok(PaginatedResult {
            total_hits: page.total_hits,
            sentence_results,
            next_cursor: page.next_cursor,
        })
    }
}

#[cfg(test)]
mod graph_pagination_tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn make_sentence_result(
        document_id: &str,
        sentence_id: &str,
        score: f32,
    ) -> SentenceResult {
        SentenceResult::new(
            Arc::from(document_id),
            Arc::from(sentence_id),
            score,
            vec![],
            HashMap::new(),
        )
    }

    fn scored_sentence(
        score: f32,
        seg: u32,
        doc: u32,
        doc_id: &str,
        sent_id: &str,
    ) -> ScoredSentence {
        ScoredSentence::new(
            score,
            DocAddress::new(seg, doc),
            make_sentence_result(doc_id, sent_id, score),
        )
    }

    #[test]
    fn scored_sentence_ordering_matches_scored_doc() {
        let a = scored_sentence(5.0, 0, 10, "d1", "s1");
        let b = scored_sentence(5.0, 0, 20, "d1", "s2");
        let c = scored_sentence(3.0, 1, 5, "d2", "s1");

        let mut items = vec![c.clone(), a.clone(), b.clone()];
        items.sort();

        assert_eq!(items[0].address, DocAddress::new(0, 10));
        assert_eq!(items[1].address, DocAddress::new(0, 20));
        assert_eq!(items[2].address, DocAddress::new(1, 5));
    }

    #[test]
    fn dedup_keeps_first_occurrence() {
        let items = vec![
            scored_sentence(5.0, 0, 10, "doc1", "sent1"),
            scored_sentence(3.0, 0, 20, "doc1", "sent1"),
            scored_sentence(4.0, 1, 5, "doc2", "sent1"),
        ];

        let mut sorted = items;
        sorted.sort();

        let mut seen = HashSet::new();
        sorted.retain(|ss| seen.insert(ss.dedup_key()));

        assert_eq!(sorted.len(), 2);
        assert_eq!(
            sorted[0].dedup_key(),
            (Arc::from("doc1"), Arc::from("sent1"))
        );
        assert!((sorted[0].score - 5.0).abs() < 1e-6);
    }

    #[test]
    fn skip_past_cursor_works() {
        let items = vec![
            scored_sentence(5.0, 0, 10, "d1", "s1"),
            scored_sentence(4.0, 0, 20, "d2", "s1"),
            scored_sentence(3.0, 1, 5, "d3", "s1"),
            scored_sentence(2.0, 1, 10, "d4", "s1"),
        ];

        let cursor = SearchCursor::new(0, 20, 4.0);

        let after_index = items
            .iter()
            .position(|ss| !ss.as_scored_doc().should_skip(&cursor))
            .unwrap_or(items.len());

        let page: Vec<_> = items.into_iter().skip(after_index).take(2).collect();

        assert_eq!(page.len(), 2);
        assert!((page[0].score - 3.0).abs() < 1e-6);
        assert!((page[1].score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn full_graph_pagination_flow() {
        let mut all = vec![
            scored_sentence(5.0, 0, 1, "d1", "s1"),
            scored_sentence(4.5, 1, 1, "d2", "s1"),
            scored_sentence(4.0, 0, 2, "d3", "s1"),
            scored_sentence(3.0, 0, 3, "d4", "s1"),
            scored_sentence(2.0, 1, 2, "d5", "s1"),
        ];
        all.sort();
        let total_hits = all.len();

        let page1: Vec<_> = all.iter().take(2).cloned().collect();
        assert!((page1[0].score - 5.0).abs() < 1e-6);
        assert!((page1[1].score - 4.5).abs() < 1e-6);

        let cursor1 = SearchCursor::new(
            page1[1].address.segment_ord,
            page1[1].address.doc_id,
            page1[1].score,
        );

        let after_idx = all
            .iter()
            .position(|ss| !ss.as_scored_doc().should_skip(&cursor1))
            .unwrap_or(all.len());
        let page2: Vec<_> = all.iter().skip(after_idx).take(2).cloned().collect();
        assert!((page2[0].score - 4.0).abs() < 1e-6);
        assert!((page2[1].score - 3.0).abs() < 1e-6);

        let cursor2 = SearchCursor::new(
            page2[1].address.segment_ord,
            page2[1].address.doc_id,
            page2[1].score,
        );

        let after_idx2 = all
            .iter()
            .position(|ss| !ss.as_scored_doc().should_skip(&cursor2))
            .unwrap_or(all.len());
        let page3: Vec<_> = all.iter().skip(after_idx2).take(2).cloned().collect();
        assert_eq!(page3.len(), 1);
        assert!((page3[0].score - 2.0).abs() < 1e-6);

        let total_seen = page1.len() + page2.len() + page3.len();
        assert_eq!(total_seen, total_hits);
    }
}

#[cfg(test)]
mod unified_pagination_tests {
    use super::*;
    use crate::engine::pagination_driver::PaginationDriver;
    use crate::tantivy_integration::paging_collector::ScoredDoc;
    use std::collections::HashMap;
    use tantivy::DocAddress;

    fn make_sentence_result(
        document_id: &str,
        sentence_id: &str,
        score: f32,
    ) -> SentenceResult {
        SentenceResult::new(
            Arc::from(document_id),
            Arc::from(sentence_id),
            score,
            vec![],
            HashMap::new(),
        )
    }

    fn make_pair(
        score: f32,
        seg: u32,
        doc: u32,
        doc_id: &str,
        sent_id: &str,
    ) -> (ScoredDoc, ScoredSentence) {
        let scored_doc = ScoredDoc {
            score,
            address: DocAddress::new(seg, doc),
        };
        let scored_sentence = ScoredSentence::new(
            score,
            DocAddress::new(seg, doc),
            make_sentence_result(doc_id, sent_id, score),
        );
        (scored_doc, scored_sentence)
    }

    #[test]
    fn basic_and_graph_produce_same_cursor() {
        let pairs: Vec<_> = vec![
            make_pair(5.0, 0, 1, "d1", "s1"),
            make_pair(4.5, 1, 1, "d2", "s1"),
            make_pair(4.0, 0, 2, "d3", "s1"),
            make_pair(3.0, 0, 3, "d4", "s1"),
            make_pair(2.0, 1, 2, "d5", "s1"),
        ];

        let scored_docs: Vec<ScoredDoc> = pairs.iter().map(|(sd, _)| sd.clone()).collect();
        let mut scored_sentences: Vec<ScoredSentence> =
            pairs.iter().map(|(_, ss)| ss.clone()).collect();
        scored_sentences.sort();

        let driver = PaginationDriver::new(2, None);

        let basic_page = driver.paginate(scored_docs);
        let graph_page = driver.paginate(scored_sentences);

        assert_eq!(basic_page.total_hits, graph_page.total_hits);
        assert_eq!(basic_page.items.len(), graph_page.items.len());
        assert_eq!(basic_page.next_cursor, graph_page.next_cursor);

        for (basic_item, graph_item) in basic_page.items.iter().zip(graph_page.items.iter()) {
            assert_eq!(basic_item.doc_address(), graph_item.doc_address());
            assert!((basic_item.score() - graph_item.score()).abs() < 1e-6);
        }
    }

    #[test]
    fn cursor_from_basic_works_in_graph_and_vice_versa() {
        let pairs: Vec<_> = vec![
            make_pair(5.0, 0, 1, "d1", "s1"),
            make_pair(4.5, 1, 1, "d2", "s1"),
            make_pair(4.0, 0, 2, "d3", "s1"),
            make_pair(3.0, 0, 3, "d4", "s1"),
        ];

        let scored_docs: Vec<ScoredDoc> = pairs.iter().map(|(sd, _)| sd.clone()).collect();
        let mut scored_sentences: Vec<ScoredSentence> =
            pairs.iter().map(|(_, ss)| ss.clone()).collect();
        scored_sentences.sort();

        let driver1 = PaginationDriver::new(2, None);
        let basic_page1 = driver1.paginate(scored_docs);
        let cursor = basic_page1.next_cursor.unwrap();

        let driver2 = PaginationDriver::new(2, Some(cursor.clone()));
        let graph_page2 = driver2.paginate(scored_sentences);

        assert_eq!(graph_page2.items.len(), 2);
        assert!((graph_page2.items[0].score() - 4.0).abs() < 1e-6);
        assert!((graph_page2.items[1].score() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_input_produces_no_cursor() {
        let driver = PaginationDriver::new(10, None);

        let basic_page: crate::engine::pagination_driver::PaginatedPage<ScoredDoc> =
            driver.paginate(vec![]);
        assert_eq!(basic_page.total_hits, 0);
        assert!(basic_page.items.is_empty());
        assert!(basic_page.next_cursor.is_none());

        let graph_page: crate::engine::pagination_driver::PaginatedPage<ScoredSentence> =
            driver.paginate(vec![]);
        assert_eq!(graph_page.total_hits, 0);
        assert!(graph_page.items.is_empty());
        assert!(graph_page.next_cursor.is_none());
    }

    #[test]
    fn page_size_larger_than_results() {
        let pairs: Vec<_> = vec![
            make_pair(5.0, 0, 1, "d1", "s1"),
            make_pair(4.0, 0, 2, "d2", "s1"),
        ];

        let scored_docs: Vec<ScoredDoc> = pairs.iter().map(|(sd, _)| sd.clone()).collect();

        let driver = PaginationDriver::new(10, None);
        let page = driver.paginate(scored_docs);

        assert_eq!(page.total_hits, 2);
        assert_eq!(page.items.len(), 2);
        assert!(page.next_cursor.is_none());
    }

    #[test]
    fn cursor_past_all_items_returns_empty() {
        let pairs: Vec<_> = vec![
            make_pair(5.0, 0, 1, "d1", "s1"),
            make_pair(4.0, 0, 2, "d2", "s1"),
        ];

        let scored_docs: Vec<ScoredDoc> = pairs.iter().map(|(sd, _)| sd.clone()).collect();

        let cursor = SearchCursor::new(1, 999, 1.0);
        let driver = PaginationDriver::new(10, Some(cursor));
        let page = driver.paginate(scored_docs);

        assert_eq!(page.total_hits, 2);
        assert!(page.items.is_empty());
        assert!(page.next_cursor.is_none());
    }
}
