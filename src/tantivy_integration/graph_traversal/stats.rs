//! Performance statistics for graph traversal queries.

use std::sync::atomic::Ordering;

use super::types::{
    CALL_COUNT, GRAPH_DESER_COUNT, GRAPH_DESER_SKIPPED,
    PREFILTER_DOCS, PREFILTER_KILLED, PREFILTER_ALLOWED_POS_SUM, PREFILTER_ALLOWED_POS_COUNT,
    SRC_DRIVER_DOCS, DST_DRIVER_DOCS, DRIVER_ALIGNMENT_DOCS,
    DRIVER_INTERSECTION_SUM, DRIVER_INTERSECTION_COUNT,
    PREFILTER_SKIPPED_ALL_COLLAPSED, TOKEN_EXTRACTION_SKIPPED,
    REGEX_EXPANSION_COUNT, REGEX_EXPANSION_TERMS,
};

/// Public function to retrieve profiling statistics
/// Returns a struct with all performance metrics
#[derive(Debug, Clone)]
pub struct GraphTraversalStats {
    pub call_count: usize,
    pub graph_deser_count: usize,
    pub graph_deser_skipped: usize,
    pub prefilter_docs: usize,
    pub prefilter_killed: usize,
    pub prefilter_allowed_pos_sum: usize,
    pub prefilter_allowed_pos_count: usize,
    pub src_driver_docs: usize,
    pub dst_driver_docs: usize,
    pub driver_alignment_docs: usize,
    pub driver_intersection_sum: usize,
    pub driver_intersection_count: usize,
    pub prefilter_skipped_all_collapsed: usize,
    pub token_extraction_skipped: usize,
    pub regex_expansion_count: usize,
    pub regex_expansion_terms: usize,
}

impl GraphTraversalStats {
    pub fn get() -> Self {
        Self {
            call_count: CALL_COUNT.load(Ordering::Relaxed),
            graph_deser_count: GRAPH_DESER_COUNT.load(Ordering::Relaxed),
            graph_deser_skipped: GRAPH_DESER_SKIPPED.load(Ordering::Relaxed),
            prefilter_docs: PREFILTER_DOCS.load(Ordering::Relaxed),
            prefilter_killed: PREFILTER_KILLED.load(Ordering::Relaxed),
            prefilter_allowed_pos_sum: PREFILTER_ALLOWED_POS_SUM.load(Ordering::Relaxed),
            prefilter_allowed_pos_count: PREFILTER_ALLOWED_POS_COUNT.load(Ordering::Relaxed),
            src_driver_docs: SRC_DRIVER_DOCS.load(Ordering::Relaxed),
            dst_driver_docs: DST_DRIVER_DOCS.load(Ordering::Relaxed),
            driver_alignment_docs: DRIVER_ALIGNMENT_DOCS.load(Ordering::Relaxed),
            driver_intersection_sum: DRIVER_INTERSECTION_SUM.load(Ordering::Relaxed),
            driver_intersection_count: DRIVER_INTERSECTION_COUNT.load(Ordering::Relaxed),
            prefilter_skipped_all_collapsed: PREFILTER_SKIPPED_ALL_COLLAPSED.load(Ordering::Relaxed),
            token_extraction_skipped: TOKEN_EXTRACTION_SKIPPED.load(Ordering::Relaxed),
            regex_expansion_count: REGEX_EXPANSION_COUNT.load(Ordering::Relaxed),
            regex_expansion_terms: REGEX_EXPANSION_TERMS.load(Ordering::Relaxed),
        }
    }

    pub fn reset() {
        CALL_COUNT.store(0, Ordering::Relaxed);
        GRAPH_DESER_COUNT.store(0, Ordering::Relaxed);
        GRAPH_DESER_SKIPPED.store(0, Ordering::Relaxed);
        PREFILTER_DOCS.store(0, Ordering::Relaxed);
        PREFILTER_KILLED.store(0, Ordering::Relaxed);
        PREFILTER_ALLOWED_POS_SUM.store(0, Ordering::Relaxed);
        PREFILTER_ALLOWED_POS_COUNT.store(0, Ordering::Relaxed);
        SRC_DRIVER_DOCS.store(0, Ordering::Relaxed);
        DST_DRIVER_DOCS.store(0, Ordering::Relaxed);
        DRIVER_ALIGNMENT_DOCS.store(0, Ordering::Relaxed);
        DRIVER_INTERSECTION_SUM.store(0, Ordering::Relaxed);
        DRIVER_INTERSECTION_COUNT.store(0, Ordering::Relaxed);
        PREFILTER_SKIPPED_ALL_COLLAPSED.store(0, Ordering::Relaxed);
        TOKEN_EXTRACTION_SKIPPED.store(0, Ordering::Relaxed);
        REGEX_EXPANSION_COUNT.store(0, Ordering::Relaxed);
        REGEX_EXPANSION_TERMS.store(0, Ordering::Relaxed);
    }

    pub fn print_summary(&self) {
        // Printing disabled for performance
        return;
        println!("\n=== Graph Traversal Performance Statistics ===");

        let skip_rate = if self.call_count > 0 {
            (self.graph_deser_skipped as f64 / self.call_count as f64) * 100.0
        } else {
            0.0
        };

        let prefilter_kill_rate = if self.prefilter_docs > 0 {
            (self.prefilter_killed as f64 / self.prefilter_docs as f64) * 100.0
        } else {
            0.0
        };

        let avg_allowed_pos = if self.prefilter_allowed_pos_count > 0 {
            self.prefilter_allowed_pos_sum as f64 / self.prefilter_allowed_pos_count as f64
        } else {
            0.0
        };

        let avg_intersection = if self.driver_intersection_count > 0 {
            self.driver_intersection_sum as f64 / self.driver_intersection_count as f64
        } else {
            0.0
        };

        println!("Document Processing:");
        println!("  Candidates checked: {}", self.call_count);
        println!("  Graphs deserialized: {} ({} skipped, {:.1}% skip rate)",
                 self.graph_deser_count, self.graph_deser_skipped, skip_rate);

        println!("\nPrefilter Performance:");
        println!("  Documents checked: {}", self.prefilter_docs);
        println!("  Killed by prefilter: {} ({:.1}% kill rate)",
                 self.prefilter_killed, prefilter_kill_rate);
        println!("  Avg allowed positions per constraint: {:.1}", avg_allowed_pos);
        println!("  Prefilter skipped (all-collapsed): {}", self.prefilter_skipped_all_collapsed);

        println!("\nOdinson Driver Performance:");
        println!("  Source driver docs: {}", self.src_driver_docs);
        println!("  Destination driver docs: {}", self.dst_driver_docs);
        println!("  Aligned docs: {}", self.driver_alignment_docs);
        println!("  Avg intersection size: {:.1}", avg_intersection);

        println!("\nOptimizations:");
        println!("  Token extractions skipped: {}", self.token_extraction_skipped);

        println!("\nRegex Expansion:");
        println!("  Regex patterns expanded: {}", self.regex_expansion_count);
        println!("  Total terms from expansion: {}", self.regex_expansion_terms);
        if self.regex_expansion_count > 0 {
            let avg_terms = self.regex_expansion_terms as f64 / self.regex_expansion_count as f64;
            println!("  Average terms per pattern: {:.1}", avg_terms);
        }

        // Calculate efficiency metrics
        if self.call_count > 0 {
            let deser_rate = (self.graph_deser_count as f64 / self.call_count as f64) * 100.0;
            println!("\nEfficiency Metrics:");
            println!("  Graph deserialization rate: {:.1}% (lower is better)", deser_rate);
            println!("  Prefilter effectiveness: {:.1}% documents filtered out", prefilter_kill_rate);
        }
    }
}
