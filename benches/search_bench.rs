// benches/search_bench.rs
//
// Criterion benchmarks comparing the old query path (query_with_limit)
// vs the new paginated path (query_paginated) for simple term queries.
//
// Run with:
//   cargo bench --bench search_bench
//
// Requires a test index. Run from project root:
//   cargo run --bin create_test_index
// Then either use default "test_index" or set BENCH_INDEX_PATH.
//
// Results are saved to target/criterion/ with HTML reports.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;

use rustie::engine::core::ExtractorEngine;

/// Build or open the test engine from an index directory.
/// Uses BENCH_INDEX_PATH env var or defaults to "test_index".
fn setup_engine() -> Arc<ExtractorEngine> {
    let index_path = std::env::var("BENCH_INDEX_PATH").unwrap_or_else(|_| "test_index".to_string());
    let engine = ExtractorEngine::from_path(&index_path)
        .expect("Failed to open benchmark index. Run `cargo run --bin create_test_index` or set BENCH_INDEX_PATH.");
    Arc::new(engine)
}

/// Simple entity query that should match documents in the test index.
const SIMPLE_QUERY: &str = "([entity=/B-Gene/])";

fn bench_old_vs_paginated(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("simple_entity_query");
    group.sample_size(30);

    let _ = engine.query_with_limit(SIMPLE_QUERY, 10);

    for limit in [10, 100, 1000, 6000] {
        let engine = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("query_with_limit", limit),
            &limit,
            |b, &limit| {
                b.iter(|| black_box(engine.query_with_limit(SIMPLE_QUERY, limit).unwrap()))
            },
        );
    }

    for page_size in [15, 50, 100] {
        let engine = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("query_paginated_page1", page_size),
            &page_size,
            |b, &page_size| {
                b.iter(|| {
                    black_box(
                        engine
                            .query_paginated(SIMPLE_QUERY, page_size, None)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_paginated_page2(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("paginated_page2");
    group.sample_size(30);

    let page1 = engine
        .query_paginated(SIMPLE_QUERY, 15, None)
        .expect("Page 1 should succeed");

    let cursor = page1
        .next_cursor
        .expect("Page 1 should have a next_cursor for this query");

    group.bench_function("page2_with_cursor", |b| {
        let engine = engine.clone();
        let cursor = cursor.clone();
        b.iter(|| {
            black_box(
                engine
                    .query_paginated(SIMPLE_QUERY, 15, Some(cursor.clone()))
                    .unwrap(),
            )
        })
    });

    group.bench_function("page1_no_cursor", |b| {
        let engine = engine.clone();
        b.iter(|| {
            black_box(
                engine
                    .query_paginated(SIMPLE_QUERY, 15, None)
                    .unwrap(),
            )
        })
    });

    group.finish();
}

fn bench_parse_and_compile(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("parse_compile");
    group.sample_size(100);

    group.bench_function("parse_only", |b| {
        let engine = engine.clone();
        b.iter(|| black_box(engine.parser().parse_query(SIMPLE_QUERY).unwrap()))
    });

    group.bench_function("compile_only", |b| {
        let engine = engine.clone();
        b.iter(|| black_box(engine.compiler().compile(SIMPLE_QUERY).unwrap()))
    });

    group.finish();
}

fn bench_references_field(c: &mut Criterion) {
    use rustie::query::ast::{Constraint, Matcher, Pattern, QuantifierKind};

    let mut group = c.benchmark_group("references_field");

    let simple = Pattern::Constraint(Constraint::Field {
        name: "entity".to_string(),
        matcher: Matcher::string("B-Gene".to_string()),
    });

    group.bench_function("simple_constraint", |b| {
        b.iter(|| black_box(simple.references_field("word")))
    });

    let deep = Pattern::NamedCapture {
        name: "deep".to_string(),
        pattern: Box::new(Pattern::Repetition {
            pattern: Box::new(Pattern::Disjunctive(vec![
                Pattern::Constraint(Constraint::Field {
                    name: "entity".to_string(),
                    matcher: Matcher::string("B-Gene".to_string()),
                }),
                Pattern::Constraint(Constraint::Field {
                    name: "word".to_string(),
                    matcher: Matcher::string("test".to_string()),
                }),
            ])),
            min: 0,
            max: None,
            kind: QuantifierKind::Lazy,
        }),
    };

    group.bench_function("deep_nested", |b| {
        b.iter(|| black_box(deep.references_field("word")))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_old_vs_paginated,
    bench_paginated_page2,
    bench_parse_and_compile,
    bench_references_field,
);
criterion_main!(benches);
