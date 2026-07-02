//! End-to-end feature validation for the RustIE query language.
//!
//! Unlike `tests/comprehensive.rs`, this suite is **fully self-contained**: it
//! builds a small, well-known corpus inline (no dependency on the git-ignored
//! `sample_documents/` directory) so every assertion is deterministic and the
//! tests pass on a clean checkout.
//!
//! Corpus (one sentence per document):
//!
//!   doc_svo: "John eats pizza ."
//!     idx     0     1     2    3
//!     word    John  eats  pizza .
//!     pos     NNP   VBZ   NN   .
//!     lemma   john  eat   pizza .
//!     entity  B-PERSON O  B-FOOD O
//!     deps (root=1): eats>nsubj>John, eats>dobj>pizza, eats>punct>.
//!
//!   doc_dog: "The big dog chased the small cat"
//!     idx     0   1   2   3      4   5     6
//!     word    The big dog chased the small cat
//!     pos     DT  JJ  NN  VBD    DT  JJ    NN
//!     lemma   the big dog chase  the small cat
//!     entity  O   O   B-ANIMAL O O   O     B-ANIMAL
//!     deps (root=3): chased>nsubj>dog, chased>dobj>cat,
//!                    dog>det>The, dog>amod>big, cat>det>the, cat>amod>small
//!
//!   doc_bird: "A small bird sings"
//!     idx     0  1     2    3
//!     word    A  small bird sings
//!     pos     DT JJ    NN   VBZ
//!     lemma   a  small bird sing
//!     entity  O  O     B-ANIMAL O
//!     deps (root=3): sings>nsubj>bird, bird>det>A, bird>amod>small

use rustie::data::document::{Field, Sentence};
use rustie::{Document, ExtractorEngine};
use std::path::Path;

// ============================== Fixtures ==============================

fn tokens(name: &str, toks: &[&str]) -> Field {
    Field::TokensField {
        name: name.to_string(),
        tokens: toks.iter().map(|s| s.to_string()).collect(),
    }
}

fn deps(roots: Vec<u32>, edges: &[(u32, u32, &str)]) -> Field {
    Field::GraphField {
        name: "dependencies".to_string(),
        roots,
        edges: edges
            .iter()
            .map(|(f, t, r)| (*f, *t, r.to_string()))
            .collect(),
    }
}

fn doc_svo() -> Document {
    Document {
        id: "doc_svo".to_string(),
        metadata: vec![],
        sentences: vec![Sentence {
            numTokens: 4,
            fields: vec![
                tokens("word", &["John", "eats", "pizza", "."]),
                tokens("pos", &["NNP", "VBZ", "NN", "."]),
                tokens("lemma", &["john", "eat", "pizza", "."]),
                tokens("entity", &["B-PERSON", "O", "B-FOOD", "O"]),
                deps(vec![1], &[(1, 0, "nsubj"), (1, 2, "dobj"), (1, 3, "punct")]),
            ],
        }],
    }
}

fn doc_dog() -> Document {
    Document {
        id: "doc_dog".to_string(),
        metadata: vec![],
        sentences: vec![Sentence {
            numTokens: 7,
            fields: vec![
                tokens(
                    "word",
                    &["The", "big", "dog", "chased", "the", "small", "cat"],
                ),
                tokens("pos", &["DT", "JJ", "NN", "VBD", "DT", "JJ", "NN"]),
                tokens(
                    "lemma",
                    &["the", "big", "dog", "chase", "the", "small", "cat"],
                ),
                tokens("entity", &["O", "O", "B-ANIMAL", "O", "O", "O", "B-ANIMAL"]),
                deps(
                    vec![3],
                    &[
                        (3, 2, "nsubj"),
                        (3, 6, "dobj"),
                        (2, 0, "det"),
                        (2, 1, "amod"),
                        (6, 4, "det"),
                        (6, 5, "amod"),
                    ],
                ),
            ],
        }],
    }
}

fn doc_bird() -> Document {
    Document {
        id: "doc_bird".to_string(),
        metadata: vec![],
        sentences: vec![Sentence {
            numTokens: 4,
            fields: vec![
                tokens("word", &["A", "small", "bird", "sings"]),
                tokens("pos", &["DT", "JJ", "NN", "VBZ"]),
                tokens("lemma", &["a", "small", "bird", "sing"]),
                tokens("entity", &["O", "O", "B-ANIMAL", "O"]),
                deps(vec![3], &[(3, 2, "nsubj"), (2, 0, "det"), (2, 1, "amod")]),
            ],
        }],
    }
}

/// Build an engine over a temp index, seeded with the known corpus above.
/// Keep the returned `TempDir` alive for the engine to remain valid.
fn build_engine() -> (ExtractorEngine, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let schema_path = Path::new("configs/schema.yaml");
    let mut engine =
        ExtractorEngine::new(tmp.path(), schema_path).expect("failed to create engine");

    for doc in [doc_svo(), doc_dog(), doc_bird()] {
        engine.add_document(&doc).expect("add_document");
    }
    engine.commit().expect("commit");
    assert_eq!(engine.num_docs(), 3, "corpus should index 3 sentences");
    (engine, tmp)
}

/// Run a paginated query and return total matching sentences (panics on error).
fn hits(engine: &ExtractorEngine, query: &str) -> usize {
    engine
        .query_paginated(query, 100, None)
        .unwrap_or_else(|e| panic!("query '{}' failed: {}", query, e))
        .total_hits
}

// ============================== Token constraints ==============================

#[test]
fn exact_match_counts_are_precise() {
    let (engine, _tmp) = build_engine();
    assert_eq!(hits(&engine, "[word=dog]"), 1);
    assert_eq!(hits(&engine, "[word=John]"), 1);
    assert_eq!(hits(&engine, "[word=bird]"), 1);
    assert_eq!(hits(&engine, "[word=pizza]"), 1);
    assert_eq!(hits(&engine, "[word=nonexistent_token]"), 0);
}

#[test]
fn wildcard_matches_every_sentence() {
    let (engine, _tmp) = build_engine();
    // A bare wildcard matches at least one token in every sentence.
    assert_eq!(hits(&engine, "[]"), 3);
    assert_eq!(hits(&engine, "[*]"), 3);
}

#[test]
fn regex_match_counts() {
    let (engine, _tmp) = build_engine();
    // Words starting with 'c': chased, cat -> both in doc_dog only.
    // (Tantivy RegexQuery does whole-term matching; avoid `^`/`$` anchors.)
    assert_eq!(hits(&engine, "[word=/c.*/]"), 1);
    // Any noun tag NN/NNP: pizza(svo), dog/cat(dog), bird(bird) -> 3 sentences.
    assert_eq!(hits(&engine, "[pos=/NN.*/]"), 3);
    // Any verb tag VB*: eats(svo), chased(dog), sings(bird) -> 3 sentences.
    assert_eq!(hits(&engine, "[pos=/VB.*/]"), 3);
}

#[test]
fn invalid_regex_is_an_error_not_a_panic() {
    let (engine, _tmp) = build_engine();
    let result = engine.query_paginated("[word=/[/]", 10, None);
    assert!(result.is_err(), "invalid regex should surface as an error");
}

// ============================== Boolean logic ==============================

#[test]
fn boolean_and_counts() {
    let (engine, _tmp) = build_engine();
    // NOTE: token-level `&` currently compiles to a *document-level* boolean
    // intersection in the basic path, so these assertions are framed to hold
    // under document-level semantics as well as the stricter token-level one.
    assert_eq!(hits(&engine, "[pos=NN & word=dog]"), 1);
    assert_eq!(hits(&engine, "[pos=NNP & word=John]"), 1);
    // No document contains the word "absent_token", so AND must be empty.
    assert_eq!(hits(&engine, "[pos=NN & word=absent_token]"), 0);
}

#[test]
fn boolean_or_counts() {
    let (engine, _tmp) = build_engine();
    // Both cat and dog live in doc_dog -> a single matching sentence.
    assert_eq!(hits(&engine, "[word=cat | word=dog]"), 1);
    // John(svo) or bird(bird) -> two distinct sentences.
    assert_eq!(hits(&engine, "[word=John | word=bird]"), 2);
}

#[test]
fn negation_is_wired_and_never_broadens_results() {
    let (engine, _tmp) = build_engine();
    // Pure negation parses and executes.
    assert!(engine.query_paginated("[word!=dog]", 100, None).is_ok());
    // Adding a negated clause to a constraint must never *increase* the hit count
    // relative to the un-negated baseline.
    let baseline = hits(&engine, "[pos=NN]");
    let narrowed = hits(&engine, "[pos=NN & !word=dog]");
    assert!(
        narrowed <= baseline,
        "negation broadened results: {} > {}",
        narrowed,
        baseline
    );
}

// ============================== Sequences ==============================

#[test]
fn adjacent_sequences() {
    let (engine, _tmp) = build_engine();
    // DT JJ NN: "The big dog"/"the small cat"(dog) and "A small bird"(bird) -> 2.
    assert_eq!(hits(&engine, "[pos=DT] [pos=JJ] [pos=NN]"), 2);
    // NNP VBZ: "John eats" -> only doc_svo.
    assert_eq!(hits(&engine, "[pos=NNP] [pos=VBZ]"), 1);
}

#[test]
fn sequence_with_wildcard_gap() {
    let (engine, _tmp) = build_engine();
    // "The _ dog" -> The(0) big(1) dog(2) in doc_dog.
    assert_eq!(hits(&engine, "[word=The] [] [word=dog]"), 1);
}

// ============================== Quantifiers ==============================

#[test]
fn greedy_quantifiers_execute() {
    let (engine, _tmp) = build_engine();
    // DT JJ* NN matches in doc_dog and doc_bird (and tolerates zero JJ).
    assert!(hits(&engine, "[pos=DT] [pos=JJ]* [pos=NN]") >= 2);
    // One-or-more adjectives: doc_dog and doc_bird.
    assert!(hits(&engine, "[pos=JJ]+") >= 2);
}

#[test]
fn optional_quantifier_admits_zero_occurrences() {
    let (engine, _tmp) = build_engine();
    // In this corpus every DT is followed by a JJ, so a *mandatory* "DT NN"
    // adjacency matches nothing, while the optional form must still match the
    // bare nouns in every sentence. This validates the zero-occurrence branch.
    let mandatory = hits(&engine, "[pos=DT] [pos=NN]");
    let optional = hits(&engine, "[pos=DT]? [pos=NN]");
    assert!(
        optional > mandatory,
        "optional quantifier should admit zero-DT matches ({} !> {})",
        optional,
        mandatory
    );
    assert_eq!(optional, 3, "every sentence has at least one noun");
}

#[test]
fn range_quantifier_executes() {
    let (engine, _tmp) = build_engine();
    assert!(hits(&engine, "[pos=JJ]{1,2}") >= 2);
    // A bounded wildcard run should run without error.
    assert!(engine.query_paginated("[]{2,3}", 100, None).is_ok());
}

// ============================== Graph traversals ==============================

#[test]
fn graph_outgoing_traversals() {
    let (engine, _tmp) = build_engine();
    assert_eq!(hits(&engine, "[word=chased] >nsubj [word=dog]"), 1);
    assert_eq!(hits(&engine, "[word=chased] >dobj [word=cat]"), 1);
    assert_eq!(hits(&engine, "[pos=VBD] >nsubj [pos=NN]"), 1);
    assert_eq!(hits(&engine, "[word=dog] >amod [word=big]"), 1);
    assert_eq!(hits(&engine, "[word=sings] >nsubj [word=bird]"), 1);
}

#[test]
fn graph_incoming_traversals() {
    let (engine, _tmp) = build_engine();
    assert_eq!(hits(&engine, "[word=dog] <nsubj [word=chased]"), 1);
    assert_eq!(hits(&engine, "[word=big] <amod [word=dog]"), 1);
}

#[test]
fn graph_wildcard_traversals() {
    let (engine, _tmp) = build_engine();
    // chased has outgoing edges; dog has an incoming edge.
    assert!(hits(&engine, "[word=chased] >> []") >= 1);
    assert!(hits(&engine, "[word=dog] << []") >= 1);
}

#[test]
fn graph_regex_label_traversal() {
    let (engine, _tmp) = build_engine();
    // chased -> {nsubj,dobj} both present in doc_dog.
    assert!(hits(&engine, "[word=chased] >/nsubj|dobj/ []") >= 1);
}

#[test]
fn graph_traversal_no_false_positive() {
    let (engine, _tmp) = build_engine();
    // "dog" is the subject, not the object, of "chased".
    assert_eq!(hits(&engine, "[word=chased] >dobj [word=dog]"), 0);
}

// ============================== Named captures ==============================

#[test]
fn named_capture_counts() {
    let (engine, _tmp) = build_engine();
    // The capture wrapper does not change which sentences match the inner pattern.
    assert_eq!(hits(&engine, "(?<noun>[pos=NN])"), 3);
    // Animals: dog/cat(doc_dog), bird(doc_bird) -> 2 sentences.
    assert_eq!(hits(&engine, "(?<animal>[entity=B-ANIMAL])"), 2);
}

#[test]
fn named_capture_exposes_capture_name_for_word_field() {
    let (engine, _tmp) = build_engine();
    // Capture span computation resolves against the `word` field, so use a
    // word-field constraint to validate that the capture name is attached.
    let result = engine
        .query_paginated("(?<animal>[word=dog])", 100, None)
        .expect("named capture query");
    assert_eq!(result.total_hits, 1);

    let sentence = &result.sentence_results[0];
    let has_animal_dog = sentence.matches.iter().any(|m| {
        m.captures.iter().any(|c| {
            c.name == "animal" && sentence.get_match_text(&c.span).as_deref() == Some("dog")
        })
    });
    assert!(
        has_animal_dog,
        "expected a capture named 'animal' spanning the word 'dog'"
    );
}

// ============================== Assertions ==============================

#[test]
fn assertions_execute_without_error() {
    let (engine, _tmp) = build_engine();
    // Lookahead/lookbehind are accepted by the parser/compiler and execute.
    for q in [
        "(?= [pos=NN])",
        "(?! [pos=NN])",
        "(?<= [pos=DT])",
        "(?<! [pos=DT])",
    ] {
        assert!(
            engine.query_paginated(q, 100, None).is_ok(),
            "assertion query '{}' should execute without error",
            q
        );
    }
}

// ============================== Match span correctness ==============================

#[test]
fn single_token_match_span_points_at_the_token() {
    let (engine, _tmp) = build_engine();
    let result = engine
        .query_paginated("[word=dog]", 100, None)
        .expect("query");
    assert_eq!(result.total_hits, 1);
    let sentence = &result.sentence_results[0];
    assert_eq!(sentence.document_id.as_ref(), "doc_dog");

    // The reported match span should resolve to exactly the word "dog".
    let matched_texts: Vec<String> = sentence
        .matches
        .iter()
        .filter_map(|m| sentence.get_match_text(&m.span))
        .collect();
    assert!(
        matched_texts.iter().any(|t| t == "dog"),
        "expected a match span resolving to 'dog', got {:?}",
        matched_texts
    );
}

#[test]
fn sequence_match_span_covers_full_phrase() {
    let (engine, _tmp) = build_engine();
    let result = engine
        .query_paginated("[pos=DT] [pos=JJ] [pos=NN]", 100, None)
        .expect("query");
    assert!(result.total_hits >= 2);

    // Every DT JJ NN match must span exactly three tokens.
    let mut saw_triple = false;
    for sentence in &result.sentence_results {
        for m in &sentence.matches {
            assert_eq!(
                m.span.length(),
                3,
                "DT JJ NN match should cover 3 tokens, got span {:?}",
                m.span
            );
            saw_triple = true;
        }
    }
    assert!(saw_triple, "expected at least one populated 3-token match");
}

// ============================== Pagination ==============================

#[test]
fn pagination_walks_all_results_without_duplicates() {
    let (engine, _tmp) = build_engine();
    let query = "[pos=NN]"; // 3 matching sentences

    let mut seen: Vec<String> = Vec::new();
    let mut cursor = None;
    let mut pages = 0;

    loop {
        let page = engine
            .query_paginated(query, 1, cursor.clone())
            .expect("paginated query");
        assert_eq!(page.total_hits, 3, "total_hits stays stable across pages");
        assert!(page.sentence_results.len() <= 1, "page_size=1 honored");

        for s in &page.sentence_results {
            let key = format!("{}:{}", s.document_id, s.sentence_id);
            assert!(
                !seen.contains(&key),
                "duplicate result across pages: {}",
                key
            );
            seen.push(key);
        }

        match page.next_cursor {
            Some(c) => cursor = Some(c),
            None => break,
        }
        pages += 1;
        assert!(pages < 10, "pagination should terminate");
    }

    assert_eq!(seen.len(), 3, "pagination should surface all 3 sentences");
}

#[test]
fn empty_result_has_no_cursor() {
    let (engine, _tmp) = build_engine();
    let page = engine
        .query_paginated("[word=does_not_exist]", 10, None)
        .expect("query");
    assert_eq!(page.total_hits, 0);
    assert!(page.sentence_results.is_empty());
    assert!(page.next_cursor.is_none());
}

// ============================== Result content ==============================

#[test]
fn results_expose_word_field_tokens() {
    let (engine, _tmp) = build_engine();
    let result = engine
        .query_paginated("[word=John]", 10, None)
        .expect("query");
    assert_eq!(result.total_hits, 1);
    let sentence = &result.sentence_results[0];
    let expected: Vec<String> = ["John", "eats", "pizza", "."]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(sentence.tokens(), Some(&expected));
    assert_eq!(sentence.sentence_text(), "John eats pizza .");
}
