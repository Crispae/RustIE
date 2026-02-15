//! Tests for query compilers (BasicCompiler, GraphCompiler, QueryCompiler).

#[cfg(test)]
mod tests {
    use crate::query::compiler::{basic_compiler::BasicCompiler, graph_compiler::GraphCompiler, QueryCompiler};
    use crate::query::ast::{
        Assertion, Constraint, Matcher, Pattern, QuantifierKind, Traversal,
    };
    use crate::tantivy_integration::{
        OptimizedGraphTraversalQuery, LookaheadQuery, RustieConcatQuery,
        RustieNamedCaptureQuery, RustieOrQuery,
    };
    use tantivy::query::{BooleanQuery, TermQuery};
    use tantivy::schema::{Schema, SchemaBuilder, TEXT, STORED};

    // ==================== Schema helpers ====================

    fn schema_basic() -> Schema {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("word", TEXT | STORED);
        builder.add_text_field("lemma", TEXT | STORED);
        builder.build()
    }

    fn schema_graph() -> Schema {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("doc_id", TEXT | STORED);
        builder.add_text_field("sentence_id", TEXT | STORED);
        builder.add_u64_field("sentence_length", STORED);
        builder.add_text_field("word", TEXT | STORED);
        builder.add_text_field("lemma", TEXT | STORED);
        builder.add_bytes_field("dependencies_binary", STORED);
        builder.add_text_field("incoming_edges", TEXT | STORED);
        builder.add_text_field("outgoing_edges", TEXT | STORED);
        builder.build()
    }

    fn schema_without_word() -> Schema {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("lemma", TEXT | STORED);
        builder.build()
    }

    fn schema_graph_missing_incoming_edges() -> Schema {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("word", TEXT | STORED);
        builder.add_bytes_field("dependencies_binary", STORED);
        builder.add_text_field("outgoing_edges", TEXT | STORED);
        // missing incoming_edges
        builder.build()
    }

    // ==================== BasicCompiler tests ====================

    #[test]
    fn test_basic_compiler_constraint_wildcard() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Wildcard should compile: {:?}", result.err());
    }

    #[test]
    fn test_basic_compiler_constraint_field_string() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("test".to_string()),
        });
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Field+String should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<TermQuery>().is_some(),
            "Expected TermQuery for exact field match"
        );
    }

    #[test]
    fn test_basic_compiler_constraint_field_regex() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::regex("test.*".to_string()),
        });
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Field+Regex should compile: {:?}", result.err());
    }

    #[test]
    fn test_basic_compiler_constraint_negated() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let inner = Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("x".to_string()),
        };
        let pattern = Pattern::Constraint(Constraint::Negated(Box::new(inner)));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Negated should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<BooleanQuery>().is_some(),
            "Expected BooleanQuery for negated constraint"
        );
    }

    #[test]
    fn test_basic_compiler_constraint_conjunctive() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Conjunctive(vec![
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("a".to_string()),
            },
            Constraint::Field {
                name: "lemma".to_string(),
                matcher: Matcher::String("b".to_string()),
            },
        ]));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Conjunctive should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<BooleanQuery>().is_some(),
            "Expected BooleanQuery for conjunctive constraint"
        );
    }

    #[test]
    fn test_basic_compiler_constraint_disjunctive() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Disjunctive(vec![
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("a".to_string()),
            },
            Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("b".to_string()),
            },
        ]));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Disjunctive constraint should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<BooleanQuery>().is_some(),
            "Expected BooleanQuery for disjunctive constraint"
        );
    }

    #[test]
    fn test_basic_compiler_assertion_positive_lookahead() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let inner = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("x".to_string()),
        });
        let pattern = Pattern::Assertion(Assertion::PositiveLookahead(Box::new(inner)));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "PositiveLookahead should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<LookaheadQuery>().is_some(),
            "Expected LookaheadQuery"
        );
    }

    #[test]
    fn test_basic_compiler_assertion_negative_lookahead() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let inner = Pattern::Constraint(Constraint::Wildcard);
        let pattern = Pattern::Assertion(Assertion::NegativeLookahead(Box::new(inner)));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "NegativeLookahead should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<LookaheadQuery>().is_some(),
            "Expected LookaheadQuery"
        );
    }

    #[test]
    fn test_basic_compiler_assertion_positive_lookbehind() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let inner = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("y".to_string()),
        });
        let pattern = Pattern::Assertion(Assertion::PositiveLookbehind(Box::new(inner)));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "PositiveLookbehind should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<LookaheadQuery>().is_some(),
            "Expected LookaheadQuery"
        );
    }

    #[test]
    fn test_basic_compiler_assertion_negative_lookbehind() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let inner = Pattern::Constraint(Constraint::Wildcard);
        let pattern = Pattern::Assertion(Assertion::NegativeLookbehind(Box::new(inner)));
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "NegativeLookbehind should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<LookaheadQuery>().is_some(),
            "Expected LookaheadQuery"
        );
    }

    #[test]
    fn test_basic_compiler_disjunctive_pattern() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Disjunctive(vec![
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("a".to_string()),
            }),
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("b".to_string()),
            }),
        ]);
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Disjunctive pattern should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<RustieOrQuery>().is_some(),
            "Expected RustieOrQuery"
        );
    }

    #[test]
    fn test_basic_compiler_concatenated_pattern() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Concatenated(vec![
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("a".to_string()),
            }),
            Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("b".to_string()),
            }),
        ]);
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Concatenated pattern should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<RustieConcatQuery>().is_some(),
            "Expected RustieConcatQuery"
        );
    }

    #[test]
    fn test_basic_compiler_named_capture() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let inner = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("John".to_string()),
        });
        let pattern = Pattern::NamedCapture {
            name: "subject".to_string(),
            pattern: Box::new(inner),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "NamedCapture should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<RustieNamedCaptureQuery>().is_some(),
            "Expected RustieNamedCaptureQuery"
        );
    }

    #[test]
    fn test_basic_compiler_repetition() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Repetition {
            pattern: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("a".to_string()),
            })),
            min: 1,
            max: Some(2),
            kind: QuantifierKind::Greedy,
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "Repetition should compile: {:?}", result.err());
        // Can be RustieOrQuery (multiple alternatives) or single query when only one length
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<RustieOrQuery>().is_some()
                || query.as_any().downcast_ref::<TermQuery>().is_some(),
            "Expected RustieOrQuery or TermQuery for repetition"
        );
    }

    #[test]
    fn test_basic_compiler_graph_traversal_returns_err() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("eat".to_string()),
            })),
            traversal: Traversal::Outgoing(Matcher::String("nsubj".to_string())),
            dst: Box::new(Pattern::Constraint(Constraint::Wildcard)),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_err(), "BasicCompiler should reject GraphTraversal");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GraphCompiler") || err_msg.contains("graph traversal"),
            "Error should mention GraphCompiler: {}",
            err_msg
        );
    }

    #[test]
    fn test_basic_compiler_mention_returns_err() {
        let schema = schema_basic();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Mention {
            arg_name: None,
            label: "Subject".to_string(),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_err(), "BasicCompiler should reject Mention");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not yet implemented") || err_msg.contains("Mention"),
            "Error should mention not implemented: {}",
            err_msg
        );
    }

    #[test]
    fn test_basic_compiler_missing_word_field_returns_err() {
        let schema = schema_without_word();
        let compiler = BasicCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_err(), "Compiling without 'word' field should fail: {:?}", result.ok());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("word") || err_msg.contains("not found"),
            "Error should mention missing field: {}",
            err_msg
        );
    }

    // ==================== GraphCompiler tests ====================

    #[test]
    fn test_graph_compiler_non_graph_traversal_returns_err() {
        let schema = schema_graph();
        let compiler = GraphCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Wildcard);
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_err(), "GraphCompiler should reject non-GraphTraversal");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("GraphTraversal") || err_msg.contains("only handles"),
            "Error should mention GraphTraversal: {}",
            err_msg
        );
    }

    #[test]
    fn test_graph_compiler_graph_traversal_simple() {
        let schema = schema_graph();
        let compiler = GraphCompiler::new(schema);
        let pattern = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("eat".to_string()),
            })),
            traversal: Traversal::Outgoing(Matcher::String("nsubj".to_string())),
            dst: Box::new(Pattern::Constraint(Constraint::Wildcard)),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "GraphTraversal should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<OptimizedGraphTraversalQuery>().is_some(),
            "Expected OptimizedGraphTraversalQuery"
        );
    }

    #[test]
    fn test_graph_compiler_graph_traversal_with_full_schema() {
        let schema = schema_graph();
        let compiler = GraphCompiler::new(schema);
        let pattern = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("eats".to_string()),
            })),
            traversal: Traversal::Incoming(Matcher::String("nsubj".to_string())),
            dst: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("John".to_string()),
            })),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "GraphTraversal with first/last constraints should compile: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<OptimizedGraphTraversalQuery>().is_some(),
            "Expected OptimizedGraphTraversalQuery"
        );
    }

    #[test]
    fn test_graph_compiler_schema_missing_required_field_returns_err() {
        let schema = schema_graph_missing_incoming_edges();
        let compiler = GraphCompiler::new(schema);
        let pattern = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("x".to_string()),
            })),
            traversal: Traversal::Outgoing(Matcher::String("nsubj".to_string())),
            dst: Box::new(Pattern::Constraint(Constraint::Wildcard)),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_err(), "Missing incoming_edges should fail: {:?}", result.ok());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("incoming_edges") || err_msg.contains("not found"),
            "Error should mention missing field: {}",
            err_msg
        );
    }

    // ==================== QueryCompiler (unified) tests ====================

    #[test]
    fn test_query_compiler_compile_basic() {
        let schema = schema_basic();
        let compiler = QueryCompiler::new(schema);
        let result = compiler.compile("[word=test]");
        assert!(result.is_ok(), "compile([word=test]) should succeed: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<TermQuery>().is_some(),
            "Basic pattern should produce TermQuery (or similar), not graph query"
        );
    }

    #[test]
    fn test_query_compiler_compile_graph() {
        let schema = schema_graph();
        let compiler = QueryCompiler::new(schema);
        let result = compiler.compile("[word=eats] >nsubj [word=John]");
        assert!(result.is_ok(), "compile(graph query) should succeed: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<OptimizedGraphTraversalQuery>().is_some(),
            "Graph query should produce OptimizedGraphTraversalQuery"
        );
    }

    #[test]
    fn test_query_compiler_compile_pattern_delegation_to_basic() {
        let schema = schema_basic();
        let compiler = QueryCompiler::new(schema);
        let pattern = Pattern::Constraint(Constraint::Field {
            name: "word".to_string(),
            matcher: Matcher::String("foo".to_string()),
        });
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "compile_pattern(Constraint) should succeed: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<TermQuery>().is_some(),
            "Constraint should be compiled by BasicCompiler (TermQuery)"
        );
    }

    #[test]
    fn test_query_compiler_compile_pattern_delegation_to_graph() {
        let schema = schema_graph();
        let compiler = QueryCompiler::new(schema);
        let pattern = Pattern::GraphTraversal {
            src: Box::new(Pattern::Constraint(Constraint::Field {
                name: "word".to_string(),
                matcher: Matcher::String("eat".to_string()),
            })),
            traversal: Traversal::Outgoing(Matcher::String("dobj".to_string())),
            dst: Box::new(Pattern::Constraint(Constraint::Wildcard)),
        };
        let result = compiler.compile_pattern(&pattern);
        assert!(result.is_ok(), "compile_pattern(GraphTraversal) should succeed: {:?}", result.err());
        let query = result.unwrap();
        assert!(
            query.as_any().downcast_ref::<OptimizedGraphTraversalQuery>().is_some(),
            "GraphTraversal should be compiled by GraphCompiler"
        );
    }
}
