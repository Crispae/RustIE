pub mod basic_compiler;
pub mod graph_compiler;

use crate::query::ast::Pattern;
use anyhow::Result;
use tantivy::query::Query;
use tantivy::schema::Schema;
use crate::query::parser::QueryParser;

/// Unified compiler that delegates to appropriate specialized compiler
pub struct QueryCompiler {
    basic_compiler: basic_compiler::BasicCompiler,
    graph_compiler: graph_compiler::GraphCompiler,
}

impl QueryCompiler {
    pub fn new(schema: Schema) -> Self {
        Self {
            basic_compiler: basic_compiler::BasicCompiler::new(schema.clone()),
            graph_compiler: graph_compiler::GraphCompiler::new(schema),
        }
    }

    pub fn compile(&self, query: &str) -> Result<Box<dyn Query>> {
        // Parse the query first to determine the type
        
        let parser = QueryParser::new("word".to_string());
        let pattern = parser.parse_query(query)?;
        self.compile_pattern(&pattern)
    }

    pub fn compile_pattern(&self, pattern: &Pattern) -> Result<Box<dyn Query>> {
        match pattern {
            
            Pattern::GraphTraversal { .. } => {
                // Use graph compiler for traversal patterns
                self.graph_compiler.compile_pattern(pattern)
            }
            _ => {
                // Use basic compiler for all other patterns
                self.basic_compiler.compile_pattern(pattern)
            }
        }
    }
} 