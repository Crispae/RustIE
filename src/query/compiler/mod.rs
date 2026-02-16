pub mod basic_compiler;
pub mod graph_compiler;

use crate::query::ast::Pattern;
use crate::query::parser::QueryParser;
use crate::tantivy_integration::concat_query::RustieConcatQuery;
use crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery;
use crate::tantivy_integration::named_capture_query::RustieNamedCaptureQuery;
use anyhow::Result;
use tantivy::query::Query;
use tantivy::schema::Schema;

/// Typed result of compiling a `Pattern`.
///
/// Each variant carries the **concrete** query type, eliminating runtime
/// downcasting in the execution layer. The executor matches on this enum
/// directly and passes the typed value to the appropriate execute_* method.
pub enum CompiledQuery {
    Graph(OptimizedGraphTraversalQuery),
    Concat(RustieConcatQuery),
    NamedCapture(RustieNamedCaptureQuery),
    Basic(Box<dyn Query>),
}

impl std::fmt::Debug for CompiledQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Graph(_) => write!(f, "CompiledQuery::Graph(..)"),
            Self::Concat(_) => write!(f, "CompiledQuery::Concat(..)"),
            Self::NamedCapture(_) => write!(f, "CompiledQuery::NamedCapture(..)"),
            Self::Basic(_) => write!(f, "CompiledQuery::Basic(..)"),
        }
    }
}

impl CompiledQuery {
    /// Borrow as a Tantivy trait object when the searcher API requires `&dyn Query`.
    pub fn as_query(&self) -> &dyn Query {
        match self {
            Self::Graph(q) => q,
            Self::Concat(q) => q,
            Self::NamedCapture(q) => q,
            Self::Basic(q) => q.as_ref(),
        }
    }

    /// Consume into `Box<dyn Query>` — used when an inner query needs boxing
    /// (e.g., `RustieNamedCaptureQuery` wrapping an inner compiled query).
    pub fn into_boxed_query(self) -> Box<dyn Query> {
        match self {
            Self::Graph(q) => Box::new(q),
            Self::Concat(q) => Box::new(q),
            Self::NamedCapture(q) => Box::new(q),
            Self::Basic(q) => q,
        }
    }
}

/// Unified compiler that delegates to the appropriate specialized compiler.
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

    /// Compile a pre-parsed `Pattern` into a `CompiledQuery`.
    ///
    /// Callers that already hold a `Pattern` (e.g., from a prior `parse_query` call)
    /// should prefer this method to avoid re-parsing the query string.
    pub fn compile_pattern(&self, pattern: &Pattern) -> Result<CompiledQuery> {
        match pattern {
            Pattern::GraphTraversal { .. } => {
                let gq = self.graph_compiler.compile_graph_traversal(pattern)?;
                Ok(CompiledQuery::Graph(gq))
            }
            _ => self.basic_compiler.compile_pattern(pattern),
        }
    }

    /// Parse and compile a query string in one step.
    ///
    /// Prefer `compile_pattern` when the `Pattern` has already been obtained
    /// from `QueryParser::parse_query`, to avoid parsing the string twice.
    pub fn compile(&self, query: &str) -> Result<CompiledQuery> {
        let parser = QueryParser::new();
        let pattern = parser.parse_query(query)?;
        self.compile_pattern(&pattern)
    }
}

#[cfg(test)]
mod tests;
