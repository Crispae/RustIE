use tantivy::{
    query::{Query, BooleanQuery, Occur, TermQuery, RegexQuery},
    schema::{Term, Field, Schema},
};
use crate::compiler::ast::{Pattern, Constraint, Matcher, Traversal};
use anyhow::{Result, anyhow};
use crate::tantivy_integration::graph_traversal::OptimizedGraphTraversalQuery;
use crate::compiler::compiler::basic_compiler::BasicCompiler;

/// Compiler for graph traversal patterns
pub struct GraphCompiler {
    basic_compiler: BasicCompiler,
    schema: Schema,
}

impl GraphCompiler {
    pub fn new(schema: Schema) -> Self {
        Self {
            basic_compiler: BasicCompiler::new(schema.clone()),
            schema,
        }
    }

    pub fn compile_pattern(&self, pattern: &Pattern) -> Result<Box<dyn Query>> {
        match pattern {
            Pattern::GraphTraversal { src, traversal, dst } => {
                self.compile_graph_traversal(src, traversal, dst)
            }
            _ => {
                // For non-traversal patterns, delegate to basic compiler
                self.basic_compiler.compile_pattern(pattern)
            }
        }
    }

    fn compile_graph_traversal(&self, src: &Pattern, traversal: &Traversal, dst: &Pattern) -> Result<Box<dyn Query>> {
        let src_query = self.compile_pattern(src)?;
        let dst_query = self.compile_pattern(dst)?;
    
        // Get the dependencies fields from schema.
        let dependencies_binary_field = self.schema.get_field("dependencies_binary")
            .map_err(|_| anyhow!("Dependencies binary field not found in schema"))?;
        let default_field = self.schema.get_field("word")
            .map_err(|_| anyhow!("Default field 'word' not found in schema"))?;
        
        
        
        Ok(Box::new(OptimizedGraphTraversalQuery::new(
            default_field,
            dependencies_binary_field,
            src_query,
            traversal.clone(),
            dst_query,
            src.clone(),
            dst.clone(),
        )))
    }
} 