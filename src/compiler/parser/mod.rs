
use crate::compiler::ast::Pattern;
use anyhow::Result;
// Integrate pest-based parser
use crate::compiler::pest_parser::{QueryParser as PestQueryParser, Rule, build_ast};
use pest::Parser;

/// Unified parser that delegates to appropriate specialized parser
pub struct QueryParser {
}

impl QueryParser {
    pub fn new(default_field: String) -> Self {
        Self {
            
        }
    }

    pub fn parse_query(&self, query: &str) -> Result<Pattern> {
        // Use pest-based parser for all queries
        let mut pairs = PestQueryParser::parse(Rule::query, query)?;
        let ast = build_ast(pairs.next().unwrap());
        Ok(ast)
    }
} 