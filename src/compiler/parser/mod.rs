use crate::compiler::ast::Pattern;
use anyhow::{Result, anyhow};
// Integrate pest-based parser
use crate::compiler::pest_parser::{QueryParser as PestQueryParser, Rule, build_ast};
use pest::Parser;
use std::panic;

/// Unified parser that delegates to appropriate specialized parser
pub struct QueryParser {
}

impl QueryParser {
    pub fn new(_default_field: String) -> Self {
        Self {
        }
    }

    pub fn parse_query(&self, query: &str) -> Result<Pattern> {
        // Use pest-based parser for all queries
        let mut pairs = PestQueryParser::parse(Rule::query, query)?;

        // Fixed: Handle empty parse result instead of unwrapping
        let first_pair = pairs.next()
            .ok_or_else(|| anyhow!("Parse error: Empty parse result for query '{}'", query))?;

        // Catch panics from build_ast and convert to Result
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            build_ast(first_pair)
        }));

        match result {
            Ok(ast) => Ok(ast),
            Err(panic_info) => {
                // Convert panic to error
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown parse error".to_string()
                };
                Err(anyhow!("Parse error in query '{}': {}", query, msg))
            }
        }
    }
} 