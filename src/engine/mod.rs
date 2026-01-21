//! Engine module for information extraction using Tantivy
//!
//! This module is organized into the following submodules:
//! - `constants`: Field name constants used across the codebase
//! - `config`: Schema configuration types (SchemaConfig, FieldConfig)
//! - `schema`: Schema creation from YAML configuration files
//! - `core`: Core ExtractorEngine struct and constructor
//! - `execution`: Query execution methods (graph traversal, pattern matching)
//! - `document`: Document management methods (add, commit, extract)

pub mod config;
pub mod constants;
pub mod core;
pub mod document;
pub mod execution;
pub mod schema;

// Re-export main types for convenience
pub use config::{FieldConfig, SchemaConfig};
pub use constants::*;
pub use core::ExtractorEngine;
pub use schema::create_schema_from_yaml;
