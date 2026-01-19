//!
//! This module re-exports types from the modular engine structure.
//! The actual implementation is split across:
//! - `core.rs`: ExtractorEngine struct and constructor
//! - `execution.rs`: Query execution methods
//! - `document.rs`: Document management methods
//! - `schema.rs`: Schema creation from YAML
//! - `config.rs`: Configuration types
//! - `constants.rs`: Field name constants

// Re-export all public types for backward compatibility
pub use crate::engine::config::{FieldConfig, SchemaConfig};
pub use crate::engine::constants::*;
pub use crate::engine::core::ExtractorEngine;
pub use crate::engine::schema::create_schema_from_yaml;


//TODO: Remove this module and use the engine module instead