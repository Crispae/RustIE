pub mod engine;
pub mod query;
pub mod tantivy_integration;
pub mod digraph;
pub mod results;
pub mod types;
pub mod data;
pub mod api;

pub use engine::ExtractorEngine;
pub use query::{QueryCompiler, QueryParser};
pub use types::{Span, SpanWithCaptures, NamedCapture};
pub use results::{RustIeResult, RustieDoc, SentenceResult};
pub use data::{Document, DocumentParser};
pub use api::{start_server};
pub use api::server::ApiConfig;
