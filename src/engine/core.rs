//! Core ExtractorEngine struct and constructor

use crate::engine::constants::*;
use crate::engine::schema::create_schema_from_yaml;
use crate::query::QueryCompiler;
use crate::query::parser::QueryParser;
use crate::tantivy_integration::graph_traversal::{RegexCache, RegexCacheInner};
use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::{
    Index, IndexReader, IndexWriter,
    schema::{Field, Schema},
    directory::MmapDirectory,
};

/// Configuration for ExtractorEngine construction.
/// Centralizes magic numbers and paths for maintainability.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Heap size in bytes for the index writer (default: 50MB)
    pub writer_heap_bytes: usize,
    /// Capacity of the regex cache for graph traversal (default: 512)
    pub regex_cache_capacity: usize,
    /// Path to the schema YAML file
    pub schema_path: std::path::PathBuf,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            writer_heap_bytes: 50_000_000,
            regex_cache_capacity: 512,
            schema_path: std::path::PathBuf::from("configs/schema.yaml"),
        }
    }
}

/// Main engine for information extraction using Tantivy
pub struct ExtractorEngine {
    pub(crate) index: Index,
    pub(crate) reader: IndexReader,
    pub(crate) writer: Option<IndexWriter>,
    pub(crate) schema: Schema,
    pub(crate) default_field: Field,
    pub(crate) sentence_length_field: Field,
    pub(crate) dependencies_binary_field: Field,
    pub(crate) incoming_edges_field: Field,
    pub(crate) outgoing_edges_field: Field,
    pub(crate) output_fields: Vec<String>,
    pub(crate) output_fields_set: HashSet<String>,
    /// Cached query compiler (avoids cloning Schema on every query)
    pub(crate) query_compiler: QueryCompiler,
    /// Cached query parser (avoids re-creating pest parser on every query)
    pub(crate) query_parser: QueryParser,
    /// Bounded regex cache for graph traversal (FST compilation shared across queries)
    pub(crate) regex_cache: RegexCache,
}

impl ExtractorEngine {
    
    /// Create a new ExtractorEngine from an index directory with a required schema file
    pub fn new(index_dir: &Path, schema_path: &Path) -> Result<Self> {
        Self::with_config(index_dir, &EngineConfig {
            schema_path: schema_path.to_path_buf(),
            ..EngineConfig::default()
        })
    }

    /// Create an ExtractorEngine with full configuration
    pub fn with_config(index_dir: &Path, config: &EngineConfig) -> Result<Self> {
        let (schema, output_fields) = create_schema_from_yaml(&config.schema_path)?;
        let output_fields_set: HashSet<String> = output_fields.iter().cloned().collect();
        let index = Self::open_index(index_dir, schema.clone())?;

        Self::register_tokenizers(&index);

        let reader = index.reader()?;
        let writer = Self::try_create_writer(&index, config.writer_heap_bytes)?;
        let fields = Self::extract_required_fields(&schema)?;

        let query_compiler = QueryCompiler::new(schema.clone());
        let query_parser = QueryParser::new();
        let regex_cache: RegexCache = Arc::new(RwLock::new(RegexCacheInner::new(
            config.regex_cache_capacity,
        )));

        Ok(Self {
            index,
            reader,
            writer,
            schema,
            default_field: fields.default,
            sentence_length_field: fields.sentence_length,
            dependencies_binary_field: fields.dependencies_binary,
            incoming_edges_field: fields.incoming_edges,
            outgoing_edges_field: fields.outgoing_edges,
            output_fields,
            output_fields_set,
            query_compiler,
            query_parser,
            regex_cache,
        })
    }

    /// Create an ExtractorEngine from a string path (uses default schema at configs/schema.yaml)
    pub fn from_path(index_dir: &str) -> Result<Self> {
        Self::with_config(Path::new(index_dir), &EngineConfig::default())
    }

    /// Create an ExtractorEngine from index and schema paths (configurable schema location)
    pub fn from_path_with_schema(index_dir: &str, schema_path: &str) -> Result<Self> {
        Self::with_config(
            Path::new(index_dir),
            &EngineConfig {
                schema_path: std::path::PathBuf::from(schema_path),
                ..EngineConfig::default()
            },
        )
    }

    fn open_index(index_dir: &Path, schema: Schema) -> Result<Index> {
        let dir = MmapDirectory::open(index_dir)?;
        Ok(Index::open_or_create(dir, schema)?)
    }

    fn register_tokenizers(index: &Index) {
        index.tokenizers().register(
            TOKENIZER_EDGE_POSITION,
            crate::tantivy_integration::position_tokenizer::PositionAwareEdgeTokenizer,
        );
        index.tokenizers().register(
            TOKENIZER_TOKEN_POSITION,
            crate::tantivy_integration::position_tokenizer::PositionAwareTokenTokenizer,
        );
        log::info!("Registered position-aware tokenizers (edge and token)");
    }

    fn try_create_writer(index: &Index, heap_bytes: usize) -> Result<Option<IndexWriter>> {
        match index.writer(heap_bytes) {
            Ok(w) => Ok(Some(w)),
            Err(tantivy::TantivyError::LockFailure(e, _)) => {
                log::warn!("Could not acquire index lock, running in READ-ONLY mode: {}", e);
                Ok(None)
            }
            Err(e) => Err(anyhow::Error::from(e)),
        }
    }

    fn extract_required_fields(schema: &Schema) -> Result<RequiredFields> {
        use crate::engine::constants::get_required_field;
        Ok(RequiredFields {
            default: get_required_field(schema, FIELD_WORD)?,
            sentence_length: get_required_field(schema, FIELD_SENTENCE_LENGTH)?,
            dependencies_binary: get_required_field(schema, FIELD_DEPENDENCIES_BINARY)?,
            incoming_edges: get_required_field(schema, FIELD_INCOMING_EDGES)?,
            outgoing_edges: get_required_field(schema, FIELD_OUTGOING_EDGES)?,
        })
    }

    // Accessor methods
    pub fn num_docs(&self) -> usize {
        self.reader.searcher().num_docs() as usize
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn default_field(&self) -> Field {
        self.default_field
    }

    pub fn sentence_length_field(&self) -> Field {
        self.sentence_length_field
    }

    pub fn dependencies_binary_field(&self) -> Field {
        self.dependencies_binary_field
    }

    pub fn searcher(&self) -> tantivy::Searcher {
        self.reader.searcher()
    }

    pub fn parent_doc_id_field(&self) -> &str {
        FIELD_DOC_ID
    }

    pub fn output_fields(&self) -> &[String] {
        &self.output_fields
    }

    pub fn is_output_field(&self, field_name: &str) -> bool {
        self.output_fields_set.contains(field_name)
    }

    pub fn compiler(&self) -> &QueryCompiler {
        &self.query_compiler
    }

    pub fn parser(&self) -> &QueryParser {
        &self.query_parser
    }

    /// Create schema from YAML file - wrapper for backward compatibility.
    /// Prefer `crate::engine::schema::create_schema_from_yaml` directly.
    #[deprecated(
        since = "0.1.0",
        note = "Use crate::engine::schema::create_schema_from_yaml directly"
    )]
    pub fn create_schema_from_yaml<P: AsRef<std::path::Path>>(
        schema_path: P,
    ) -> anyhow::Result<(tantivy::schema::Schema, Vec<String>)> {
        crate::engine::schema::create_schema_from_yaml(schema_path)
    }
}

impl Drop for ExtractorEngine {
    fn drop(&mut self) {
        if let Some(ref mut writer) = self.writer {
            if let Err(e) = writer.commit() {
                log::error!("Failed to commit writer on drop: {}", e);
            }
        }
    }
}

struct RequiredFields {
    default: Field,
    sentence_length: Field,
    dependencies_binary: Field,
    incoming_edges: Field,
    outgoing_edges: Field,
}