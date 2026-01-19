//! Core ExtractorEngine struct and constructor

use crate::engine::constants::*;
use crate::engine::schema::create_schema_from_yaml;
use crate::query::QueryCompiler;
use anyhow::{Result, anyhow};
use std::path::Path;
use tantivy::{
    Index, IndexReader, IndexWriter,
    schema::{Field, Schema},
    directory::MmapDirectory,
};

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
    pub(crate) parent_doc_id_field: String,
    pub(crate) output_fields: Vec<String>,
}

impl ExtractorEngine {
    /// Create a new ExtractorEngine from an index directory with a required schema file
    pub fn new(index_dir: &Path, schema_path: &Path) -> Result<Self> {
        let (schema, output_fields) = create_schema_from_yaml(schema_path)?;
        let index = Self::open_index(index_dir, schema.clone())?;
        
        Self::register_tokenizers(&index);
        
        let reader = index.reader()?;
        let writer = Self::try_create_writer(&index)?;
        let fields = Self::extract_required_fields(&schema)?;

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
            parent_doc_id_field: FIELD_DOC_ID.to_string(),
            output_fields,
        })
    }

    /// Create an ExtractorEngine from a string path
    pub fn from_path(index_dir: &str) -> Result<Self> {
        let schema_path = Path::new("configs/schema.yaml");
        Self::new(Path::new(index_dir), schema_path)
    }

    fn open_index(index_dir: &Path, schema: Schema) -> Result<Index> {
        let dir = MmapDirectory::open(index_dir)?;
        Ok(Index::open_or_create(dir, schema)?)
    }

    fn register_tokenizers(index: &Index) {
        index.tokenizers().register(
            "edge_position_tokenizer",
            crate::tantivy_integration::position_tokenizer::PositionAwareEdgeTokenizer,
        );
        index.tokenizers().register(
            "token_position_tokenizer",
            crate::tantivy_integration::position_tokenizer::PositionAwareTokenTokenizer,
        );
        log::info!("Registered position-aware tokenizers (edge and token)");
    }

    fn try_create_writer(index: &Index) -> Result<Option<IndexWriter>> {
        match index.writer(50_000_000) {
            Ok(w) => Ok(Some(w)),
            Err(tantivy::TantivyError::LockFailure(e, _)) => {
                log::warn!("Could not acquire index lock, running in READ-ONLY mode: {}", e);
                Ok(None)
            }
            Err(e) => Err(anyhow::Error::from(e)),
        }
    }

    fn extract_required_fields(schema: &Schema) -> Result<RequiredFields> {
        Ok(RequiredFields {
            default: schema.get_field(FIELD_WORD)
                .map_err(|_| anyhow!("Field '{}' not found", FIELD_WORD))?,
            sentence_length: schema.get_field(FIELD_SENTENCE_LENGTH)
                .map_err(|_| anyhow!("Field '{}' not found", FIELD_SENTENCE_LENGTH))?,
            dependencies_binary: schema.get_field(FIELD_DEPENDENCIES_BINARY)
                .map_err(|_| anyhow!("Field '{}' not found", FIELD_DEPENDENCIES_BINARY))?,
            incoming_edges: schema.get_field(FIELD_INCOMING_EDGES)
                .map_err(|_| anyhow!("Field '{}' not found", FIELD_INCOMING_EDGES))?,
            outgoing_edges: schema.get_field(FIELD_OUTGOING_EDGES)
                .map_err(|_| anyhow!("Field '{}' not found", FIELD_OUTGOING_EDGES))?,
        })
    }

    // Accessor methods
    pub fn num_docs(&self) -> usize {
        self.reader.searcher().num_docs().try_into().unwrap()
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
        &self.parent_doc_id_field
    }

    pub fn output_fields(&self) -> &[String] {
        &self.output_fields
    }

    pub fn is_output_field(&self, field_name: &str) -> bool {
        self.output_fields.iter().any(|f| f == field_name)
    }

    pub fn compiler(&self) -> QueryCompiler {
        QueryCompiler::new(self.schema.clone())
    }

    /// Create schema from YAML file - wrapper for backward compatibility
    pub fn create_schema_from_yaml<P: AsRef<std::path::Path>>(
        schema_path: P,
    ) -> anyhow::Result<(tantivy::schema::Schema, Vec<String>)> {
        crate::engine::schema::create_schema_from_yaml(schema_path)
    }
}

struct RequiredFields {
    default: Field,
    sentence_length: Field,
    dependencies_binary: Field,
    incoming_edges: Field,
    outgoing_edges: Field,
}