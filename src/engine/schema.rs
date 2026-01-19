//! Schema creation from YAML configuration

use crate::engine::config::{SchemaConfig, FieldConfig};
use crate::engine::constants::*;
use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;
use tantivy::schema::{
    Schema, TextFieldIndexing, TextOptions, IndexRecordOption,
    STORED, FAST,
};

/// Create schema from YAML file
pub fn create_schema_from_yaml<P: AsRef<Path>>(
    schema_path: P,
) -> Result<(Schema, Vec<String>)> {
    let path = schema_path.as_ref();
    
    if !path.exists() {
        return Err(anyhow!("Schema file not found: {}", path.display()));
    }

    let yaml_str = fs::read_to_string(path)
        .map_err(|e| anyhow!("Failed to read schema file {}: {}", path.display(), e))?;

    let config: SchemaConfig = serde_yaml::from_str(&yaml_str)
        .map_err(|e| anyhow!("Invalid YAML schema in {}: {}", path.display(), e))?;

    let schema = build_schema(&config)?;
    let output_fields = get_output_fields(&config);

    log::info!("Schema created: all text/string fields indexed with positions (Odinson-style)");
    
    Ok((schema, output_fields))
}

fn build_schema(config: &SchemaConfig) -> Result<Schema> {
    let mut builder = Schema::builder();
    
    for field in &config.fields {
        add_field_to_schema(&mut builder, field)?;
    }
    
    Ok(builder.build())
}

fn add_field_to_schema(
    builder: &mut tantivy::schema::SchemaBuilder,
    field: &FieldConfig,
) -> Result<()> {
    match field.field_type.as_str() {
        "text" => add_text_field(builder, field),
        "string" => add_string_field(builder, field),
        "edge_positions" => add_edge_positions_field(builder, field),
        "u64" => add_u64_field(builder, field),
        "bytes" => add_bytes_field(builder, field),
        _ => return Err(anyhow!("Unknown field type in schema: {}", field.field_type)),
    }
    Ok(())
}

fn add_text_field(builder: &mut tantivy::schema::SchemaBuilder, field: &FieldConfig) {
    let indexing = TextFieldIndexing::default()
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let mut options = TextOptions::default().set_indexing_options(indexing);
    if field.stored {
        options = options.set_stored();
    }
    builder.add_text_field(&field.name, options);
    log::debug!("Added text field '{}' with positions", field.name);
}

fn add_string_field(builder: &mut tantivy::schema::SchemaBuilder, field: &FieldConfig) {
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("token_position_tokenizer")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let mut options = TextOptions::default().set_indexing_options(indexing);
    if field.stored {
        options = options.set_stored();
    }
    builder.add_text_field(&field.name, options);
    log::debug!("Added string field '{}' with position-aware tokenizer", field.name);
}

fn add_edge_positions_field(builder: &mut tantivy::schema::SchemaBuilder, field: &FieldConfig) {
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("edge_position_tokenizer")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let mut options = TextOptions::default().set_indexing_options(indexing);
    if field.stored {
        options = options.set_stored();
    }
    builder.add_text_field(&field.name, options);
    log::info!("Added position-aware edge field: {}", field.name);
}

fn add_u64_field(builder: &mut tantivy::schema::SchemaBuilder, field: &FieldConfig) {
    if field.name == FIELD_SENTENCE_LENGTH {
        builder.add_u64_field(&field.name, STORED | FAST);
        log::debug!("Added u64 field '{}' as STORED and FAST", field.name);
    } else {
        builder.add_u64_field(&field.name, STORED);
    }
}

fn add_bytes_field(builder: &mut tantivy::schema::SchemaBuilder, field: &FieldConfig) {
    builder.add_bytes_field(&field.name, STORED);
}

fn get_output_fields(config: &SchemaConfig) -> Vec<String> {
    config.output_fields.clone().unwrap_or_else(|| {
        vec![
            FIELD_WORD.to_string(),
            FIELD_LEMMA.to_string(),
            FIELD_POS.to_string(),
            FIELD_ENTITY.to_string(),
        ]
    })
}