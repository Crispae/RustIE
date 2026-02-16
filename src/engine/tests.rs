//! Tests for the engine module (config, constants, schema, core, document, execution).

#![cfg(test)]

mod tests {
    use crate::data::document::{Document, Field, Sentence};
    use crate::engine::config::{FieldConfig, SchemaConfig};
    use crate::engine::constants::get_required_field;
    use crate::engine::schema::create_schema_from_yaml;
    use crate::engine::ExtractorEngine;
    use std::fs;
    use std::path::Path;
    use tantivy::schema::{Schema, SchemaBuilder, TEXT, STORED};

    // ==================== Helpers ====================

    fn minimal_schema_yaml() -> String {
        r#"
output_fields:
  - "word"
fields:
  - name: "word"
    type: "string"
    stored: true
  - name: "lemma"
    type: "string"
    stored: true
  - name: "doc_id"
    type: "text"
    stored: true
  - name: "sentence_id"
    type: "text"
    stored: true
  - name: "sentence_length"
    type: "u64"
    stored: true
  - name: "dependencies_binary"
    type: "bytes"
    stored: true
  - name: "incoming_edges"
    type: "edge_positions"
    stored: true
  - name: "outgoing_edges"
    type: "edge_positions"
    stored: true
"#
        .trim()
        .to_string()
    }

    fn write_schema_to_temp_dir(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        let schema_path = tmp.path().join("schema.yaml");
        fs::write(&schema_path, minimal_schema_yaml()).expect("write schema");
        schema_path
    }

    fn minimal_document() -> Document {
        Document {
            id: "test_doc".to_string(),
            metadata: vec![],
            sentences: vec![Sentence {
                numTokens: 3,
                fields: vec![
                    Field::TokensField {
                        name: "word".to_string(),
                        tokens: vec!["John".to_string(), "eats".to_string(), "pizza".to_string()],
                    },
                    Field::GraphField {
                        name: "dependencies".to_string(),
                        roots: vec![1],
                        edges: vec![
                            (1, 0, "nsubj".to_string()),
                            (1, 2, "dobj".to_string()),
                        ],
                    },
                ],
            }],
        }
    }

    fn engine_with_temp_dir() -> (ExtractorEngine, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let schema_path = write_schema_to_temp_dir(&tmp);
        let engine = ExtractorEngine::new(tmp.path(), &schema_path).expect("create engine");
        (engine, tmp)
    }

    // ==================== Config tests ====================

    #[test]
    fn test_config_deserialize_schema_config() {
        let yaml = r#"
output_fields:
  - "word"
  - "lemma"
fields:
  - name: "word"
    type: "string"
    stored: true
  - name: "doc_id"
    type: "text"
    stored: false
"#;
        let config: SchemaConfig = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(config.output_fields.as_ref().map(|v| v.len()), Some(2));
        assert_eq!(config.fields.len(), 2);
        assert_eq!(config.fields[0].name, "word");
        assert_eq!(config.fields[0].field_type, "string");
        assert!(config.fields[0].stored);
        assert_eq!(config.fields[1].name, "doc_id");
        assert!(!config.fields[1].stored);
    }

    #[test]
    fn test_config_deserialize_field_config() {
        let yaml = r#"
name: "sentence_length"
type: "u64"
stored: true
"#;
        let field: FieldConfig = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(field.name, "sentence_length");
        assert_eq!(field.field_type, "u64");
        assert!(field.stored);
    }

    // ==================== Constants tests ====================

    #[test]
    fn test_constants_get_required_field_success() {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("word", TEXT | STORED);
        let schema: Schema = builder.build();
        let result = get_required_field(&schema, "word");
        assert!(result.is_ok(), "get_required_field should succeed: {:?}", result.err());
        let _field = result.unwrap();
        assert!(schema.get_field("word").is_ok());
    }

    #[test]
    fn test_constants_get_required_field_missing() {
        let mut builder = SchemaBuilder::new();
        builder.add_text_field("word", TEXT | STORED);
        let schema: Schema = builder.build();
        let result = get_required_field(&schema, "nonexistent");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("nonexistent") || err_msg.contains("not found"),
            "Error should mention field: {}",
            err_msg
        );
    }

    // ==================== Schema tests ====================

    #[test]
    fn test_schema_create_from_yaml_valid() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let schema_path = write_schema_to_temp_dir(&tmp);
        let result = create_schema_from_yaml(&schema_path);
        assert!(result.is_ok(), "create_schema_from_yaml should succeed: {:?}", result.err());
        let (schema, output_fields) = result.unwrap();
        assert!(schema.get_field("word").is_ok());
        assert!(schema.get_field("dependencies_binary").is_ok());
        assert!(schema.get_field("incoming_edges").is_ok());
        assert!(schema.get_field("outgoing_edges").is_ok());
        assert!(!output_fields.is_empty());
    }

    #[test]
    fn test_schema_create_from_yaml_missing_file() {
        let result = create_schema_from_yaml(Path::new("nonexistent_schema_12345.yaml"));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found") || err_msg.contains("Schema file"),
            "Error should mention missing file: {}",
            err_msg
        );
    }

    #[test]
    fn test_schema_create_from_yaml_invalid_yaml() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let schema_path = tmp.path().join("bad.yaml");
        fs::write(&schema_path, "fields: [").expect("write");
        let result = create_schema_from_yaml(&schema_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_create_from_yaml_unknown_field_type() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let schema_path = tmp.path().join("unknown_type.yaml");
        let yaml = r#"
output_fields: ["word"]
fields:
  - name: "word"
    type: "unknown_type"
    stored: true
"#;
        fs::write(&schema_path, yaml).expect("write");
        let result = create_schema_from_yaml(&schema_path);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unknown field type") || err_msg.contains("unknown_type"),
            "Error should mention unknown type: {}",
            err_msg
        );
    }

    // ==================== Core tests ====================

    #[test]
    fn test_core_engine_new_success() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let schema_path = write_schema_to_temp_dir(&tmp);
        let result = ExtractorEngine::new(tmp.path(), &schema_path);
        assert!(result.is_ok(), "ExtractorEngine::new should succeed: {:?}", result.err());
        let engine = result.unwrap();
        assert_eq!(engine.num_docs(), 0);
        assert!(engine.schema().get_field("word").is_ok());
        assert!(engine.compiler().compile_pattern(&crate::query::ast::Pattern::Constraint(
            crate::query::ast::Constraint::Wildcard
        )).is_ok());
        assert!(engine.parser().parse_query("[word=test]").is_ok());
        assert!(!engine.output_fields().is_empty());
    }

    #[test]
    fn test_core_engine_new_missing_schema() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let result = ExtractorEngine::new(tmp.path(), Path::new("nonexistent.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_core_engine_from_path_invalid() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let nonexistent = tmp.path().join("nested").join("index_does_not_exist");
        let result = ExtractorEngine::from_path(nonexistent.to_str().unwrap());
        assert!(result.is_err(), "from_path with nonexistent index dir should fail");
    }

    #[test]
    fn test_core_engine_accessors() {
        let (engine, _tmp) = engine_with_temp_dir();
        assert!(engine.schema().get_field("word").is_ok());
        let _default = engine.default_field();
        let outputs = engine.output_fields();
        assert!(!outputs.is_empty());
        assert!(engine.is_output_field("word"));
    }

    // ==================== Document tests ====================

    #[test]
    fn test_document_add_commit_num_docs() {
        let (mut engine, _tmp) = engine_with_temp_dir();
        let doc = minimal_document();
        engine.add_document(&doc).expect("add_document");
        engine.commit().expect("commit");
        assert_eq!(engine.num_docs(), 1);
    }

    #[test]
    fn test_document_extract_field_value_and_values() {
        let (engine, _tmp) = engine_with_temp_dir();
        let schema = engine.schema();
        let mut tantivy_doc = tantivy::schema::TantivyDocument::default();
        tantivy_doc.add_text(schema.get_field("doc_id").unwrap(), "test_doc");
        tantivy_doc.add_text(schema.get_field("sentence_id").unwrap(), "test_doc_0");
        tantivy_doc.add_u64(schema.get_field("sentence_length").unwrap(), 3);
        tantivy_doc.add_text(schema.get_field("word").unwrap(), "John|eats|pizza");

        let doc_id = engine.extract_field_value(&tantivy_doc, "doc_id");
        assert_eq!(doc_id.as_deref(), Some("test_doc"));
        let words = engine.extract_field_values(&tantivy_doc, "word");
        assert_eq!(words, vec!["John", "eats", "pizza"]);
    }

    #[test]
    fn test_document_extract_sentence_result() {
        let (engine, _tmp) = engine_with_temp_dir();
        let schema = engine.schema();
        let mut tantivy_doc = tantivy::schema::TantivyDocument::default();
        tantivy_doc.add_text(schema.get_field("doc_id").unwrap(), "test_doc");
        tantivy_doc.add_text(schema.get_field("sentence_id").unwrap(), "test_doc_0");
        tantivy_doc.add_u64(schema.get_field("sentence_length").unwrap(), 3);
        tantivy_doc.add_text(schema.get_field("word").unwrap(), "John|eats|pizza");

        let sentence_result = engine.extract_sentence_result(&tantivy_doc, 1.0).expect("extract_sentence_result");
        assert_eq!(sentence_result.document_id.as_ref(), "test_doc");
        assert_eq!(sentence_result.sentence_id.as_ref(), "test_doc_0");
        assert!(sentence_result.fields.contains_key("word"));
        assert_eq!(
            sentence_result.fields.get("word").unwrap(),
            &["John".to_string(), "eats".to_string(), "pizza".to_string()]
        );
    }

    // ==================== Execution tests ====================

    #[test]
    fn test_execution_query_with_limit_basic() {
        let (mut engine, _tmp) = engine_with_temp_dir();
        let doc = minimal_document();
        engine.add_document(&doc).expect("add_document");
        engine.commit().expect("commit");

        let result = engine.query_with_limit("[word=John]", 10).expect("query");
        assert!(result.total_hits >= 1, "expected at least one hit");
        assert!(!result.sentence_results.is_empty());
    }

    #[test]
    fn test_execution_query_no_match() {
        let (mut engine, _tmp) = engine_with_temp_dir();
        let doc = minimal_document();
        engine.add_document(&doc).expect("add_document");
        engine.commit().expect("commit");

        let result = engine.query("[word=nonexistentword]").expect("query");
        assert_eq!(result.total_hits, 0);
        assert!(result.sentence_results.is_empty());
    }

    #[test]
    fn test_execution_query_with_limit_respected() {
        let (mut engine, _tmp) = engine_with_temp_dir();
        let doc = minimal_document();
        engine.add_document(&doc).expect("add_document");
        engine.commit().expect("commit");

        let result = engine.query_with_limit("[word=John]", 1).expect("query");
        assert!(result.sentence_results.len() <= 1, "limit 1 should return at most 1 result");
    }
}
