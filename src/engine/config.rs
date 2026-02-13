//! Schema configuration types

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SchemaConfig {
    pub output_fields: Option<Vec<String>>,
    pub fields: Vec<FieldConfig>,
}

#[derive(Debug, Deserialize)]
pub struct FieldConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub stored: bool,
}