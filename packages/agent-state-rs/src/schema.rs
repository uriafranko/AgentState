//! Schema Module - Structured Data Extraction & Validation
//!
//! This module enables agents to define schemas for structured data types and
//! automatically extract typed fields from natural language input.
//!
//! # Example
//! ```no_run
//! use agent_brain::schema::{Schema, FieldType};
//!
//! let contact = Schema::new("contact")
//!     .field("name", FieldType::String)
//!     .field("email", FieldType::Email)
//!     .optional_field("company", FieldType::String)
//!     .optional_field("phone", FieldType::Phone);
//! ```

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Field types supported by the schema system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Plain text string
    String,
    /// Email address (validated format)
    Email,
    /// Phone number (various formats supported)
    Phone,
    /// Date (ISO 8601 or natural language)
    Date,
    /// Integer number
    Integer,
    /// Floating point number
    Float,
    /// Boolean (true/false, yes/no)
    Boolean,
    /// URL (validated format)
    Url,
}

impl FieldType {
    /// Returns a regex pattern for extracting this field type
    fn extraction_pattern(&self) -> &'static str {
        match self {
            FieldType::String => r"[A-Za-z][A-Za-z\s\-'\.]+",
            FieldType::Email => r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            FieldType::Phone => r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}",
            FieldType::Date => r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?",
            FieldType::Integer => r"-?\d+",
            FieldType::Float => r"-?\d+\.?\d*",
            FieldType::Boolean => r"(?i)true|false|yes|no",
            FieldType::Url => r"https?://[^\s]+",
        }
    }

    /// Validates that a value matches this field type
    pub fn validate(&self, value: &str) -> bool {
        let pattern = format!("^{}$", self.extraction_pattern());
        let re = Regex::new(&pattern).unwrap();

        match self {
            FieldType::Boolean => {
                let lower = value.to_lowercase();
                matches!(lower.as_str(), "true" | "false" | "yes" | "no")
            }
            FieldType::Integer => value.parse::<i64>().is_ok(),
            FieldType::Float => value.parse::<f64>().is_ok(),
            _ => re.is_match(value),
        }
    }

    /// Normalizes a value to a standard format
    pub fn normalize(&self, value: &str) -> String {
        match self {
            FieldType::Boolean => {
                let lower = value.to_lowercase();
                if lower == "yes" || lower == "true" {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            FieldType::Email => value.to_lowercase(),
            FieldType::Phone => {
                // Remove all non-digits except leading +
                let cleaned: String = value.chars()
                    .filter(|c| c.is_ascii_digit() || *c == '+')
                    .collect();
                cleaned
            }
            _ => value.trim().to_string(),
        }
    }

    /// Returns the category name for this field type
    pub fn as_str(&self) -> &'static str {
        match self {
            FieldType::String => "string",
            FieldType::Email => "email",
            FieldType::Phone => "phone",
            FieldType::Date => "date",
            FieldType::Integer => "integer",
            FieldType::Float => "float",
            FieldType::Boolean => "boolean",
            FieldType::Url => "url",
        }
    }

    /// Parse from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "string" => Some(FieldType::String),
            "email" => Some(FieldType::Email),
            "phone" => Some(FieldType::Phone),
            "date" => Some(FieldType::Date),
            "integer" => Some(FieldType::Integer),
            "float" => Some(FieldType::Float),
            "boolean" => Some(FieldType::Boolean),
            "url" => Some(FieldType::Url),
            _ => None,
        }
    }
}

/// Definition of a single field in a schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Whether this field is required
    pub required: bool,
    /// Optional default value
    pub default: Option<String>,
    /// Optional description for the field
    pub description: Option<String>,
    /// Aliases for this field (alternative names to look for)
    pub aliases: Vec<String>,
}

impl FieldDefinition {
    /// Creates a new required field definition
    pub fn new(name: &str, field_type: FieldType) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required: true,
            default: None,
            description: None,
            aliases: Vec::new(),
        }
    }

    /// Creates a new optional field definition
    pub fn optional(name: &str, field_type: FieldType) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required: false,
            default: None,
            description: None,
            aliases: Vec::new(),
        }
    }

    /// Sets the default value for this field
    pub fn with_default(mut self, default: &str) -> Self {
        self.default = Some(default.to_string());
        self
    }

    /// Sets the description for this field
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Adds aliases for this field
    pub fn with_aliases(mut self, aliases: &[&str]) -> Self {
        self.aliases = aliases.iter().map(|s| s.to_string()).collect();
        self
    }
}

/// A schema defines the structure of a data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Unique name for this schema
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Field definitions in order
    pub fields: Vec<FieldDefinition>,
}

impl Schema {
    /// Creates a new empty schema
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            fields: Vec::new(),
        }
    }

    /// Adds a description to the schema
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Adds a required field to the schema
    pub fn field(mut self, name: &str, field_type: FieldType) -> Self {
        self.fields.push(FieldDefinition::new(name, field_type));
        self
    }

    /// Adds an optional field to the schema
    pub fn optional_field(mut self, name: &str, field_type: FieldType) -> Self {
        self.fields.push(FieldDefinition::optional(name, field_type));
        self
    }

    /// Adds a field with full definition
    pub fn add_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// Gets a field by name
    pub fn get_field(&self, name: &str) -> Option<&FieldDefinition> {
        self.fields.iter().find(|f| f.name == name || f.aliases.contains(&name.to_string()))
    }

    /// Returns all required field names
    pub fn required_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.required)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Serializes the schema to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("Failed to serialize schema")
    }

    /// Deserializes a schema from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("Failed to deserialize schema")
    }
}

/// Extracted structured data from natural language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedData {
    /// The schema this data conforms to
    pub schema_name: String,
    /// Extracted field values
    pub fields: HashMap<String, String>,
    /// Original raw text that was parsed
    pub raw_text: String,
    /// Confidence score for the extraction (0.0 to 1.0)
    pub confidence: f32,
    /// Fields that couldn't be extracted
    pub missing_fields: Vec<String>,
    /// Validation errors (field name -> error message)
    pub validation_errors: HashMap<String, String>,
}

impl ExtractedData {
    /// Creates new extracted data
    pub fn new(schema_name: &str, raw_text: &str) -> Self {
        Self {
            schema_name: schema_name.to_string(),
            fields: HashMap::new(),
            raw_text: raw_text.to_string(),
            confidence: 0.0,
            missing_fields: Vec::new(),
            validation_errors: HashMap::new(),
        }
    }

    /// Returns true if extraction was fully successful (all required fields present, no errors)
    pub fn is_complete(&self) -> bool {
        self.missing_fields.is_empty() && self.validation_errors.is_empty()
    }

    /// Gets a field value
    pub fn get(&self, field: &str) -> Option<&String> {
        self.fields.get(field)
    }

    /// Gets a field value as a specific type
    pub fn get_as<T: std::str::FromStr>(&self, field: &str) -> Option<T> {
        self.fields.get(field).and_then(|v| v.parse().ok())
    }

    /// Serializes to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("Failed to serialize extracted data")
    }

    /// Just the fields as JSON (for compact storage)
    pub fn fields_to_json(&self) -> Result<String> {
        serde_json::to_string(&self.fields).context("Failed to serialize fields")
    }
}

/// Validation result for extracted data
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors by field
    pub errors: HashMap<String, String>,
    /// Missing required fields
    pub missing_required: Vec<String>,
}

impl ValidationResult {
    /// Creates a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: HashMap::new(),
            missing_required: Vec::new(),
        }
    }

    /// Creates a failed validation result
    pub fn failure(errors: HashMap<String, String>, missing: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            missing_required: missing,
        }
    }
}

/// Extractor handles the logic of extracting structured data from text
pub struct Extractor;

impl Extractor {
    /// Extracts structured data from text based on a schema
    ///
    /// Uses a combination of:
    /// 1. Pattern matching for specific field types (email, phone, etc.)
    /// 2. Contextual hints (e.g., "email: ", "phone number is ")
    /// 3. Position-based extraction for ordered schemas
    pub fn extract(text: &str, schema: &Schema) -> ExtractedData {
        let mut data = ExtractedData::new(&schema.name, text);
        let text_lower = text.to_lowercase();
        let mut extracted_count = 0;

        for field in &schema.fields {
            if let Some(value) = Self::extract_field(text, &text_lower, field) {
                let normalized = field.field_type.normalize(&value);
                if field.field_type.validate(&normalized) {
                    data.fields.insert(field.name.clone(), normalized);
                    extracted_count += 1;
                } else {
                    data.validation_errors.insert(
                        field.name.clone(),
                        format!("Invalid {} format: {}", field.field_type.as_str(), value),
                    );
                }
            } else {
                // Field not extracted - apply default if available, or mark as missing
                if let Some(ref default) = field.default {
                    data.fields.insert(field.name.clone(), default.clone());
                    extracted_count += 1;
                } else if field.required {
                    data.missing_fields.push(field.name.clone());
                }
            }
        }

        // Calculate confidence based on extraction success
        let total_fields = schema.fields.len() as f32;
        if total_fields > 0.0 {
            data.confidence = extracted_count as f32 / total_fields;
        } else {
            data.confidence = 1.0;
        }

        data
    }

    /// Extracts a single field from text
    fn extract_field(text: &str, text_lower: &str, field: &FieldDefinition) -> Option<String> {
        // First, try contextual extraction with field name hints
        if let Some(value) = Self::extract_with_context(text, text_lower, field) {
            return Some(value);
        }

        // Then try pattern-based extraction
        Self::extract_by_pattern(text, field)
    }

    /// Extracts using contextual hints (e.g., "name: John", "email is john@example.com")
    fn extract_with_context(text: &str, text_lower: &str, field: &FieldDefinition) -> Option<String> {
        let field_name_lower = field.name.to_lowercase();

        // Build list of names to search for (including aliases)
        let mut search_names = vec![field_name_lower.clone()];
        for alias in &field.aliases {
            search_names.push(alias.to_lowercase());
        }

        for name in &search_names {
            // Try patterns like "name: value", "name is value", "name = value"
            let patterns = [
                format!(r"{}[\s]*[:=][\s]*", regex::escape(name)),
                format!(r"{}\s+is\s+", regex::escape(name)),
                format!(r"{}\s+", regex::escape(name)),
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(&format!("(?i){}", pattern)) {
                    if let Some(m) = re.find(text_lower) {
                        let after_match = &text[m.end()..];
                        // Extract the value after the match
                        if let Some(value) = Self::extract_value_after(after_match, &field.field_type) {
                            return Some(value);
                        }
                    }
                }
            }
        }

        None
    }

    /// Extracts the value portion after a field label
    fn extract_value_after(text: &str, field_type: &FieldType) -> Option<String> {
        let pattern = field_type.extraction_pattern();
        let re = Regex::new(pattern).ok()?;

        // Find the first match
        re.find(text).map(|m| m.as_str().to_string())
    }

    /// Extracts using only the field type pattern (fallback)
    fn extract_by_pattern(text: &str, field: &FieldDefinition) -> Option<String> {
        let pattern = field.field_type.extraction_pattern();
        let re = Regex::new(pattern).ok()?;

        // For certain types, we can extract without context
        match field.field_type {
            FieldType::Email | FieldType::Phone | FieldType::Url => {
                // These are distinctive enough to extract directly
                re.find(text).map(|m| m.as_str().to_string())
            }
            FieldType::Date => {
                // Dates are also fairly distinctive
                re.find(text).map(|m| m.as_str().to_string())
            }
            _ => None, // String, Integer, etc. need context
        }
    }

    /// Validates extracted data against a schema
    pub fn validate(data: &ExtractedData, schema: &Schema) -> ValidationResult {
        let mut errors = HashMap::new();
        let mut missing = Vec::new();

        for field in &schema.fields {
            match data.fields.get(&field.name) {
                Some(value) => {
                    if !field.field_type.validate(value) {
                        errors.insert(
                            field.name.clone(),
                            format!("Invalid {} format", field.field_type.as_str()),
                        );
                    }
                }
                None if field.required && field.default.is_none() => {
                    missing.push(field.name.clone());
                }
                None => {}
            }
        }

        if errors.is_empty() && missing.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failure(errors, missing)
        }
    }
}

/// Schema registry for managing multiple schemas
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    schemas: HashMap<String, Schema>,
}

impl SchemaRegistry {
    /// Creates a new empty registry
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Registers a schema
    pub fn register(&mut self, schema: Schema) {
        self.schemas.insert(schema.name.clone(), schema);
    }

    /// Gets a schema by name
    pub fn get(&self, name: &str) -> Option<&Schema> {
        self.schemas.get(name)
    }

    /// Removes a schema
    pub fn unregister(&mut self, name: &str) -> Option<Schema> {
        self.schemas.remove(name)
    }

    /// Lists all registered schema names
    pub fn list(&self) -> Vec<&str> {
        self.schemas.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of registered schemas
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Returns true if no schemas are registered
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    /// Clears all registered schemas
    pub fn clear(&mut self) {
        self.schemas.clear();
    }
}

// ============================================================================
// Common Schema Templates
// ============================================================================

/// Creates a Contact schema
pub fn contact_schema() -> Schema {
    Schema::new("contact")
        .with_description("Contact information for a person")
        .add_field(
            FieldDefinition::new("name", FieldType::String)
                .with_description("Full name of the contact")
                .with_aliases(&["person", "contact"]),
        )
        .add_field(
            FieldDefinition::optional("email", FieldType::Email)
                .with_description("Email address")
                .with_aliases(&["mail", "e-mail"]),
        )
        .add_field(
            FieldDefinition::optional("phone", FieldType::Phone)
                .with_description("Phone number")
                .with_aliases(&["tel", "telephone", "mobile", "cell"]),
        )
        .add_field(
            FieldDefinition::optional("company", FieldType::String)
                .with_description("Company or organization")
                .with_aliases(&["org", "organization", "employer", "works at"]),
        )
}

/// Creates a Task schema
pub fn task_schema() -> Schema {
    Schema::new("task")
        .with_description("An action item or todo")
        .add_field(
            FieldDefinition::new("title", FieldType::String)
                .with_description("Task title or description"),
        )
        .add_field(
            FieldDefinition::optional("due_date", FieldType::Date)
                .with_description("Due date for the task")
                .with_aliases(&["deadline", "due", "by"]),
        )
        .add_field(
            FieldDefinition::optional("priority", FieldType::String)
                .with_description("Priority level (high, medium, low)")
                .with_default("medium"),
        )
        .add_field(
            FieldDefinition::optional("assignee", FieldType::String)
                .with_description("Person assigned to the task")
                .with_aliases(&["assigned to", "owner", "responsible"]),
        )
}

/// Creates an Event schema
pub fn event_schema() -> Schema {
    Schema::new("event")
        .with_description("A calendar event or appointment")
        .add_field(
            FieldDefinition::new("title", FieldType::String)
                .with_description("Event title"),
        )
        .add_field(
            FieldDefinition::optional("date", FieldType::Date)
                .with_description("Event date")
                .with_aliases(&["when", "on"]),
        )
        .add_field(
            FieldDefinition::optional("location", FieldType::String)
                .with_description("Event location")
                .with_aliases(&["where", "at", "venue"]),
        )
        .add_field(
            FieldDefinition::optional("attendees", FieldType::String)
                .with_description("People attending")
                .with_aliases(&["with", "participants"]),
        )
}

/// Creates a Note schema
pub fn note_schema() -> Schema {
    Schema::new("note")
        .with_description("A simple note or piece of information")
        .add_field(
            FieldDefinition::new("content", FieldType::String)
                .with_description("Note content"),
        )
        .add_field(
            FieldDefinition::optional("tags", FieldType::String)
                .with_description("Comma-separated tags")
                .with_aliases(&["labels", "categories"]),
        )
        .add_field(
            FieldDefinition::optional("source", FieldType::String)
                .with_description("Source of the information")
                .with_aliases(&["from", "origin"]),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_type_validation() {
        assert!(FieldType::Email.validate("test@example.com"));
        assert!(!FieldType::Email.validate("not-an-email"));

        assert!(FieldType::Phone.validate("555-1234"));
        assert!(FieldType::Phone.validate("+1 (555) 123-4567"));

        assert!(FieldType::Integer.validate("42"));
        assert!(FieldType::Integer.validate("-10"));
        assert!(!FieldType::Integer.validate("3.14"));

        assert!(FieldType::Float.validate("3.14"));
        assert!(FieldType::Float.validate("-2.5"));

        assert!(FieldType::Boolean.validate("true"));
        assert!(FieldType::Boolean.validate("YES"));
        assert!(FieldType::Boolean.validate("no"));

        assert!(FieldType::Url.validate("https://example.com"));
        assert!(!FieldType::Url.validate("not-a-url"));
    }

    #[test]
    fn test_schema_creation() {
        let schema = Schema::new("test")
            .field("name", FieldType::String)
            .optional_field("email", FieldType::Email);

        assert_eq!(schema.name, "test");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.fields[0].required);
        assert!(!schema.fields[1].required);
    }

    #[test]
    fn test_extract_email() {
        let schema = Schema::new("contact")
            .field("name", FieldType::String)
            .field("email", FieldType::Email);

        let text = "Add John Smith, email: john@example.com";
        let data = Extractor::extract(text, &schema);

        assert_eq!(data.get("email"), Some(&"john@example.com".to_string()));
    }

    #[test]
    fn test_extract_phone() {
        let schema = Schema::new("contact")
            .optional_field("phone", FieldType::Phone);

        let text = "Call me at 555-123-4567";
        let data = Extractor::extract(text, &schema);

        assert!(data.get("phone").is_some());
    }

    #[test]
    fn test_extract_with_context() {
        let schema = Schema::new("contact")
            .field("name", FieldType::String)
            .optional_field("company", FieldType::String);

        let text = "name: John Doe, company: Acme Corp";
        let data = Extractor::extract(text, &schema);

        assert_eq!(data.get("name"), Some(&"John Doe".to_string()));
        assert_eq!(data.get("company"), Some(&"Acme Corp".to_string()));
    }

    #[test]
    fn test_contact_schema_template() {
        let schema = contact_schema();
        assert_eq!(schema.name, "contact");
        assert!(schema.get_field("name").is_some());
        assert!(schema.get_field("email").is_some());
        assert!(schema.get_field("phone").is_some());
    }

    #[test]
    fn test_schema_registry() {
        let mut registry = SchemaRegistry::new();

        registry.register(contact_schema());
        registry.register(task_schema());

        assert_eq!(registry.len(), 2);
        assert!(registry.get("contact").is_some());
        assert!(registry.get("task").is_some());
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_extracted_data_completeness() {
        let schema = Schema::new("test")
            .field("required_field", FieldType::String)
            .optional_field("optional_field", FieldType::String);

        let text = "required_field: value";
        let data = Extractor::extract(text, &schema);

        // Should be complete since the only required field is present
        assert!(data.is_complete());
    }

    #[test]
    fn test_validation_result() {
        let schema = Schema::new("test")
            .field("email", FieldType::Email);

        let text = "email: not-valid";
        let data = Extractor::extract(text, &schema);
        let result = Extractor::validate(&data, &schema);

        // Should fail validation due to invalid email
        assert!(!result.is_valid);
    }

    #[test]
    fn test_field_normalization() {
        assert_eq!(FieldType::Boolean.normalize("YES"), "true");
        assert_eq!(FieldType::Boolean.normalize("no"), "false");
        assert_eq!(FieldType::Email.normalize("TEST@Example.COM"), "test@example.com");
        assert_eq!(FieldType::Phone.normalize("+1 (555) 123-4567"), "+15551234567");
    }

    #[test]
    fn test_schema_serialization() {
        let schema = contact_schema();
        let json = schema.to_json().unwrap();
        let restored = Schema::from_json(&json).unwrap();

        assert_eq!(restored.name, schema.name);
        assert_eq!(restored.fields.len(), schema.fields.len());
    }
}
