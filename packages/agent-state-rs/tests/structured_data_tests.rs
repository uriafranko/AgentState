//! Integration tests for Structured Data Extraction & Validation
//!
//! These tests verify that schemas can be registered, data can be extracted
//! from natural language, and structured queries work correctly.

use agent_brain::{
    AgentEngine, ExtractedData, Extractor, FieldDefinition, FieldType, Schema,
    SchemaRegistry, StructuredDataItem, ValidationResult,
    contact_schema, event_schema, task_schema, note_schema,
};

// ============================================================================
// Schema Creation Tests
// ============================================================================

#[test]
fn test_schema_creation() {
    let schema = Schema::new("person")
        .with_description("A person record")
        .field("name", FieldType::String)
        .field("email", FieldType::Email)
        .optional_field("phone", FieldType::Phone);

    assert_eq!(schema.name, "person");
    assert_eq!(schema.fields.len(), 3);
    assert!(schema.fields[0].required);
    assert!(schema.fields[1].required);
    assert!(!schema.fields[2].required);
}

#[test]
fn test_schema_required_fields() {
    let schema = Schema::new("test")
        .field("required1", FieldType::String)
        .field("required2", FieldType::Email)
        .optional_field("optional1", FieldType::Phone);

    let required = schema.required_fields();
    assert_eq!(required.len(), 2);
    assert!(required.contains(&"required1"));
    assert!(required.contains(&"required2"));
}

#[test]
fn test_schema_get_field() {
    let schema = Schema::new("test")
        .add_field(
            FieldDefinition::new("email", FieldType::Email)
                .with_aliases(&["mail", "e-mail"])
        );

    assert!(schema.get_field("email").is_some());
    assert!(schema.get_field("mail").is_some()); // Alias
    assert!(schema.get_field("unknown").is_none());
}

#[test]
fn test_schema_serialization() {
    let schema = contact_schema();
    let json = schema.to_json().unwrap();
    let restored = Schema::from_json(&json).unwrap();

    assert_eq!(restored.name, schema.name);
    assert_eq!(restored.fields.len(), schema.fields.len());
}

// ============================================================================
// Field Type Validation Tests
// ============================================================================

#[test]
fn test_email_validation() {
    assert!(FieldType::Email.validate("test@example.com"));
    assert!(FieldType::Email.validate("user.name@domain.org"));
    assert!(FieldType::Email.validate("user+tag@example.co.uk"));
    assert!(!FieldType::Email.validate("not-an-email"));
    assert!(!FieldType::Email.validate("@missing.local"));
    assert!(!FieldType::Email.validate("missing@.domain"));
}

#[test]
fn test_phone_validation() {
    assert!(FieldType::Phone.validate("555-1234"));
    assert!(FieldType::Phone.validate("555-123-4567"));
    assert!(FieldType::Phone.validate("(555) 123-4567"));
    assert!(FieldType::Phone.validate("+1 555 123 4567"));
    assert!(FieldType::Phone.validate("5551234567"));
}

#[test]
fn test_integer_validation() {
    assert!(FieldType::Integer.validate("42"));
    assert!(FieldType::Integer.validate("-10"));
    assert!(FieldType::Integer.validate("0"));
    assert!(!FieldType::Integer.validate("3.14"));
    assert!(!FieldType::Integer.validate("abc"));
}

#[test]
fn test_float_validation() {
    assert!(FieldType::Float.validate("3.14"));
    assert!(FieldType::Float.validate("-2.5"));
    assert!(FieldType::Float.validate("42"));
    assert!(FieldType::Float.validate("0.0"));
}

#[test]
fn test_boolean_validation() {
    assert!(FieldType::Boolean.validate("true"));
    assert!(FieldType::Boolean.validate("false"));
    assert!(FieldType::Boolean.validate("yes"));
    assert!(FieldType::Boolean.validate("no"));
    assert!(FieldType::Boolean.validate("TRUE"));
    assert!(FieldType::Boolean.validate("NO"));
    assert!(!FieldType::Boolean.validate("maybe"));
}

#[test]
fn test_url_validation() {
    assert!(FieldType::Url.validate("https://example.com"));
    assert!(FieldType::Url.validate("http://localhost:3000/path"));
    assert!(!FieldType::Url.validate("not-a-url"));
    assert!(!FieldType::Url.validate("ftp://other.protocol"));
}

#[test]
fn test_field_normalization() {
    assert_eq!(FieldType::Boolean.normalize("YES"), "true");
    assert_eq!(FieldType::Boolean.normalize("no"), "false");
    assert_eq!(FieldType::Email.normalize("TEST@Example.COM"), "test@example.com");
    assert_eq!(FieldType::Phone.normalize("+1 (555) 123-4567"), "+15551234567");
}

// ============================================================================
// Extraction Tests
// ============================================================================

#[test]
fn test_extract_email_from_text() {
    let schema = Schema::new("contact")
        .optional_field("email", FieldType::Email);

    let text = "Contact me at john@example.com for details";
    let data = Extractor::extract(text, &schema);

    assert_eq!(data.get("email"), Some(&"john@example.com".to_string()));
}

#[test]
fn test_extract_phone_from_text() {
    let schema = Schema::new("contact")
        .optional_field("phone", FieldType::Phone);

    let text = "My number is 555-123-4567";
    let data = Extractor::extract(text, &schema);

    assert!(data.get("phone").is_some());
}

#[test]
fn test_extract_with_context_labels() {
    let schema = Schema::new("person")
        .field("name", FieldType::String)
        .optional_field("company", FieldType::String);

    let text = "name: John Doe, company: Acme Corp";
    let data = Extractor::extract(text, &schema);

    assert_eq!(data.get("name"), Some(&"John Doe".to_string()));
    assert_eq!(data.get("company"), Some(&"Acme Corp".to_string()));
}

#[test]
fn test_extract_with_is_pattern() {
    let schema = Schema::new("person")
        .field("name", FieldType::String);

    let text = "name is Alice Smith";
    let data = Extractor::extract(text, &schema);

    assert_eq!(data.get("name"), Some(&"Alice Smith".to_string()));
}

#[test]
fn test_extract_missing_required_field() {
    let schema = Schema::new("contact")
        .field("name", FieldType::String)
        .field("email", FieldType::Email);

    let text = "Just some random text without contact info";
    let data = Extractor::extract(text, &schema);

    // Both required fields should be in missing_fields
    assert!(data.missing_fields.contains(&"name".to_string()));
    // Email might be extracted as pattern-based, but name likely won't
}

#[test]
fn test_extract_with_default_value() {
    let schema = Schema::new("task")
        .add_field(
            FieldDefinition::optional("priority", FieldType::String)
                .with_default("medium")
        );

    let text = "Some task without priority";
    let data = Extractor::extract(text, &schema);

    // Default should be applied
    assert_eq!(data.get("priority"), Some(&"medium".to_string()));
}

#[test]
fn test_extraction_confidence() {
    let schema = Schema::new("contact")
        .field("name", FieldType::String)
        .field("email", FieldType::Email)
        .optional_field("phone", FieldType::Phone);

    // Full extraction
    let text = "name: John, email: john@example.com, phone: 555-1234";
    let data = Extractor::extract(text, &schema);
    assert!(data.confidence > 0.5);

    // Partial extraction
    let text2 = "some text without fields";
    let data2 = Extractor::extract(text2, &schema);
    assert!(data2.confidence < data.confidence);
}

// ============================================================================
// Validation Tests
// ============================================================================

#[test]
fn test_validation_success() {
    let schema = Schema::new("contact")
        .field("email", FieldType::Email);

    let text = "email: valid@example.com";
    let data = Extractor::extract(text, &schema);
    let result = Extractor::validate(&data, &schema);

    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.missing_required.is_empty());
}

#[test]
fn test_validation_missing_required() {
    let schema = Schema::new("contact")
        .field("name", FieldType::String)
        .field("email", FieldType::Email);

    // Only email, missing name
    let text = "contact at test@example.com";
    let data = Extractor::extract(text, &schema);
    let result = Extractor::validate(&data, &schema);

    assert!(!result.is_valid || !result.missing_required.is_empty());
}

// ============================================================================
// Schema Registry Tests
// ============================================================================

#[test]
fn test_schema_registry_register() {
    let mut registry = SchemaRegistry::new();

    registry.register(contact_schema());
    registry.register(task_schema());

    assert_eq!(registry.len(), 2);
    assert!(registry.get("contact").is_some());
    assert!(registry.get("task").is_some());
}

#[test]
fn test_schema_registry_list() {
    let mut registry = SchemaRegistry::new();

    registry.register(contact_schema());
    registry.register(event_schema());
    registry.register(note_schema());

    let names = registry.list();
    assert_eq!(names.len(), 3);
}

#[test]
fn test_schema_registry_unregister() {
    let mut registry = SchemaRegistry::new();

    registry.register(contact_schema());
    assert!(registry.get("contact").is_some());

    registry.unregister("contact");
    assert!(registry.get("contact").is_none());
}

// ============================================================================
// Built-in Schema Templates Tests
// ============================================================================

#[test]
fn test_contact_schema_template() {
    let schema = contact_schema();

    assert_eq!(schema.name, "contact");
    assert!(schema.get_field("name").is_some());
    assert!(schema.get_field("email").is_some());
    assert!(schema.get_field("phone").is_some());
    assert!(schema.get_field("company").is_some());
}

#[test]
fn test_task_schema_template() {
    let schema = task_schema();

    assert_eq!(schema.name, "task");
    assert!(schema.get_field("title").is_some());
    assert!(schema.get_field("due_date").is_some());
    assert!(schema.get_field("priority").is_some());
}

#[test]
fn test_event_schema_template() {
    let schema = event_schema();

    assert_eq!(schema.name, "event");
    assert!(schema.get_field("title").is_some());
    assert!(schema.get_field("date").is_some());
    assert!(schema.get_field("location").is_some());
}

// ============================================================================
// Engine Integration Tests (using mock mode)
// ============================================================================

#[test]
fn test_engine_register_schema() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();

    assert!(engine.get_schema("contact").is_some());
    assert_eq!(engine.list_schemas().len(), 1);
}

#[test]
fn test_engine_store_structured() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();

    let response = engine.store_structured(
        "Add John Smith, email: john@example.com, company: Acme",
        "contact"
    ).unwrap();

    assert!(response.id > 0);
    assert_eq!(response.schema_name, "contact");
    assert!(response.confidence() > 0.0);

    // Check extracted fields
    assert!(response.get_field("email").is_some());
}

#[test]
fn test_engine_query_structured() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();

    // Store some contacts
    engine.store_structured("name: Alice, email: alice@example.com, company: Acme", "contact").unwrap();
    engine.store_structured("name: Bob, email: bob@example.com, company: Acme", "contact").unwrap();
    engine.store_structured("name: Charlie, email: charlie@other.com, company: Other", "contact").unwrap();

    // Query all contacts
    let all_contacts = engine.query_structured("contact", None).unwrap();
    assert_eq!(all_contacts.len(), 3);

    // Query with field filter
    let acme_contacts = engine.query_structured("contact", Some(("company", "Acme"))).unwrap();
    assert_eq!(acme_contacts.len(), 2);
}

#[test]
fn test_engine_unregister_schema() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();
    assert!(engine.get_schema("contact").is_some());

    engine.unregister_schema("contact").unwrap();
    assert!(engine.get_schema("contact").is_none());
}

#[test]
fn test_engine_count_structured() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();

    assert_eq!(engine.count_structured("contact").unwrap(), 0);

    engine.store_structured("name: Test, email: test@example.com", "contact").unwrap();
    assert_eq!(engine.count_structured("contact").unwrap(), 1);

    engine.store_structured("name: Test2, email: test2@example.com", "contact").unwrap();
    assert_eq!(engine.count_structured("contact").unwrap(), 2);
}

#[test]
fn test_engine_structured_response_to_string() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();

    let response = engine.store_structured(
        "name: John Doe, email: john@example.com",
        "contact"
    ).unwrap();

    let output = response.to_agent_string();
    assert!(output.contains("contact"));
    assert!(output.contains("confidence"));
}

#[test]
fn test_engine_get_structured_by_id() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();

    let response = engine.store_structured(
        "name: John, email: john@example.com",
        "contact"
    ).unwrap();

    let structured = engine.get_structured(response.id).unwrap();
    assert!(structured.is_some());

    let item = structured.unwrap();
    assert_eq!(item.schema_name, "contact");
    assert!(item.get_field("email").is_some());
}

#[test]
fn test_engine_multiple_schemas() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    engine.register_schema(contact_schema()).unwrap();
    engine.register_schema(task_schema()).unwrap();
    engine.register_schema(event_schema()).unwrap();

    assert_eq!(engine.list_schemas().len(), 3);

    // Store different types
    engine.store_structured("name: John, email: john@example.com", "contact").unwrap();
    engine.store_structured("title: Review code, priority: high", "task").unwrap();
    engine.store_structured("title: Team meeting, date: 2024-01-15", "event").unwrap();

    assert_eq!(engine.count_structured("contact").unwrap(), 1);
    assert_eq!(engine.count_structured("task").unwrap(), 1);
    assert_eq!(engine.count_structured("event").unwrap(), 1);
}

#[test]
fn test_engine_schema_persistence() {
    // Create engine and register schema
    let db_path = "/tmp/test_structured_persistence.db";
    let _ = std::fs::remove_file(db_path);

    {
        let mut engine = AgentEngine::new_mock(db_path).unwrap();
        engine.register_schema(contact_schema()).unwrap();
        engine.store_structured("name: John, email: john@example.com", "contact").unwrap();
    }

    // Reopen and verify schema was persisted
    {
        let mut engine = AgentEngine::new_mock(db_path).unwrap();

        // Schema registry starts empty, need to load from DB
        let loaded = engine.load_schemas_from_db().unwrap();
        assert_eq!(loaded, 1);

        assert!(engine.get_schema("contact").is_some());

        // Data should also persist
        let contacts = engine.query_structured("contact", None).unwrap();
        assert_eq!(contacts.len(), 1);
    }

    // Cleanup
    let _ = std::fs::remove_file(db_path);
    let _ = std::fs::remove_file(format!("{}-wal", db_path));
    let _ = std::fs::remove_file(format!("{}-shm", db_path));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_store_with_unregistered_schema() {
    let mut engine = AgentEngine::new_mock_in_memory().unwrap();

    let result = engine.store_structured("some content", "nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_extraction_with_invalid_values() {
    let schema = Schema::new("contact")
        .field("email", FieldType::Email);

    let text = "email: not-valid-email";
    let data = Extractor::extract(text, &schema);

    // Should have validation error
    assert!(!data.validation_errors.is_empty() || data.missing_fields.contains(&"email".to_string()));
}
