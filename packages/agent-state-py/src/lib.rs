//! Python bindings for AgentState using PyO3
//!
//! This module provides Python bindings for the AgentState semantic state engine.
//! It wraps the core Rust functionality and exposes it to Python.

use pyo3::prelude::*;

/// AgentState Python bindings
///
/// Provides a semantic state engine for AI agents with a unified intent-based API.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AgentEngine>()?;
    m.add_class::<DataType>()?;
    m.add_class::<AgentResponse>()?;
    Ok(())
}

/// Type of data being stored or queried
#[pyclass]
#[derive(Clone)]
pub enum DataType {
    Task,
    Memory,
}

/// Response from AgentEngine operations
#[pyclass]
#[derive(Clone)]
pub struct AgentResponse {
    #[pyo3(get)]
    pub response_type: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub id: Option<i64>,
    #[pyo3(get)]
    pub results: Option<Vec<String>>,
    #[pyo3(get)]
    pub count: Option<usize>,
}

#[pymethods]
impl AgentResponse {
    fn to_agent_string(&self) -> String {
        self.content.clone()
    }

    fn __repr__(&self) -> String {
        format!("AgentResponse(type={}, content={})", self.response_type, self.content)
    }
}

/// Semantic state engine for AI agents
///
/// Provides a unified API where agents express intent naturally,
/// and the engine determines the appropriate action.
#[pyclass]
pub struct AgentEngine {
    // TODO: Add actual engine implementation
    // This will wrap the Rust AgentEngine from agent-state-rs
    db_path: Option<String>,
}

#[pymethods]
impl AgentEngine {
    /// Create a new AgentEngine
    ///
    /// Args:
    ///     db_path: Optional path to SQLite database. Uses in-memory if None.
    #[new]
    #[pyo3(signature = (db_path=None))]
    fn new(db_path: Option<String>) -> Self {
        AgentEngine { db_path }
    }

    /// Process natural language input
    ///
    /// Automatically detects whether the input is a store or query operation.
    ///
    /// Args:
    ///     text: Natural language input from the agent
    ///
    /// Returns:
    ///     AgentResponse with the result
    fn process(&self, text: &str) -> PyResult<AgentResponse> {
        // TODO: Implement actual processing using the Rust engine
        // For now, return a placeholder response
        Ok(AgentResponse {
            response_type: "processed".to_string(),
            content: format!("Processed: {}", text),
            id: None,
            results: None,
            count: None,
        })
    }

    /// Explicitly store data
    ///
    /// Args:
    ///     text: Content to store
    ///     data_type: Optional data type (auto-detected if None)
    #[pyo3(signature = (text, data_type=None))]
    fn store(&self, text: &str, data_type: Option<DataType>) -> PyResult<AgentResponse> {
        let dt = data_type.map(|d| match d {
            DataType::Task => "task",
            DataType::Memory => "memory",
        }).unwrap_or("auto");

        Ok(AgentResponse {
            response_type: "stored".to_string(),
            content: text.to_string(),
            id: Some(1), // TODO: Return actual ID
            results: None,
            count: None,
        })
    }

    /// Explicitly search for data
    ///
    /// Args:
    ///     query: Search query
    ///     data_type: Optional filter by data type
    ///     limit: Maximum number of results (default 5)
    #[pyo3(signature = (query, data_type=None, limit=5))]
    fn search(&self, query: &str, data_type: Option<DataType>, limit: usize) -> PyResult<AgentResponse> {
        // TODO: Implement actual search
        Ok(AgentResponse {
            response_type: "query_result".to_string(),
            content: format!("Search results for: {}", query),
            id: None,
            results: Some(vec![]),
            count: Some(0),
        })
    }
}
