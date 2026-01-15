//! Python bindings for AgentState using PyO3
//!
//! This module provides Python bindings for the AgentState semantic state engine.
//! All AI processing happens in the Rust core - Python is just a thin wrapper.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use agent_brain::{
    AgentEngine as RustEngine,
    AgentResponse as RustResponse,
    DataType as RustDataType,
    Action as RustAction,
    Intent as RustIntent,
    TimeFilter as RustTimeFilter,
};

/// Convert Rust errors to Python exceptions
fn to_py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

/// AgentState Python bindings
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AgentEngine>()?;
    m.add_class::<DataType>()?;
    m.add_class::<Action>()?;
    m.add_class::<Intent>()?;
    m.add_class::<TimeFilter>()?;
    m.add_class::<StoredResponse>()?;
    m.add_class::<QueryResultResponse>()?;
    m.add_class::<NotFoundResponse>()?;
    m.add_class::<NeedsClarificationResponse>()?;
    Ok(())
}

/// Type of data being stored or queried
#[pyclass]
#[derive(Clone, Debug)]
pub enum DataType {
    /// Action items, reminders, todos
    Task,
    /// General facts and information
    Memory,
    /// User preferences and likes/dislikes
    Preference,
    /// Relationships between entities
    Relationship,
    /// Time-based events and appointments
    Event,
}

impl From<RustDataType> for DataType {
    fn from(dt: RustDataType) -> Self {
        match dt {
            RustDataType::Task => DataType::Task,
            RustDataType::Memory => DataType::Memory,
            RustDataType::Preference => DataType::Preference,
            RustDataType::Relationship => DataType::Relationship,
            RustDataType::Event => DataType::Event,
        }
    }
}

impl From<DataType> for RustDataType {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::Task => RustDataType::Task,
            DataType::Memory => RustDataType::Memory,
            DataType::Preference => RustDataType::Preference,
            DataType::Relationship => RustDataType::Relationship,
            DataType::Event => RustDataType::Event,
        }
    }
}

#[pymethods]
impl DataType {
    /// Get the string representation of this data type
    fn as_str(&self) -> &'static str {
        match self {
            DataType::Task => "task",
            DataType::Memory => "memory",
            DataType::Preference => "preference",
            DataType::Relationship => "relationship",
            DataType::Event => "event",
        }
    }

    fn __repr__(&self) -> String {
        format!("DataType.{}", self.as_str().to_uppercase())
    }

    fn __str__(&self) -> &'static str {
        self.as_str()
    }
}

/// The action the agent wants to perform
#[pyclass]
#[derive(Clone, Debug)]
pub enum Action {
    /// Store information
    Store,
    /// Query/retrieve information
    Query,
}

impl From<RustAction> for Action {
    fn from(a: RustAction) -> Self {
        match a {
            RustAction::Store => Action::Store,
            RustAction::Query => Action::Query,
        }
    }
}

impl From<Action> for RustAction {
    fn from(a: Action) -> Self {
        match a {
            Action::Store => RustAction::Store,
            Action::Query => RustAction::Query,
        }
    }
}

#[pymethods]
impl Action {
    fn __repr__(&self) -> &'static str {
        match self {
            Action::Store => "Action.STORE",
            Action::Query => "Action.QUERY",
        }
    }
}

/// Time filter for queries
#[pyclass]
#[derive(Clone, Debug)]
pub enum TimeFilter {
    /// No time filtering
    All,
    /// Last 24 hours
    Today,
    /// Last 7 days
    LastWeek,
    /// Last 30 days
    LastMonth,
}

impl From<TimeFilter> for RustTimeFilter {
    fn from(tf: TimeFilter) -> Self {
        match tf {
            TimeFilter::All => RustTimeFilter::All,
            TimeFilter::Today => RustTimeFilter::Today,
            TimeFilter::LastWeek => RustTimeFilter::LastWeek,
            TimeFilter::LastMonth => RustTimeFilter::LastMonth,
        }
    }
}

#[pymethods]
impl TimeFilter {
    fn __repr__(&self) -> &'static str {
        match self {
            TimeFilter::All => "TimeFilter.ALL",
            TimeFilter::Today => "TimeFilter.TODAY",
            TimeFilter::LastWeek => "TimeFilter.LAST_WEEK",
            TimeFilter::LastMonth => "TimeFilter.LAST_MONTH",
        }
    }
}

/// Full intent classification result
#[pyclass]
#[derive(Clone)]
pub struct Intent {
    #[pyo3(get)]
    pub action: Action,
    #[pyo3(get)]
    pub data_type: DataType,
    #[pyo3(get)]
    pub action_confidence: f32,
    #[pyo3(get)]
    pub data_type_confidence: f32,
}

impl From<RustIntent> for Intent {
    fn from(i: RustIntent) -> Self {
        Intent {
            action: i.action.into(),
            data_type: i.data_type.into(),
            action_confidence: i.action_confidence,
            data_type_confidence: i.data_type_confidence,
        }
    }
}

#[pymethods]
impl Intent {
    /// Returns true if the classification is ambiguous
    fn is_ambiguous(&self) -> bool {
        self.action_confidence < 0.15 || self.data_type_confidence < 0.15
    }

    /// Returns the overall confidence
    fn overall_confidence(&self) -> f32 {
        self.action_confidence.min(self.data_type_confidence)
    }

    fn __repr__(&self) -> String {
        format!(
            "Intent(action={:?}, data_type={:?}, confidence={:.2})",
            self.action, self.data_type, self.overall_confidence()
        )
    }
}

/// Response when data is stored
#[pyclass]
#[derive(Clone)]
pub struct StoredResponse {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub data_type: DataType,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub latency_ms: f64,
}

#[pymethods]
impl StoredResponse {
    fn to_agent_string(&self) -> String {
        format!("Stored as {:?}: {}", self.data_type, self.content)
    }

    fn __repr__(&self) -> String {
        format!("StoredResponse(id={}, data_type={:?})", self.id, self.data_type)
    }
}

/// Response when a query returns results
#[pyclass]
#[derive(Clone)]
pub struct QueryResultResponse {
    #[pyo3(get)]
    pub results: Vec<String>,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub data_type: Option<DataType>,
    #[pyo3(get)]
    pub latency_ms: f64,
}

#[pymethods]
impl QueryResultResponse {
    fn to_agent_string(&self) -> String {
        if self.results.is_empty() {
            "No results found.".to_string()
        } else {
            self.results.join("\n")
        }
    }

    fn __repr__(&self) -> String {
        format!("QueryResultResponse(count={})", self.count)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<ResultsIterator>> {
        let iter = ResultsIterator {
            results: slf.results.clone(),
            index: 0,
        };
        Py::new(slf.py(), iter)
    }

    fn __len__(&self) -> usize {
        self.count
    }
}

#[pyclass]
struct ResultsIterator {
    results: Vec<String>,
    index: usize,
}

#[pymethods]
impl ResultsIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<String> {
        if slf.index < slf.results.len() {
            let result = slf.results[slf.index].clone();
            slf.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Response when no results are found
#[pyclass]
#[derive(Clone)]
pub struct NotFoundResponse {
    #[pyo3(get)]
    pub query: String,
    #[pyo3(get)]
    pub latency_ms: f64,
}

#[pymethods]
impl NotFoundResponse {
    fn to_agent_string(&self) -> String {
        format!("No information found for: {}", self.query)
    }

    fn __repr__(&self) -> String {
        format!("NotFoundResponse(query='{}')", self.query)
    }
}

/// Response when clarification is needed
#[pyclass]
#[derive(Clone)]
pub struct NeedsClarificationResponse {
    #[pyo3(get)]
    pub original_input: String,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub detected_intent: Intent,
    #[pyo3(get)]
    pub latency_ms: f64,
}

#[pymethods]
impl NeedsClarificationResponse {
    fn to_agent_string(&self) -> String {
        format!("I need more context: {}", self.message)
    }

    fn __repr__(&self) -> String {
        format!("NeedsClarificationResponse(message='{}')", self.message)
    }
}

/// Union type helper for response types
enum AgentResponseEnum {
    Stored(StoredResponse),
    QueryResult(QueryResultResponse),
    NotFound(NotFoundResponse),
    NeedsClarification(NeedsClarificationResponse),
}

impl AgentResponseEnum {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            AgentResponseEnum::Stored(r) => r.into_py(py),
            AgentResponseEnum::QueryResult(r) => r.into_py(py),
            AgentResponseEnum::NotFound(r) => r.into_py(py),
            AgentResponseEnum::NeedsClarification(r) => r.into_py(py),
        }
    }
}

fn convert_response(resp: RustResponse) -> AgentResponseEnum {
    match resp {
        RustResponse::Stored { id, data_type, content, latency_ms } => {
            AgentResponseEnum::Stored(StoredResponse {
                id,
                data_type: data_type.into(),
                content,
                latency_ms,
            })
        }
        RustResponse::QueryResult { results, data_type, count, latency_ms } => {
            AgentResponseEnum::QueryResult(QueryResultResponse {
                results,
                count,
                data_type: data_type.map(|dt| dt.into()),
                latency_ms,
            })
        }
        RustResponse::NotFound { query, latency_ms } => {
            AgentResponseEnum::NotFound(NotFoundResponse {
                query,
                latency_ms,
            })
        }
        RustResponse::NeedsClarification { original_input, message, detected_intent, latency_ms } => {
            AgentResponseEnum::NeedsClarification(NeedsClarificationResponse {
                original_input,
                message,
                detected_intent: detected_intent.into(),
                latency_ms,
            })
        }
    }
}

/// Semantic state engine for AI agents
///
/// Provides a unified API where agents express intent naturally,
/// and the engine determines the appropriate action.
///
/// All AI processing (embeddings, intent classification) happens in Rust.
#[pyclass]
pub struct AgentEngine {
    engine: RustEngine,
}

#[pymethods]
impl AgentEngine {
    /// Create a new AgentEngine
    ///
    /// Args:
    ///     db_path: Path to SQLite database. Uses in-memory if None or ":memory:".
    ///     mock: If True, uses mock embeddings for testing (no model download required).
    #[new]
    #[pyo3(signature = (db_path=None, mock=false))]
    fn new(db_path: Option<String>, mock: bool) -> PyResult<Self> {
        let path = db_path.as_deref().unwrap_or(":memory:");

        let engine = if mock {
            RustEngine::new_mock(path).map_err(to_py_err)?
        } else {
            RustEngine::new(path).map_err(to_py_err)?
        };

        Ok(AgentEngine { engine })
    }

    /// Create an in-memory engine (shorthand for AgentEngine(":memory:"))
    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        let engine = RustEngine::new_in_memory().map_err(to_py_err)?;
        Ok(AgentEngine { engine })
    }

    /// Create a mock engine for testing (no model download required)
    #[staticmethod]
    fn mock(db_path: Option<String>) -> PyResult<Self> {
        let path = db_path.as_deref().unwrap_or(":memory:");
        let engine = RustEngine::new_mock(path).map_err(to_py_err)?;
        Ok(AgentEngine { engine })
    }

    /// Returns whether this engine is running in mock mode
    fn is_mock(&self) -> bool {
        self.engine.is_mock()
    }

    /// Process natural language input
    ///
    /// Automatically detects whether the input is a store or query operation.
    ///
    /// Args:
    ///     text: Natural language input from the agent
    ///
    /// Returns:
    ///     One of: StoredResponse, QueryResultResponse, NotFoundResponse, NeedsClarificationResponse
    fn process(&mut self, py: Python<'_>, text: &str) -> PyResult<PyObject> {
        let response = self.engine.process(text).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Process with forced action (bypasses ambiguity check)
    fn process_as(&mut self, py: Python<'_>, text: &str, action: Action) -> PyResult<PyObject> {
        let response = self.engine.process_as(text, action.into()).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Explicitly store data
    fn store(&mut self, py: Python<'_>, text: &str) -> PyResult<PyObject> {
        let response = self.engine.store(text).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Store with explicit data type
    fn store_as(&mut self, py: Python<'_>, text: &str, data_type: DataType) -> PyResult<PyObject> {
        let response = self.engine.store_as(text, data_type.into()).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Batch store multiple items efficiently
    fn store_batch(&mut self, contents: Vec<String>) -> PyResult<Vec<i64>> {
        let refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
        self.engine.store_batch(&refs).map_err(to_py_err)
    }

    /// Explicitly search for data
    #[pyo3(signature = (query, limit=5))]
    fn search(&mut self, py: Python<'_>, query: &str, limit: usize) -> PyResult<PyObject> {
        let response = self.engine.search_with_limit(query, limit).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Search by data type
    #[pyo3(signature = (query, data_type, limit=5))]
    fn search_by_type(&mut self, py: Python<'_>, query: &str, data_type: DataType, limit: usize) -> PyResult<PyObject> {
        let response = self.engine.search_by_type(query, data_type.into(), limit).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Search with time filter
    #[pyo3(signature = (query, time_filter, limit=5))]
    fn search_with_time(&mut self, py: Python<'_>, query: &str, time_filter: TimeFilter, limit: usize) -> PyResult<PyObject> {
        let response = self.engine.search_with_time(query, time_filter.into(), limit).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Smart search with auto-detected time references
    #[pyo3(signature = (query, limit=5))]
    fn smart_search(&mut self, py: Python<'_>, query: &str, limit: usize) -> PyResult<PyObject> {
        let response = self.engine.smart_search(query, limit).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Search for tasks only
    #[pyo3(signature = (query, limit=5))]
    fn search_tasks(&mut self, py: Python<'_>, query: &str, limit: usize) -> PyResult<PyObject> {
        let response = self.engine.search_tasks(query, limit).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Search for memories only
    #[pyo3(signature = (query, limit=5))]
    fn search_memories(&mut self, py: Python<'_>, query: &str, limit: usize) -> PyResult<PyObject> {
        let response = self.engine.search_memories(query, limit).map_err(to_py_err)?;
        Ok(convert_response(response).into_py(py))
    }

    /// Classify intent without storing
    fn classify(&mut self, text: &str) -> PyResult<Intent> {
        let intent = self.engine.classify(text).map_err(to_py_err)?;
        Ok(intent.into())
    }

    /// Get total count of stored items
    fn count(&self) -> PyResult<usize> {
        self.engine.count().map_err(to_py_err)
    }

    /// Get count by data type
    fn count_by_type(&self, data_type: DataType) -> PyResult<usize> {
        self.engine.count_by_type(data_type.into()).map_err(to_py_err)
    }

    /// Delete an item by ID
    fn delete(&mut self, id: i64) -> PyResult<bool> {
        self.engine.delete(id).map_err(to_py_err)
    }

    /// Clear all stored data
    fn clear(&mut self) -> PyResult<()> {
        self.engine.clear().map_err(to_py_err)
    }

    /// Get metrics summary
    fn metrics_summary(&self) -> String {
        self.engine.metrics_summary()
    }

    /// Reset metrics
    fn reset_metrics(&self) {
        self.engine.reset_metrics()
    }

    fn __repr__(&self) -> String {
        let mock_str = if self.engine.is_mock() { ", mock=True" } else { "" };
        format!("AgentEngine(count={}{})", self.engine.count().unwrap_or(0), mock_str)
    }
}
