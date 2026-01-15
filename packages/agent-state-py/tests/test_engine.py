"""Tests for AgentState Python SDK.

These tests use mock mode which doesn't require the ML model to be downloaded.
"""

import pytest
from agent_state import (
    AgentEngine,
    DataType,
    Action,
    TimeFilter,
    Intent,
    StoredResponse,
    QueryResultResponse,
    NotFoundResponse,
    NeedsClarificationResponse,
)


class TestAgentEngine:
    """Tests for the AgentEngine class."""

    def test_create_engine_mock(self):
        """Test creating a mock AgentEngine instance."""
        engine = AgentEngine.mock()
        assert engine is not None
        assert engine.is_mock()

    def test_create_engine_with_db_path(self):
        """Test creating an AgentEngine with a database path."""
        engine = AgentEngine.mock(":memory:")
        assert engine is not None

    def test_store_returns_stored_response(self):
        """Test that store returns a StoredResponse."""
        engine = AgentEngine.mock()
        response = engine.store("My favorite color is blue")
        assert isinstance(response, StoredResponse)
        assert response.id > 0
        assert "blue" in response.content
        assert response.latency_ms >= 0

    def test_store_as_with_data_type(self):
        """Test storing with explicit data type."""
        engine = AgentEngine.mock()
        response = engine.store_as("Call John tomorrow", DataType.Task)
        assert isinstance(response, StoredResponse)
        assert response.data_type == DataType.Task

    def test_process_returns_response(self):
        """Test that process returns some response."""
        engine = AgentEngine.mock()
        response = engine.process("My name is Alice")
        # Could be stored, query result, not found, or needs clarification
        assert isinstance(response, (StoredResponse, QueryResultResponse, NotFoundResponse, NeedsClarificationResponse))

    def test_search_on_empty_db(self):
        """Test search on empty database."""
        engine = AgentEngine.mock()
        response = engine.search("anything")
        assert isinstance(response, NotFoundResponse)

    def test_search_returns_results(self):
        """Test searching for stored data."""
        engine = AgentEngine.mock()
        engine.store("I love programming in Python")
        response = engine.search("Python")
        # Should find the result or return not found (depends on mock embedding similarity)
        assert isinstance(response, (QueryResultResponse, NotFoundResponse))

    def test_search_with_limit(self):
        """Test search with result limit."""
        engine = AgentEngine.mock()
        for i in range(10):
            engine.store(f"Item number {i}")
        response = engine.search("Item", limit=3)
        if isinstance(response, QueryResultResponse):
            assert len(response.results) <= 3

    def test_store_batch(self):
        """Test batch storing multiple items."""
        engine = AgentEngine.mock()
        ids = engine.store_batch([
            "First item",
            "Second item",
            "Third item",
        ])
        assert len(ids) == 3
        assert all(id > 0 for id in ids)

    def test_count(self):
        """Test counting stored items."""
        engine = AgentEngine.mock()
        assert engine.count() == 0
        engine.store("Item 1")
        engine.store("Item 2")
        assert engine.count() == 2

    def test_delete(self):
        """Test deleting an item."""
        engine = AgentEngine.mock()
        response = engine.store("To be deleted")
        assert engine.count() == 1
        result = engine.delete(response.id)
        assert result is True
        assert engine.count() == 0

    def test_clear(self):
        """Test clearing all items."""
        engine = AgentEngine.mock()
        engine.store("Item 1")
        engine.store("Item 2")
        assert engine.count() == 2
        engine.clear()
        assert engine.count() == 0

    def test_classify(self):
        """Test intent classification."""
        engine = AgentEngine.mock()
        intent = engine.classify("What is my name?")
        assert isinstance(intent, Intent)
        assert hasattr(intent, "action")
        assert hasattr(intent, "data_type")
        assert hasattr(intent, "action_confidence")
        assert hasattr(intent, "data_type_confidence")

    def test_metrics_summary(self):
        """Test getting metrics summary."""
        engine = AgentEngine.mock()
        engine.store("Test item")
        summary = engine.metrics_summary()
        assert isinstance(summary, str)


class TestDataType:
    """Tests for the DataType enum."""

    def test_data_type_variants(self):
        """Test all DataType variants exist."""
        assert DataType.Task is not None
        assert DataType.Memory is not None
        assert DataType.Preference is not None
        assert DataType.Relationship is not None
        assert DataType.Event is not None

    def test_data_type_as_str(self):
        """Test DataType.as_str() method."""
        assert DataType.Task.as_str() == "task"
        assert DataType.Memory.as_str() == "memory"


class TestAction:
    """Tests for the Action enum."""

    def test_action_variants(self):
        """Test all Action variants exist."""
        assert Action.Store is not None
        assert Action.Query is not None


class TestTimeFilter:
    """Tests for the TimeFilter enum."""

    def test_time_filter_variants(self):
        """Test all TimeFilter variants exist."""
        assert TimeFilter.All is not None
        assert TimeFilter.Today is not None
        assert TimeFilter.LastWeek is not None
        assert TimeFilter.LastMonth is not None


class TestResponses:
    """Tests for response types."""

    def test_stored_response_to_string(self):
        """Test StoredResponse.to_agent_string()."""
        engine = AgentEngine.mock()
        response = engine.store("Test content")
        result = response.to_agent_string()
        assert "Stored" in result

    def test_query_result_iteration(self):
        """Test QueryResultResponse is iterable."""
        engine = AgentEngine.mock()
        engine.store("Item one")
        engine.store("Item two")
        response = engine.search("Item")
        if isinstance(response, QueryResultResponse):
            items = list(response)
            assert isinstance(items, list)

    def test_not_found_response_to_string(self):
        """Test NotFoundResponse.to_agent_string()."""
        engine = AgentEngine.mock()
        response = engine.search("nonexistent")
        assert isinstance(response, NotFoundResponse)
        result = response.to_agent_string()
        assert "No information found" in result


class TestIntent:
    """Tests for Intent class."""

    def test_intent_is_ambiguous(self):
        """Test Intent.is_ambiguous() method."""
        engine = AgentEngine.mock()
        intent = engine.classify("Hello")
        # Just check the method exists and returns bool
        assert isinstance(intent.is_ambiguous(), bool)

    def test_intent_overall_confidence(self):
        """Test Intent.overall_confidence() method."""
        engine = AgentEngine.mock()
        intent = engine.classify("Remember my birthday is tomorrow")
        confidence = intent.overall_confidence()
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
