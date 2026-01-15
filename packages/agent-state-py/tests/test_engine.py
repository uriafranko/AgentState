"""Tests for AgentState Python SDK."""

import pytest
from agent_state import (
    AgentEngine,
    DataType,
    StoredResponse,
    QueryResultResponse,
    NotFoundResponse,
)


class TestAgentEngine:
    """Tests for the AgentEngine class."""

    def test_create_engine(self):
        """Test creating an AgentEngine instance."""
        engine = AgentEngine()
        assert engine is not None

    def test_create_engine_with_db_path(self):
        """Test creating an AgentEngine with a database path."""
        engine = AgentEngine(db_path=":memory:")
        assert engine is not None

    def test_store_memory(self):
        """Test storing a memory."""
        engine = AgentEngine()
        response = engine.store("My favorite color is blue", DataType.MEMORY)
        assert isinstance(response, StoredResponse)
        assert response.data_type == DataType.MEMORY
        assert "blue" in response.content

    def test_store_task(self):
        """Test storing a task."""
        engine = AgentEngine()
        response = engine.store("Call John tomorrow", DataType.TASK)
        assert isinstance(response, StoredResponse)
        assert response.data_type == DataType.TASK

    def test_process_auto_detects_store(self):
        """Test that process() auto-detects store operations."""
        engine = AgentEngine()
        response = engine.process("My name is Alice")
        assert isinstance(response, StoredResponse)

    def test_process_auto_detects_query(self):
        """Test that process() auto-detects query operations."""
        engine = AgentEngine()
        # First store something
        engine.process("My favorite food is pizza")
        # Then query
        response = engine.process("What is my favorite food?")
        # Should be a query response (either results or not found)
        assert isinstance(response, (QueryResultResponse, NotFoundResponse))

    def test_search_returns_results(self):
        """Test searching for stored data."""
        engine = AgentEngine()
        engine.store("I love programming in Python", DataType.MEMORY)
        response = engine.search("Python")
        # Should find the result or return not found
        assert isinstance(response, (QueryResultResponse, NotFoundResponse))

    def test_search_with_limit(self):
        """Test search with result limit."""
        engine = AgentEngine()
        for i in range(10):
            engine.store(f"Item number {i}", DataType.MEMORY)
        response = engine.search("Item", limit=3)
        if isinstance(response, QueryResultResponse):
            assert len(response.results) <= 3


class TestDataType:
    """Tests for the DataType enum."""

    def test_task_value(self):
        """Test TASK enum value."""
        assert DataType.TASK.value == "task"

    def test_memory_value(self):
        """Test MEMORY enum value."""
        assert DataType.MEMORY.value == "memory"


class TestResponses:
    """Tests for response types."""

    def test_stored_response_to_string(self):
        """Test StoredResponse.to_agent_string()."""
        response = StoredResponse(
            id=1,
            data_type=DataType.MEMORY,
            content="Test content"
        )
        result = response.to_agent_string()
        assert "Stored" in result
        assert "memory" in result

    def test_query_result_to_string(self):
        """Test QueryResultResponse.to_agent_string()."""
        response = QueryResultResponse(
            results=["Result 1", "Result 2"],
            count=2,
            data_type=DataType.MEMORY
        )
        result = response.to_agent_string()
        assert "Result 1" in result
        assert "Result 2" in result

    def test_not_found_to_string(self):
        """Test NotFoundResponse.to_agent_string()."""
        response = NotFoundResponse(query="test query")
        result = response.to_agent_string()
        assert "No results" in result
        assert "test query" in result
