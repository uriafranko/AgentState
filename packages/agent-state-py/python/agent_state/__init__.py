"""
AgentState Python SDK

A semantic state engine for AI agents with a unified intent-based API.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union

__version__ = "0.1.0"

# Try to import the native Rust bindings, fall back to pure Python implementation
try:
    from agent_state._core import (
        AgentEngine as _NativeEngine,
        AgentResponse as _NativeResponse,
    )
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


class DataType(Enum):
    """Type of data being stored or queried."""
    TASK = "task"
    MEMORY = "memory"


class Action(Enum):
    """Type of action to perform."""
    STORE = "store"
    QUERY = "query"


@dataclass
class StoredResponse:
    """Response when data is stored."""
    id: int
    data_type: DataType
    content: str

    def to_agent_string(self) -> str:
        """Convert to human-readable string."""
        return f"Stored {self.data_type.value}: {self.content}"


@dataclass
class QueryResultResponse:
    """Response when a query returns results."""
    results: List[str]
    count: int
    data_type: Optional[DataType]

    def to_agent_string(self) -> str:
        """Convert to human-readable string."""
        if not self.results:
            return "No results found."
        return "\n".join(f"- {r}" for r in self.results)


@dataclass
class NotFoundResponse:
    """Response when no results are found."""
    query: str

    def to_agent_string(self) -> str:
        """Convert to human-readable string."""
        return f"No results found for: {self.query}"


AgentResponse = Union[StoredResponse, QueryResultResponse, NotFoundResponse]


class AgentEngine:
    """
    Semantic state engine for AI agents.

    Provides a unified API where agents just express intent naturally,
    and the engine figures out the rest.

    Example:
        >>> engine = AgentEngine()
        >>> engine.process("My favorite color is blue")  # Auto-stores as MEMORY
        >>> engine.process("What is my favorite color?")  # Auto-queries
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the AgentEngine.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory database.
        """
        if _HAS_NATIVE:
            self._engine = _NativeEngine(db_path)
        else:
            # Pure Python fallback (limited functionality)
            self._db_path = db_path
            self._storage: List[dict] = []
            self._id_counter = 0

    def process(self, text: str) -> AgentResponse:
        """
        Process natural language input.

        Automatically detects whether the input is a store or query operation,
        and what type of data is involved.

        Args:
            text: Natural language input from the agent

        Returns:
            AgentResponse with the result of the operation
        """
        if _HAS_NATIVE:
            return self._engine.process(text)

        # Pure Python fallback - simple keyword-based classification
        text_lower = text.lower()

        # Simple query detection
        is_query = any(q in text_lower for q in [
            "what", "who", "where", "when", "how", "find", "search",
            "show", "list", "get", "tell me", "?"
        ])

        if is_query:
            return self.search(text)
        else:
            return self.store(text)

    def store(
        self,
        text: str,
        data_type: Optional[DataType] = None
    ) -> AgentResponse:
        """
        Explicitly store data.

        Args:
            text: Content to store
            data_type: Type of data (auto-detected if None)

        Returns:
            StoredResponse with storage details
        """
        if _HAS_NATIVE:
            return self._engine.store(text, data_type)

        # Pure Python fallback
        if data_type is None:
            # Simple task detection
            text_lower = text.lower()
            is_task = any(t in text_lower for t in [
                "remind", "todo", "task", "call", "meeting", "schedule"
            ])
            data_type = DataType.TASK if is_task else DataType.MEMORY

        self._id_counter += 1
        self._storage.append({
            "id": self._id_counter,
            "content": text,
            "data_type": data_type
        })

        return StoredResponse(
            id=self._id_counter,
            data_type=data_type,
            content=text
        )

    def search(
        self,
        query: str,
        data_type: Optional[DataType] = None,
        limit: int = 5
    ) -> AgentResponse:
        """
        Explicitly search for data.

        Args:
            query: Search query
            data_type: Filter by data type (None for all)
            limit: Maximum number of results

        Returns:
            QueryResultResponse or NotFoundResponse
        """
        if _HAS_NATIVE:
            return self._engine.search(query, data_type, limit)

        # Pure Python fallback - simple keyword matching
        results = []
        query_words = set(query.lower().split())

        for item in self._storage:
            if data_type and item["data_type"] != data_type:
                continue

            content_words = set(item["content"].lower().split())
            if query_words & content_words:
                results.append(item["content"])

        results = results[:limit]

        if not results:
            return NotFoundResponse(query=query)

        return QueryResultResponse(
            results=results,
            count=len(results),
            data_type=data_type
        )


__all__ = [
    "AgentEngine",
    "AgentResponse",
    "StoredResponse",
    "QueryResultResponse",
    "NotFoundResponse",
    "DataType",
    "Action",
    "__version__",
]
