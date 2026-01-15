"""
AgentState Python SDK

A semantic state engine for AI agents with a unified intent-based API.
All AI processing (embeddings, intent classification) is powered by the Rust core.
"""

__version__ = "0.1.0"

# Import everything from the native Rust bindings
# The native module is required - there is no pure Python fallback
from agent_state._core import (
    # Main engine
    AgentEngine,
    # Enums
    DataType,
    Action,
    TimeFilter,
    # Intent classification
    Intent,
    # Response types
    StoredResponse,
    QueryResultResponse,
    NotFoundResponse,
    NeedsClarificationResponse,
)

# Type alias for union of all response types
AgentResponse = StoredResponse | QueryResultResponse | NotFoundResponse | NeedsClarificationResponse

__all__ = [
    # Version
    "__version__",
    # Main engine
    "AgentEngine",
    # Enums
    "DataType",
    "Action",
    "TimeFilter",
    # Intent
    "Intent",
    # Response types
    "AgentResponse",
    "StoredResponse",
    "QueryResultResponse",
    "NotFoundResponse",
    "NeedsClarificationResponse",
]
