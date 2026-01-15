"""
AgentState Python SDK

A semantic state engine for AI agents with a unified intent-based API.
All AI processing (embeddings, intent classification) is powered by the Rust core.

Supports multiple embedding models and backends:
- Models: MiniLM-L6, MiniLM-L12, BGE-Small (default), BGE-Base, E5-Small, GTE-Small
- Backends: Candle (pure Rust, default), ONNX Runtime (optimized inference)

Example usage:
    from agent_state import AgentEngine, EmbeddingModel, Backend

    # Default configuration (BGE-Small with Candle)
    engine = AgentEngine()

    # With specific model for higher accuracy
    engine = AgentEngine(model=EmbeddingModel.BgeBase)

    # With ONNX backend for faster inference
    engine = AgentEngine(model=EmbeddingModel.BgeSmall, backend=Backend.Onnx)

    # Mock mode for testing (no model download)
    engine = AgentEngine(mock=True)
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
    # Model configuration
    EmbeddingModel,
    Backend,
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
    # Model configuration
    "EmbeddingModel",
    "Backend",
    # Intent
    "Intent",
    # Response types
    "AgentResponse",
    "StoredResponse",
    "QueryResultResponse",
    "NotFoundResponse",
    "NeedsClarificationResponse",
]
