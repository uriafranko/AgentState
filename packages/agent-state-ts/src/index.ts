/**
 * AgentState TypeScript SDK
 *
 * A semantic state engine for AI agents with a unified intent-based API.
 *
 * @packageDocumentation
 */

/**
 * Type of data being stored or queried.
 */
export enum DataType {
  /** Action item or reminder */
  TASK = "task",
  /** Fact or preference */
  MEMORY = "memory",
}

/**
 * Type of action to perform.
 */
export enum Action {
  /** Store data */
  STORE = "store",
  /** Query data */
  QUERY = "query",
}

/**
 * Response when data is stored.
 */
export interface StoredResponse {
  type: "stored";
  id: number;
  dataType: DataType;
  content: string;
}

/**
 * Response when a query returns results.
 */
export interface QueryResultResponse {
  type: "query_result";
  results: string[];
  count: number;
  dataType: DataType | null;
}

/**
 * Response when no results are found.
 */
export interface NotFoundResponse {
  type: "not_found";
  query: string;
}

/**
 * Union type for all possible agent responses.
 */
export type AgentResponse = StoredResponse | QueryResultResponse | NotFoundResponse;

/**
 * Options for creating an AgentEngine.
 */
export interface AgentEngineOptions {
  /** Path to SQLite database (not yet implemented) */
  dbPath?: string;
}

/**
 * Convert an AgentResponse to a human-readable string.
 */
export function toAgentString(response: AgentResponse): string {
  switch (response.type) {
    case "stored":
      return `Stored ${response.dataType}: ${response.content}`;
    case "query_result":
      if (response.results.length === 0) {
        return "No results found.";
      }
      return response.results.map((r) => `- ${r}`).join("\n");
    case "not_found":
      return `No results found for: ${response.query}`;
  }
}

/**
 * Semantic state engine for AI agents.
 *
 * Provides a unified API where agents express intent naturally,
 * and the engine determines the appropriate action.
 *
 * @example
 * ```typescript
 * const engine = new AgentEngine();
 * engine.process("My favorite color is blue"); // Auto-stores as MEMORY
 * engine.process("What is my favorite color?"); // Auto-queries
 * ```
 *
 * @remarks
 * This is a placeholder implementation. Full functionality requires
 * native bindings to the Rust core (coming soon).
 */
export class AgentEngine {
  private storage: Array<{ id: number; content: string; dataType: DataType }> = [];
  private idCounter = 0;

  /**
   * Create a new AgentEngine.
   *
   * @param options - Engine configuration options
   */
  constructor(private options: AgentEngineOptions = {}) {
    // TODO: Initialize native bindings when available
  }

  /**
   * Process natural language input.
   *
   * Automatically detects whether the input is a store or query operation,
   * and what type of data is involved.
   *
   * @param text - Natural language input from the agent
   * @returns AgentResponse with the result of the operation
   */
  process(text: string): AgentResponse {
    const textLower = text.toLowerCase();

    // Simple query detection
    const isQuery = [
      "what",
      "who",
      "where",
      "when",
      "how",
      "find",
      "search",
      "show",
      "list",
      "get",
      "tell me",
    ].some((q) => textLower.includes(q)) || textLower.includes("?");

    if (isQuery) {
      return this.search(text);
    } else {
      return this.store(text);
    }
  }

  /**
   * Explicitly store data.
   *
   * @param text - Content to store
   * @param dataType - Type of data (auto-detected if not provided)
   * @returns StoredResponse with storage details
   */
  store(text: string, dataType?: DataType): StoredResponse {
    const textLower = text.toLowerCase();

    // Auto-detect data type if not provided
    const detectedType =
      dataType ??
      (["remind", "todo", "task", "call", "meeting", "schedule"].some((t) =>
        textLower.includes(t)
      )
        ? DataType.TASK
        : DataType.MEMORY);

    this.idCounter++;
    this.storage.push({
      id: this.idCounter,
      content: text,
      dataType: detectedType,
    });

    return {
      type: "stored",
      id: this.idCounter,
      dataType: detectedType,
      content: text,
    };
  }

  /**
   * Explicitly search for data.
   *
   * @param query - Search query
   * @param dataType - Filter by data type (searches all if not provided)
   * @param limit - Maximum number of results (default 5)
   * @returns QueryResultResponse or NotFoundResponse
   */
  search(query: string, dataType?: DataType, limit = 5): AgentResponse {
    const queryWords = new Set(query.toLowerCase().split(/\s+/));

    const results = this.storage
      .filter((item) => {
        if (dataType && item.dataType !== dataType) {
          return false;
        }
        const contentWords = new Set(item.content.toLowerCase().split(/\s+/));
        return [...queryWords].some((w) => contentWords.has(w));
      })
      .slice(0, limit)
      .map((item) => item.content);

    if (results.length === 0) {
      return {
        type: "not_found",
        query,
      };
    }

    return {
      type: "query_result",
      results,
      count: results.length,
      dataType: dataType ?? null,
    };
  }
}

export default AgentEngine;
