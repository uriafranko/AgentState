import { describe, it, expect, beforeEach } from "vitest";
import {
  AgentEngine,
  DataType,
  toAgentString,
  type StoredResponse,
  type QueryResultResponse,
  type NotFoundResponse,
} from "../src";

describe("AgentEngine", () => {
  let engine: AgentEngine;

  beforeEach(() => {
    engine = new AgentEngine();
  });

  describe("constructor", () => {
    it("should create an engine instance", () => {
      expect(engine).toBeInstanceOf(AgentEngine);
    });

    it("should accept options", () => {
      const engineWithOptions = new AgentEngine({ dbPath: ":memory:" });
      expect(engineWithOptions).toBeInstanceOf(AgentEngine);
    });
  });

  describe("store", () => {
    it("should store a memory", () => {
      const response = engine.store("My favorite color is blue", DataType.MEMORY);
      expect(response.type).toBe("stored");
      expect(response.dataType).toBe(DataType.MEMORY);
      expect(response.content).toContain("blue");
    });

    it("should store a task", () => {
      const response = engine.store("Call John tomorrow", DataType.TASK);
      expect(response.type).toBe("stored");
      expect(response.dataType).toBe(DataType.TASK);
    });

    it("should auto-detect task type", () => {
      const response = engine.store("Remind me to call John");
      expect(response.dataType).toBe(DataType.TASK);
    });

    it("should auto-detect memory type", () => {
      const response = engine.store("My name is Alice");
      expect(response.dataType).toBe(DataType.MEMORY);
    });

    it("should assign unique IDs", () => {
      const r1 = engine.store("First");
      const r2 = engine.store("Second");
      expect(r1.id).not.toBe(r2.id);
    });
  });

  describe("search", () => {
    it("should find stored data", () => {
      engine.store("I love programming in Python", DataType.MEMORY);
      const response = engine.search("Python");
      expect(response.type).toBe("query_result");
      if (response.type === "query_result") {
        expect(response.results).toHaveLength(1);
        expect(response.results[0]).toContain("Python");
      }
    });

    it("should return not found for missing data", () => {
      const response = engine.search("nonexistent");
      expect(response.type).toBe("not_found");
    });

    it("should filter by data type", () => {
      engine.store("Task: call John", DataType.TASK);
      engine.store("Memory: favorite color blue", DataType.MEMORY);

      const taskResults = engine.search("call", DataType.TASK);
      expect(taskResults.type).toBe("query_result");

      const memoryResults = engine.search("call", DataType.MEMORY);
      expect(memoryResults.type).toBe("not_found");
    });

    it("should respect limit", () => {
      for (let i = 0; i < 10; i++) {
        engine.store(`Item number ${i}`, DataType.MEMORY);
      }
      const response = engine.search("Item", undefined, 3);
      if (response.type === "query_result") {
        expect(response.results.length).toBeLessThanOrEqual(3);
      }
    });
  });

  describe("process", () => {
    it("should auto-detect store operations", () => {
      const response = engine.process("My name is Alice");
      expect(response.type).toBe("stored");
    });

    it("should auto-detect query operations", () => {
      engine.process("My name is Alice");
      const response = engine.process("What is my name?");
      expect(["query_result", "not_found"]).toContain(response.type);
    });

    it("should detect questions with ?", () => {
      const response = engine.process("Favorite color?");
      expect(["query_result", "not_found"]).toContain(response.type);
    });
  });
});

describe("toAgentString", () => {
  it("should format stored response", () => {
    const response: StoredResponse = {
      type: "stored",
      id: 1,
      dataType: DataType.MEMORY,
      content: "Test content",
    };
    const result = toAgentString(response);
    expect(result).toContain("Stored");
    expect(result).toContain("memory");
  });

  it("should format query results", () => {
    const response: QueryResultResponse = {
      type: "query_result",
      results: ["Result 1", "Result 2"],
      count: 2,
      dataType: DataType.MEMORY,
    };
    const result = toAgentString(response);
    expect(result).toContain("Result 1");
    expect(result).toContain("Result 2");
  });

  it("should format not found response", () => {
    const response: NotFoundResponse = {
      type: "not_found",
      query: "test query",
    };
    const result = toAgentString(response);
    expect(result).toContain("No results");
    expect(result).toContain("test query");
  });
});

describe("DataType", () => {
  it("should have correct values", () => {
    expect(DataType.TASK).toBe("task");
    expect(DataType.MEMORY).toBe("memory");
  });
});
