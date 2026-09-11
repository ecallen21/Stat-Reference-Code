"""Function Calling (OpenAI-style, Reference Sec 47.204).

OpenAI 2023 'Function Calling'; anthropic 2024 'Tool Use'. Instead
of embedding tool calls in natural language (Toolformer / ReAct),
the LM is trained to emit STRUCTURED JSON calls when it decides a
tool is needed:

    user: "What's the weather in Tokyo?"
    assistant: { "name": "get_weather",
                 "arguments": { "city": "Tokyo", "unit": "celsius" } }
    tool_result: { "temp": 18, "conditions": "cloudy" }
    assistant: "It's 18 °C and cloudy in Tokyo."

The tool schemas are declared in advance (JSON schema); the LM
chooses among them by name and fills required arguments.
"""
from __future__ import annotations    # stdlib

import json    # tool call parsing


TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "calc",
        "description": "Evaluate a numeric expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def toy_lm_decide(prompt):
    """Toy LM: rule-based routing to the right tool."""
    p = prompt.lower()
    if "weather" in p:
        # Extract city — very naive
        for c in ["tokyo", "paris", "new york"]:
            if c in p:
                return {"name": "get_weather", "arguments": {"city": c.title(), "unit": "celsius"}}
    if any(op in p for op in ["+", "-", "*", "/"]) or "calculate" in p:
        import re
        m = re.search(r"[\d]+(?:\s*[+\-*/]\s*[\d]+)+", prompt)
        if m:
            return {"name": "calc", "arguments": {"expression": m.group(0)}}
    return None                                                  # no tool needed


def execute_tool(call):
    """Toy tool sandbox."""
    if call["name"] == "get_weather":
        db = {"Tokyo": {"temp": 18, "conditions": "cloudy"},
               "Paris": {"temp": 12, "conditions": "rain"},
               "New York": {"temp": 22, "conditions": "sunny"}}
        return db.get(call["arguments"]["city"], {"error": "unknown city"})
    if call["name"] == "calc":
        try: return {"result": eval(call["arguments"]["expression"])}
        except Exception: return {"error": "calc-error"}
    return {"error": "unknown tool"}


def validate_call(call, schemas):
    """Check that the call matches one of the declared schemas."""
    for s in schemas:
        if s["name"] == call["name"]:
            required = s["parameters"].get("required", [])
            if all(r in call["arguments"] for r in required):
                return True
    return False


if __name__ == "__main__":
    print("=== Function Calling / Tool Use (OpenAI 2023, Anthropic 2024) ===\n")

    prompts = [
        "What's the weather in Tokyo?",
        "Calculate 42 * 17 + 5.",
        "Tell me a joke.",                                       # no tool needed
        "How's the weather in Paris?",
    ]

    for p in prompts:
        call = toy_lm_decide(p)
        if call is None:
            print(f"  User: {p}\n    Assistant (direct): [answers from parametric memory]\n")
            continue
        valid = validate_call(call, TOOL_SCHEMAS)
        result = execute_tool(call) if valid else {"error": "invalid call"}
        print(f"  User: {p}")
        print(f"    Assistant (tool call): {json.dumps(call)}")
        print(f"    Validated against schema: {valid}")
        print(f"    Tool result: {json.dumps(result)}\n")

    print("  Structured function calling is JSON-schema-validated, unlike free-form")
    print("  Toolformer / ReAct, so downstream code can execute directly.")

    print("\n--- library cross-check (openai.chat.completions.create tool_choice; anthropic messages tools) ---")
