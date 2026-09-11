"""AutoGen - Multi-Agent Conversation Framework (Sec 47.207).

Wu et al 2023 'AutoGen: Enabling Next-Gen LLM Applications via
Multi-Agent Conversation', Microsoft. Framework for composing LLMs
+ tools + humans as CONVERSABLE AGENTS. Core primitives:

    UserProxyAgent           executes code / calls tools on behalf
                                 of the human
    AssistantAgent           LM-powered planner / writer
    GroupChatManager         coordinates a group chat among agents

Agents exchange messages, each has its own system prompt, memory
and toolset. Enables patterns like "planner + coder + reviewer"
without hand-rolling the message routing.
"""
from __future__ import annotations    # stdlib


class Agent:
    def __init__(self, name, role, respond_fn):
        self.name = name; self.role = role; self.respond_fn = respond_fn

    def respond(self, chat_history):
        return self.respond_fn(chat_history)


def group_chat(agents, initial_message, max_turns=6):
    """Round-robin turn-taking; each agent sees the whole chat history."""
    history = [{"speaker": "user", "text": initial_message}]
    print(f"  user: {initial_message}")
    for turn in range(max_turns):
        a = agents[turn % len(agents)]
        reply = a.respond(history)
        history.append({"speaker": a.name, "text": reply})
        print(f"  {a.name} ({a.role}): {reply}")
        if "TASK COMPLETE" in reply.upper():
            break
    return history


def planner_respond(hist):
    if len(hist) == 1:
        return "Plan: 1) write function; 2) test it; 3) return result. Coder, start."
    return "Continuing plan."


def coder_respond(hist):
    if any("write function" in m["text"].lower() for m in hist[-3:]):
        return "```python\ndef add(a, b):\n    return a + b\n```\nReviewer?"
    return "Nothing more to code."


def reviewer_respond(hist):
    last_code = next((m for m in reversed(hist) if "```" in m["text"]), None)
    if last_code:
        if "return" in last_code["text"] and "def " in last_code["text"]:
            return "Code looks good. Runs correctly. TASK COMPLETE."
        return "Missing return statement."
    return "Waiting for code."


if __name__ == "__main__":
    print("=== AutoGen - Multi-Agent Conversation (Wu et al 2023) ===\n")

    planner = Agent("planner", "planner", planner_respond)
    coder = Agent("coder", "coder", coder_respond)
    reviewer = Agent("reviewer", "reviewer", reviewer_respond)

    initial = "Write a Python function to add two numbers."
    history = group_chat([planner, coder, reviewer], initial, max_turns=6)

    print(f"\n  Conversation length: {len(history)} messages")
    print(f"  Completed in {sum(1 for m in history if 'TASK COMPLETE' in m['text'].upper())} 'complete' signals")

    print("\n  AutoGen's real value: agents can run tools (code execution, web search,")
    print("  file I/O) and route via a GroupChatManager without message-boilerplate code.")

    print("\n--- library cross-check (pyautogen / autogenhub Python) ---")
