"""ReAct - Reasoning and Acting (Reference Sec 47.202).

Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao 2023 'ReAct:
Synergizing Reasoning and Acting in Language Models', ICLR.
Interleaves REASONING traces with ACTIONS in a tool-using LM:

    Thought: "I need to look up the founding year..."
    Action: search[Elon Musk company founded 2002]
    Observation: SpaceX was founded in 2002...
    Thought: "Confirmed. So the answer is..."
    Action: finish[2002]

Combines the strengths of chain-of-thought (structured reasoning)
and tool use (external knowledge + actions on the world).
"""
from __future__ import annotations    # stdlib

import re    # regex for parsing actions


class ToyEnv:
    """Wikipedia-search + calculator toy environment."""

    def __init__(self, docs):
        self.docs = docs

    def act(self, action):
        m = re.match(r"(search|calc|finish)\[(.*)\]", action)
        if not m: return "unknown-action"
        kind, arg = m.group(1), m.group(2)
        if kind == "search":
            for d in self.docs:
                if arg.lower() in d.lower():
                    return d
            return "no results"
        if kind == "calc":
            try: return str(eval(arg))
            except Exception: return "calc-error"
        return f"DONE({arg})"


def react_toy_lm(question, env, max_steps=6, verbose=True):
    """Toy 'LM' that parses the question, decides on tool calls, and finishes."""
    trace = []
    q = question.lower()
    # Math: extract the arithmetic-only chunk (numbers + ops + spaces)
    if any(w in q for w in ["+", "-", "*", "/"]):
        expr_matches = [m.strip() for m in re.findall(r"[\d]+(?:\s*[+\-*/]\s*[\d]+)+", question)]
        if expr_matches:
            trace.append(("Thought", "This is a math question. Use calculator."))
            trace.append(("Action", f"calc[{expr_matches[0]}]"))
            obs = env.act(trace[-1][1])
            trace.append(("Observation", obs))
            trace.append(("Action", f"finish[{obs}]"))
            return trace, obs
    # Lookup question: skip common sentence-start words
    stopwords = {"In", "What", "When", "Where", "Who", "How", "Why", "The", "A", "An"}
    keyword = [w for w in re.findall(r"[A-Z][a-zA-Z]+", question) if w not in stopwords]
    if keyword:
        trace.append(("Thought", "This is a factual question. Search for entity."))
        trace.append(("Action", f"search[{keyword[0]}]"))
        obs = env.act(trace[-1][1])
        trace.append(("Observation", obs))
        # Extract year / number
        yrs = re.findall(r"\b(1[89]\d{2}|20\d{2})\b", obs)
        if yrs:
            trace.append(("Thought", f"Found year {yrs[0]} in the document."))
            trace.append(("Action", f"finish[{yrs[0]}]"))
            return trace, yrs[0]
    trace.append(("Action", "finish[unknown]"))
    return trace, "unknown"


def cot_baseline(question):
    """Chain-of-thought without tools: just try to guess."""
    return "unknown"                                             # LM alone can't look up


if __name__ == "__main__":
    print("=== ReAct - Reasoning and Acting (Yao et al 2023 ICLR) ===\n")

    docs = [
        "Space Exploration Technologies (SpaceX) was founded by Elon Musk in 2002.",
        "Tesla Motors was founded in 2003 by Martin Eberhard and Marc Tarpenning.",
        "The Boring Company was founded in 2016 by Elon Musk.",
    ]
    env = ToyEnv(docs)

    questions = [
        "In what year was SpaceX founded?",
        "In what year was Tesla founded?",
        "What is 7 * 6 + 3?",
    ]
    truths = ["2002", "2003", "45"]

    react_hits = 0; cot_hits = 0
    for q, gt in zip(questions, truths):
        trace, ans = react_toy_lm(q, env)
        cot_ans = cot_baseline(q)
        react_hits += (ans == gt); cot_hits += (cot_ans == gt)
        print(f"\n  Q: {q}")
        for role, txt in trace:
            print(f"    {role}: {txt}")
        print(f"    ReAct answer: {ans}   CoT-only baseline: {cot_ans}   truth: {gt}")

    print(f"\n  Answered correctly: ReAct {react_hits}/{len(questions)}   "
          f"CoT-only baseline {cot_hits}/{len(questions)}")

    print("\n--- library cross-check (langchain ReAct agent, llama-index ReActAgent) ---")
