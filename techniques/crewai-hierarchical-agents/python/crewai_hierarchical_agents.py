"""CrewAI - Hierarchical Multi-Agent (Reference Sec 47.208).

Moura 2024 (CrewAI OSS). Structures multi-agent workflows around a
CREW of specialised agents, a set of TASKS with declared inputs/
outputs, and either SEQUENTIAL or HIERARCHICAL process (where a
manager agent routes tasks).

    Agent(role, goal, backstory, tools)
    Task(description, agent, expected_output, context=[prev_tasks])
    Crew(agents, tasks, process="sequential" | "hierarchical")

Contrasts with AutoGen's turn-taking chat: CrewAI's tasks have
explicit DAG dependencies and typed outputs.
"""
from __future__ import annotations    # stdlib


class CrewAgent:
    def __init__(self, role, goal, backstory, respond):
        self.role = role; self.goal = goal; self.backstory = backstory
        self.respond = respond


class CrewTask:
    def __init__(self, description, agent, expected_output, context=None):
        self.description = description; self.agent = agent
        self.expected_output = expected_output; self.context = context or []


class Crew:
    def __init__(self, agents, tasks, process="sequential", manager=None):
        self.agents = agents; self.tasks = tasks
        self.process = process; self.manager = manager

    def kickoff(self, verbose=True):
        results = {}
        for i, t in enumerate(self.tasks):
            # Assemble context from dependency tasks
            ctx = "\n".join(results[c] for c in t.context if c in results)
            if self.process == "hierarchical" and self.manager:
                agent = self.manager.respond(
                    {"task": t.description, "expected": t.expected_output})
                agent = self.agents.get(agent, list(self.agents.values())[0])
            else:
                agent = t.agent
            output = agent.respond({"task": t.description, "context": ctx,
                                       "expected": t.expected_output})
            results[t.description] = output
            if verbose:
                print(f"  Task {i + 1}: {t.description}")
                print(f"    Assigned to: {agent.role}")
                print(f"    Output: {output}\n")
        return results


def researcher_respond(msg):
    return "Facts collected: Python was created by Guido van Rossum in 1991; " \
             "widely used for scripting, ML, and web backends."


def writer_respond(msg):
    ctx = msg.get("context", "")
    if "1991" in ctx:
        return "Draft article: 'Since 1991, Python has grown into the leading " \
                 "language for data science and rapid application development.'"
    return "Draft article: [no facts available]"


def editor_respond(msg):
    ctx = msg.get("context", "")
    if "1991" in ctx and "Python" in ctx:
        return "Final: 'Python, created by Guido van Rossum in 1991, dominates " \
                 "modern data science and web-backend work.'"
    return "Final: [insufficient input]"


def manager_route(msg):
    """Manager reads the task description and picks the specialist role."""
    desc = msg["task"].lower()
    if "research" in desc or "fact" in desc: return "researcher"
    if "write" in desc or "draft" in desc: return "writer"
    return "editor"


if __name__ == "__main__":
    print("=== CrewAI - Hierarchical Multi-Agent (Moura 2024) ===\n")

    researcher = CrewAgent("researcher", "Collect facts", "Historian.", researcher_respond)
    writer = CrewAgent("writer", "Draft article", "Journalist.", writer_respond)
    editor = CrewAgent("editor", "Polish prose", "Editor.", editor_respond)
    agents = {"researcher": researcher, "writer": writer, "editor": editor}

    tasks = [
        CrewTask("Research Python's history", researcher, "list of facts"),
        CrewTask("Write a short article", writer, "draft article",
                    context=["Research Python's history"]),
        CrewTask("Edit for tone", editor, "final article",
                    context=["Write a short article"]),
    ]

    print("  --- Sequential process (fixed pipeline) ---\n")
    crew_seq = Crew(agents, tasks, process="sequential")
    crew_seq.kickoff()

    print("\n  --- Hierarchical process (manager routes) ---\n")
    manager = CrewAgent("manager", "Route tasks", "Coordinator.", manager_route)
    crew_hier = Crew(agents, tasks, process="hierarchical", manager=manager)
    crew_hier.kickoff()

    print("--- library cross-check (crewai Python) ---")
