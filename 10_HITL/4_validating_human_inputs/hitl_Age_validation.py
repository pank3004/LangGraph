import sqlite3
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver


# -----------------------
# 1. State
# -----------------------
class LoanFormState(TypedDict):
    age: int | None


# -----------------------
# 2. HITL Validation Node
# -----------------------
def collect_age_node(state: LoanFormState):
    """
    Collect and validate age from a human.
    Re-prompts until valid.
    """

    prompt = "Please enter your age (18–65):"

    while True:
        answer = interrupt(prompt)

        # Validation rules
        if isinstance(answer, int) and 18 <= answer <= 65:
            return {"age": answer}

        prompt = (
            f"'{answer}' is invalid. "
            "Age must be a number between 18 and 65. Try again:"
        )


# -----------------------
# 3. Graph
# -----------------------
builder = StateGraph(LoanFormState)

builder.add_node("collect_age", collect_age_node)

builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

graph = builder.compile(
    checkpointer=SqliteSaver(sqlite3.connect("loan_form.db", check_same_thread=False))
)


# -----------------------
# 4. Run (CLI Simulation)
# -----------------------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "loan-form-001"}}

    # First run → ask for age
    result = graph.invoke({"age": None}, config=config)
    print("INTERRUPT:", result["__interrupt__"][0].value)

    # User enters invalid value
    result = graph.invoke(Command(resume="seventeen"), config=config)
    print("INTERRUPT:", result["__interrupt__"][0].value)

    # User enters valid value
    result = graph.invoke(Command(resume=30), config=config)
    print("\n✅ FINAL STATE")
    print("Approved Age:", result["age"])
