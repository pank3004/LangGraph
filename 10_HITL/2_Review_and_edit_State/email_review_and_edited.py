from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_cerebras import ChatCerebras
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()
# -----------------------------
# 1. State
# -----------------------------
class ReviewState(TypedDict):
    topic: str
    generated_text: str


# -----------------------------
# 2. LLM
# -----------------------------
llm=ChatCerebras(model='llama-3.3-70b')

# -----------------------------
# 3. LLM Draft Node
# -----------------------------
def generate_draft(state: ReviewState) -> ReviewState:
    prompt = f"Write a professional email about: {state['topic']}"
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "generated_text": response.content,
    }


# -----------------------------
# 4. HITL Review Node (EDIT STATE)
# -----------------------------
def review_node(state: ReviewState) -> ReviewState:
    """
    Human edits the generated content.
    Whatever the human returns becomes the new state.
    """
    edited_text = interrupt({
        "instruction": "Review and edit the email if needed",
        "draft": state["generated_text"],
    })

    return {
        **state,
        "generated_text": edited_text,
    }


# -----------------------------
# 5. Build Graph
# -----------------------------
builder = StateGraph(ReviewState)

builder.add_node("generate", generate_draft)
builder.add_node("review", review_node)

builder.add_edge(START, "generate")
builder.add_edge("generate", "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=MemorySaver())


# -----------------------------
# 6. Run (CLI Simulation)
# -----------------------------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "email-review-1"}}

    # Step 1: LLM generates draft → graph pauses
    result = graph.invoke(
        {"topic": "Requesting leave for next week"},
        config=config,
    )

    interrupt_data = result["__interrupt__"][0].value
    print("\n--- HUMAN REVIEW REQUIRED ---")
    print(interrupt_data["draft"])

    # Simulate human editing the draft
    edited = (
        "Dear Manager,\n\n"
        "I would like to request leave for two days next week due to personal reasons.\n"
        "Please let me know if this works.\n\n"
        "Regards,\n"
        "Pankaj"
    )

    # Step 2: Resume graph with edited content
    final_state = graph.invoke(
        Command(resume=edited),
        config=config,
    )

    print("\n✅ FINAL APPROVED EMAIL\n")
    print(final_state["generated_text"])
