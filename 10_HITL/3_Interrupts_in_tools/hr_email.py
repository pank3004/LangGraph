from typing import TypedDict
import sqlite3

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_cerebras import ChatCerebras

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from dotenv import load_dotenv
load_dotenv()


# -----------------------
# 1. State
# -----------------------
class AgentState(TypedDict):
    messages: list


# -----------------------
# 2. HITL Tool (REAL USE)
# -----------------------
@tool
def send_hr_email(to: str, subject: str, body: str):
    """
    Send an HR email (requires human approval).
    """

    approval = interrupt({
        "action": "send_hr_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this HR email?",
    })

    # Human cancelled
    if not isinstance(approval, dict) or approval.get("action") != "approve":
        return "❌ Email cancelled by HR"

    # Human may edit content
    final_subject = approval.get("subject", subject)
    final_body = approval.get("body", body)

    # Simulate sending email
    print("\n📧 EMAIL SENT")
    print("To:", to)
    print("Subject:", final_subject)
    print("Body:", final_body)

    return f"✅ HR email sent to {to}"


# -----------------------
# 3. Model
# -----------------------
llm = ChatCerebras(
    model="llama-3.3-70b"
).bind_tools([send_hr_email])


# -----------------------
# 4. Agent Node
# -----------------------
def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# -----------------------
# 5. Graph
# -----------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode([send_hr_email]))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(
    checkpointer=SqliteSaver(
        sqlite3.connect("hr-email.db", check_same_thread=False)
    )
)


# -----------------------
# 6. Run (CLI Simulation)
# -----------------------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hr-approval-flow"}}

    # Step 1: AI decides to send email → INTERRUPT
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Send an HR email to john@company.com about leave approval"
                )
            ]
        },
        config=config,
    )

    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        print("❌ No approval requested")
        exit()

    approval_request = interrupts[0].value

    print("\n🧑‍⚖️ HUMAN REVIEW")
    print("To:", approval_request["to"])
    print("Subject:", approval_request["subject"])
    print("Body:", approval_request["body"])

    # Simulate HR editing content
    approval_response = {
        "action": "approve",
        "subject": "Leave Approved ✅",
        "body": (
            "Dear John,\n\n"
            "Your leave has been approved.\n\n"
            "Regards,\nHR Team"
        ),
    }

    # Step 2: Resume execution
    final = graph.invoke(
        Command(resume=approval_response),
        config=config,
    )

    print("\n🤖 AI FINAL MESSAGE")
    print(final["messages"][-1].content)
