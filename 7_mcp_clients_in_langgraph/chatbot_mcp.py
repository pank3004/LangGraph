from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatCerebras(model="llama-3.3-70b")

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
    "Arithmetic MCP Server": {
      "transport":"stdio",
      "command": "C:\\Users\\panka\\anaconda3\\Scripts\\uv.exe",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "C:\\Users\\panka\\OneDrive\\Desktop\\local mcp maths server\\main.py"
      ]
    },
        # "expense": {
        #     "transport": "streamable_http",  # if this fails, try "sse"
        #     "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        # }
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="add 78 in 687")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())