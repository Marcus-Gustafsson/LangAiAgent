"""
Simple Test Agent with MCP + LangChain

⚡️ Purpose:
This script is meant as a *learning/testing agent* — a minimal example of
how to connect:
- An LLM (OpenAI via LangChain)
- An MCP server (Firecrawl in this case)
- A simple REPL loop where the user types questions and the agent answers.

The idea is to experiment and understand the building blocks before 
attempting to build a larger/more advanced AI agent.

Key Concepts:
- MCP (Model Context Protocol) lets external tools (like Firecrawl) be
  connected to an AI model so the model can use them.
- Firecrawl MCP Server is a Node.js process started in the background that
  provides web scraping/crawling tools for the agent.
- LangChain is used to wire the tools + LLM into a "ReAct agent", which
  decides step by step when to call tools and when to answer.
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os
from pathlib import Path


# --- Load Environment Variables ---------------------------------------------
# We keep secrets (like API keys) in a .env file at the project root.
# Using an absolute path avoids issues with relative paths when run from
# different working directories.
ROOT = Path(__file__).resolve().parent  # this file's folder (simple-agent/)
ENV_PATH = ROOT.parent / ".env"         # points to project/.env
load_dotenv(dotenv_path=ENV_PATH)

# --- Setup LLM --------------------------------------------------------------
# Create a ChatOpenAI model wrapper (LangChain handles the integration).
# "gpt-5-nano" is used here for cheap testing.
# Temperature=0 → deterministic answers (no randomness).
model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


# --- Setup MCP Server (Firecrawl) -------------------------------------------
# The Firecrawl MCP server is an external Node.js process that exposes
# crawling/scraping tools. We tell MCP how to launch it:
server_params = StdioServerParameters(
    command="npx",  # runs Node package executables
    args=["firecrawl-mcp"],  # tells npx which MCP package to start
    env={
        # Firecrawl requires its own API key
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
)


# --- Main Async Loop --------------------------------------------------------
async def main():
    """
    Main loop of the agent:
    - Connects to MCP server
    - Loads available tools
    - Creates a LangChain ReAct agent (LLM + tools)
    - Lets the user chat with the agent in a REPL
    """

    # Connect to the Firecrawl MCP server using stdio (stdin/stdout pipes)
    async with stdio_client(server_params) as (read, write):

        # Session manages communication with MCP server
        async with ClientSession(read, write) as session:
            await session.initialize()  # handshake with MCP server

            # Load available MCP tools (here: Firecrawl scraping tools)
            tools = await load_mcp_tools(session)

            # Create a LangChain ReAct agent that can:
            # - reason step by step
            # - call tools when needed
            # - return final answers
            agent = create_react_agent(model, tools)

            # System prompt → sets the role and instructions for the agent
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that can scrape websites, "
                        "crawl pages, and extract data using Firecrawl tools. "
                        "Think step by step and use the appropriate tools."
                    ),
                }
            ]

            # Show the tools available from MCP
            print("Available Tools in MCP:", *[tool.name for tool in tools])
            print("-" * 60)

            # Simple REPL loop: user types → agent responds
            while True:
                user_input = input("\nYou: ")

                # Exit condition
                if user_input in {"quit", "q"}:
                    print("Goodbye!")
                    return

                # Add user message to conversation history
                # Note: user input is capped at 175k characters
                messages.append(
                    {"role": "user", "content": user_input[:175000]}
                )

                try:
                    # Asynchronously invoke the agent
                    agent_response = await agent.ainvoke({"messages": messages})

                    # Get the final agent message (last one in the list)
                    ai_message = agent_response["messages"][-1].content

                    print("\nAgent:", ai_message)

                except Exception as err:
                    print("Error:", err)


# --- Script Entry Point -----------------------------------------------------
if __name__ == "__main__":
    # asyncio.run launches the async main() loop.
    asyncio.run(main())
