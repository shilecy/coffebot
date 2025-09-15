import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel

import requests

# Import your tools
from chatbot_app.tools.calculator import calculate
from chatbot_app.tools.products import rag_tool
from chatbot_app.tools.outlets import outlet_tool

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"

class MindhiveChatbot:
    def __init__(self, llm: BaseChatModel = None, memory_obj: ConversationBufferMemory = None):
        # Init LLM
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

        # Init Memory
        self.memory = memory_obj or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False  # For string-based prompts (PromptTemplate)
        )

        # Define tools
        self.tools = [
            Tool(
                name="Calculator",
                func=calculate,
                description="Use this to perform math or arithmetic calculations."
            ),
            Tool(
                name="ProductInfo",
                func=rag_tool,
                description="Use this to answer questions about ZUS Coffee products."
            ),
            Tool(
                name="OutletInfo",
                func=outlet_tool,
                description="Use this to answer questions about ZUS Coffee outlets."
            )
        ]

        # Tool metadata
        tool_names = ", ".join([tool.name for tool in self.tools])
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # PromptTemplate (string-based)
        self.prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "chat_history", "tools", "tool_names"],
            template="""
You are a helpful and knowledgeable assistant for ZUS Coffee.

TOOLS:
------
You can use the following tools:
{tools}

TOOL NAMES:
-----------
You can refer to the tools by their names:
{tool_names}

FORMAT INSTRUCTIONS (STRICTLY FOLLOW):
-------------
Your job is to help users by answering their questions or solving tasks. You can use tools to do this, but ONLY WHEN NEEDED.

Use the following tools when appropriate:
- Calculator: for math calculations.
- product_search_tool: for product-related queries (e.g. "What tumblers are BPA free?", "What is the height of ZUS drinkware?").
- outlet_search_tool: for outlet-related queries (e.g. "Where is the Shah Alam outlet?", "Which outlet opens 24 hours?", "How many outlets in Kuala Lumpur?").

**Before using a tool**, ask yourself:
- Is the question clear and specific?
- Do I have all the info I need?
- If not, ask the user a clarifying (follow-up) question first.

You must be 100% sure about the user's intent (e.g., what specific product, location, or information they want), NEVER use a tool immediately.

Instead, ask a clear follow-up question first to clarify what they’re looking for.

IF you need to ask a clarifying or follow-up question:
Always use this format:
Thought: I need more information to proceed.
Final Answer: <your clarifying question here>

EXAMPLES:
---------
Thought: The user asked "What are the opening hours?" but did not specify the outlet.
Final Answer: Which outlet are you referring to? Please provide the outlet name or location.
---------

DO NOT use a tool until the user's intent is fully understood. Ask a follow-up question first if anything is ambiguous.

When you understand the question AND a tool is needed:
Use this format:
Thought: <what you are thinking>
Action: <ToolName>
Action Input: <input string to send to tool>

When you receive the tool output:
Observation: <tool response>
Thought: Do I now have the answer?
Final Answer: <answer to user>

Do **not** use a tool again unless absolutely necessary. Stop when you’re confident with the answer.
Avoid excessive thinking loops. Ask follow-up questions early when necessary.

EXAMPLES:
---------
Thought: The user wants to calculate 123 + 45.
Action: Calculator
Action Input: 123 + 45

Observation: 168
Thought: I now know the final answer.
Final Answer: 123 + 45 is 168.

Thought: The user asked "What is 2 + 3?" so this is also a math question.
Action: Calculator
Action Input: 2 + 3

Thought: The user asked "Is there any outlet in Shah Alam?" which means they want outlet info.
Action: OutletInfo
Action Input: Is there any outlet in Shah Alam?

Thought: The user asked "Is there any outlet in Shah Alam?" which means they want outlet info.
Action: OutletInfo
Action Input: Is there any outlet in Shah Alam?

Thought: The user asked "Do you sell mugs or tumblers?" which means they are asking about product categories.
Action: ProductInfo
Action Input: mugs or tumblers

----

Previous conversation:
{chat_history}

User question: {input}

{agent_scratchpad}
"""
        )

        # Agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # AgentExecutor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True, 
            handle_parsing_errors=True,
            max_iterations=3,
            agent_kwargs={
                "tool_names": tool_names,
                "tools": tool_descriptions
            }
        )

        # Memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False
        )

    def chat_4(self, user_input: str) -> str:
        try:
            response = self.agent_executor.invoke({"input": user_input})
            output = response.get("output", "").strip()

        # If final output is empty or contains common agent failure messages
            if not output or "Agent stopped" in output or "try again" in output.lower():
                steps = response.get("intermediate_steps", [])
                # Loop backwards through steps to find last meaningful observation
                for action, observation in reversed(steps):
                    if isinstance(observation, str) and len(observation.strip()) > 30:
                        return observation.strip()
                return output or "Sorry, the agent couldn't complete the task."
        
            return output

        except Exception as e:
            print(f"Error during chat: {e}")
            return "Sorry, something went wrong. Try again."

# CLI testing
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Missing GOOGLE_API_KEY in .env.")
        exit()

    print("ZUS Coffee Chatbot (Gemini 2.5 Flash, Agentic)")
    print("Type 'exit' to quit.")
    chatbot = MindhiveChatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot.chat_4(user_input)
        print(f"Bot: {response}")