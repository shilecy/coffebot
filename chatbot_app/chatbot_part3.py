import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel

# Import your tools
from chatbot_app.tools.calculator import calculate # Your calculator function

load_dotenv()

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

INSTRUCTIONS:
-------------
You are a helpful assistant that answers arithmetic questions using a calculator tool.

Rules:
- If the user's input is clearly a math expression, use the Calculator tool.
  - **Before using a tool**, ask yourself:
     - Is the question clear and specific?
     - Do I have all the info I need?
     - If not, ask the user a clarifying (follow-up) question first.
- If it's not valid math, respond gracefully and explain the error.

Use this format:
Thought: Explain what you're doing.
Action: <ToolName>
Action Input: <specific input to the tool>

Observation: <result>
Final Answer: <response to user>

When you get the tool’s result, respond like this:

Observation: <what you saw from the tool>
Thought: Do I now have the answer?
Final Answer: <your response to the user>

Do **not** use a tool again unless absolutely necessary. Stop when you’re confident with the answer.

EXAMPLE:

Thought: The user wants to calculate 123 + 45.
Action: Calculator
Action Input: 123 + 45

Observation: 168
Thought: I now know the final answer.
Final Answer: 123 + 45 is 168.

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
            handle_parsing_errors=True,
            agent_kwargs={
                "tool_names": tool_names,
                "tools": tool_descriptions
            }
        )

    def chat_3(self, user_input: str) -> str:
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response["output"].strip()
        except Exception as e:
            print(f"Error during chat: {e}")
            return "Sorry, something went wrong. Try again."

# CLI testing
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Missing GOOGLE_API_KEY in .env.")
        exit()

    print("Calculator Agent - Part 3. Type 'exit' to quit.")
    chatbot = MindhiveChatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.chat_3(user_input)
        print(f"Bot: {response}")
