import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel

# Import your tools
from chatbot_app.tools.calculator import calculate
from chatbot_app.tools.rag_placeholder import zus_info_retriever

# Load .env
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
            ),
            Tool(
                name="ZUSInfoRetriever",
                func=zus_info_retriever,
                description="Use this to answer questions about ZUS Coffee outlets and products."
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
Your job is to help users by answering their questions or solving tasks. You can use tools to do this, but ONLY WHEN NEEDED.

**Before using a tool**, ask yourself:
- Is the question clear and specific?
- Do I have all the info I need?
- If not, ask the user a clarifying (follow-up) question first.

You must be 100% sure about the user's intent (e.g., what specific product, location, or information they want), NEVER use a tool immediately.

Instead, ask a clear follow-up question first to clarify what they’re looking for.

If you clearly understand the user's question and know that a tool is needed, use this format:

Thought: Explain what you're doing.
Action: <ToolName>
Action Input: <specific input to the tool>

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

    def chat(self, user_input: str) -> str:
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response["output"].strip()
        except Exception as e:
            print(f"Error during chat: {e}")
            return "Sorry, something went wrong. Try again."


# CLI entrypoint
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Missing GOOGLE_API_KEY in .env.")
        exit()

    print("Mindhive Chatbot (LangChain v0.3.26) initialized.")
    print("Type 'exit' to quit.")
    print("Try asking questions like:")
    print("- 'What is 123+45?'")
    print("- 'Tell me about ZUS Coffee outlets.'")
    print("- 'What kind of products does ZUS Coffee offer?'")
    print("- 'Is there an outlet in Petaling Jaya?'")
    chatbot = MindhiveChatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
