import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from chatbot_app.tools.calculator import calculate # Your calculator function

load_dotenv()

# Tool wrapper
calculator_tool = Tool(
    name="Calculator",
    func=calculate,
    description="Useful for evaluating basic arithmetic expressions like addition, subtraction, multiplication, and division."
)

# System prompt focused on math only
prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers arithmetic questions using a calculator tool.

Rules:
- If the user's input is clearly a math expression, use the Calculator tool.
- If it's not valid math, respond gracefully and explain the error.

Use this format:
Thought: Do I need to use the calculator?
Action: Calculator
Action Input: <expression>

Observation: <result>
Final Answer: <response to user>
""")

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# Agent initialization
agent = initialize_agent(
    tools=[calculator_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

def run_calculator_agent(user_input: str) -> str:
    try:
        return agent.run(user_input)
    except Exception as e:
        return f"Sorry, something went wrong: {e}"

# CLI testing
if __name__ == "__main__":
    print("Calculator Agent - Part 3. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = run_calculator_agent(user_input)
        print(f"Bot: {response}")
