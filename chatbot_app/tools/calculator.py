# calculator.py
from pydantic import BaseModel, Field
from langchain.tools import tool

class CalculatorTool(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

@tool("Calculator", args_schema=CalculatorTool)
def calculate(expression: str) -> str:
    """Evaluates a math expression like '2 + 3'."""
    if not expression or expression.strip() == "":
        return "Error: No expression provided. Please enter a valid mathematical expression."

    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: Could not evaluate the expression. {e}"
