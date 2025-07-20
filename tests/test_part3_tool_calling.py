import pytest
from chatbot_app.tools.calculator import calculate
from chatbot_app.chatbot_part3 import run_calculator_agent

def test_calculator_simple_addition():
    result = run_calculator_agent("2 + 3")
    assert "5" in result

def test_calculator_nested_expression():
    result = run_calculator_agent("2 + (3 * 4)")
    assert "14" in result

def test_calculator_invalid_expression():
    result = run_calculator_agent("2 + cat")
    lowered = result.lower()
    assert any(keyword in lowered for keyword in [
        "non-numeric", "invalid", "not a number", "error", 
        "unable to", "couldn’t", "not a valid", "mathematical expression"
    ])

def test_calculator_divide_by_zero():
    result = run_calculator_agent("10 / (5 - 5)")
    lowered = result.lower()
    assert any(keyword in lowered for keyword in [
        "division by zero", "undefined", "error", "cannot divide", "couldn’t"
    ])

