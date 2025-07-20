import pytest
from chatbot_app.tools.calculator import calculate
from chatbot_app.tools.rag_placeholder import zus_info_retriever

# --- Calculator Tests ---
def test_calculator_addition():
    result = calculate("2 + 3")
    assert result == "5"

def test_calculator_complex_expression():
    result = calculate("10 * (5 + 2)")
    assert result == "70"

def test_calculator_invalid_expression():
    result = calculate("10 / (5 - 5)")
    assert "error" in result.lower()  # Updated assertion

# --- RAG Placeholder Tests ---
def test_zus_retriever_outlet_query():
    response = zus_info_retriever("Is there an outlet in PJ?")
    assert "outlet" in response.lower()

def test_zus_retriever_product_query():
    response = zus_info_retriever("Tell me about your frappes")
    assert "looking up" in response.lower() and "zus coffee" in response.lower()  # Updated assertion

def test_zus_retriever_unknown_query():
    response = zus_info_retriever("What is ZUS Coffee's philosophy?")
    assert "zus coffee information" in response.lower()

