import pytest
from chatbot_app.tools.outlets import outlet_tool

def test_outlet_tool_missing_query():
    # Simulate user giving an empty query
    response = outlet_tool.invoke({"query": ""})

    # Make sure the tool responds with a helpful error message
    assert "no query provided" in response.lower()

from chatbot_app.tools.calculator import calculate

def test_calculator_tool_missing_expression():
    # Simulate user giving no expression
    response = calculate.invoke({"expression": ""})

    # Should return a helpful error message, not crash
    assert "no expression provided" in response.lower()


from unittest.mock import patch
from chatbot_app.tools.products import rag_tool, ProductTool

def test_product_tool_api_downtime():
    with patch("chatbot_app.tools.products.semantic_search") as mock_search:
        mock_search.side_effect = Exception("Simulated FAISS/VectorStore downtime")

        response = rag_tool.invoke({"query": "Tell me about stainless steel bottles"})
        assert "ZUS server is currently unavailable" in response

from chatbot_app.tools.outlets import outlet_tool, OutletTool

def test_outlet_tool_malicious_input():
    # Simulated SQL injection-like input
    response = outlet_tool.invoke({"query": "'; DROP TABLE outlets; --"})

    # Check if it failed gracefully with a safe message
    assert (
        "sorry" in response.lower()
        or "couldn't" in response.lower()
        or "suspicious" in response.lower()
    )


