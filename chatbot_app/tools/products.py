from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List
from app.rag import semantic_search, summarize_results


class ProductTool(BaseModel):
    query: str = Field(..., description="The user's question about ZUS Coffee products.")

@tool("product_search_tool", args_schema=ProductTool)
def rag_tool(query: str) -> str:
    """
    Use this to answer questions about ZUS Coffee products such as material (e.g. BPA-free, stainless steel), volume, product types (tumblers, mugs), price, variations, measurements, etc.
    """
    try:
        results = semantic_search(query)
        summary = summarize_results(query, results)
        return summary
    except Exception:
        return "Sorry, the ZUS server is currently unavailable. Please try again later."


