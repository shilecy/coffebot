from langchain.tools import tool
from pydantic import BaseModel, Field
import logging
from typing import Optional

from app.llm_sql_generator import generate_sql_query, extract_sql_codeblock
from app.text2sql_outlets import query_outlets_from_db

logger = logging.getLogger(__name__)

class OutletTool(BaseModel):
    query: str = Field(..., description="The user's question about ZUS Coffee outlets.")

@tool("outlet_search_tool", args_schema=OutletTool)
def outlet_tool(query: str) -> str:
    """Uses LLM to convert natural language into SQL and returns query results from the outlets database like location, opening hours, services."""
    try:
        if not query or query.strip() == "":
            return "Error: No query provided. Please ask something like 'Show all outlets in Selangor'."

        logger.debug(f"Received user query: {query}")
        
                # 🛡️ Basic protection against malicious input
        if any(kw in query.lower() for kw in ["drop table", "--", ";", "'"]):
            return "Your query looks suspicious. Please ask about outlets using natural language."


        # Step 1: Generate SQL from natural language
        sql_raw = generate_sql_query(query)
        sql_clean = extract_sql_codeblock(sql_raw)
        logger.debug(f"Generated SQL: {sql_clean}")

        # Step 2: Query SQLite database
        result = query_outlets_from_db(sql_clean)
        logger.debug(f"Raw result from query_outlets_from_db: {result}")

        if not result or not result.get("result"):
            return "I couldn't find any information matching your query."

        rows = result["result"]

        # Add this line to debug
        print("DEBUG - Raw result from query_outlets_from_db:", result)

        # Step 3: Format output
        if "COUNT(" in sql_clean.upper():
            count_value = list(rows[0].values())[0]
            outlet_word = "outlet" if count_value == 1 else "outlets"
            return f"There {'is' if count_value == 1 else 'are'} {count_value} ZUS Coffee {outlet_word} matching your query."

        formatted_rows = []
        for row in rows:
            outlet_info = ', '.join(f"{k}: {v}" for k, v in row.items())
            formatted_rows.append(outlet_info)

        return "\n\n".join(formatted_rows)

    except Exception as e:
        logger.exception("Error while processing outlet search.")
        return f"Sorry, something went wrong while processing your request. Details: {e}"

