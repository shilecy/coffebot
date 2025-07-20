from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)

PROMPT_TEMPLATE = """
You are an expert SQL generator for an SQLite database.

Generate an SQL query for the table `outlets` with the following columns:
- id (integer)
- name (text)
- address (text)
- hours (text)
- services (text)

Guidelines:
1. If the user asks for the number of outlets or uses phrases like:
   - "how many", "total number", "count", etc.
   Then use: SELECT COUNT(*) FROM outlets WHERE ...

2. For location-based or fuzzy queries (e.g. "in Shah Alam", "around Setia Alam"):
   Use `address LIKE '%...%'` with appropriate casing or wildcards.

3. Do not include the `id` column in SELECT unless asked.

4. Return the SQL query.

Now, generate an SQL query for the following user question:

Question: {question}
SQL:
"""

prompt = PromptTemplate(input_variables=["question"], template=PROMPT_TEMPLATE)

# Main generator function
def generate_sql_query(question: str) -> str:
    chain = prompt | llm
    ai_message = chain.invoke({"question": question})
    text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
    sql = extract_sql_codeblock(text)
    return sql

import re

def extract_sql_codeblock(text: str) -> str:
    # Extract SQL inside code block if it exists
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()
