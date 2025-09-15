# app/main.py
from fastapi import FastAPI, Query, APIRouter
from pydantic import BaseModel
from app.rag import semantic_search, summarize_results
from chatbot_app.chatbot_part4 import chat_4
from app.text2sql_outlets import query_outlets_from_db
from app.calculator_logic import calculate_expression
from dotenv import load_dotenv
import asyncio

load_dotenv()
app = FastAPI()

# Chatbot endpoint

class ChatRequest(BaseModel):
    question: str

@app.post("/chatbot")
async def chatbot_route(req: ChatRequest):
    try:
        # Set a timeout for chatbot logic (e.g., 15 seconds)
        answer = await asyncio.to_thread(chat_4, req.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# Products endpoint

@app.get("/products")
def query_products(query: str = Query(..., min_length=3)):
    results = semantic_search(query, top_k=3)
    summary = summarize_results(query, results)

    return {
        "query": query,
        "summary": summary,
        "results": results
    }

# Outlets endpoint

class QueryRequest(BaseModel):
    question: str

@app.post("/outlets")
def query_outlets(request: QueryRequest):
    return query_outlets_from_db(request.question)

# Calculator endpoint

class CalcRequest(BaseModel):
    expression: str

@app.post("/calculator")
def calculator_endpoint(req: CalcRequest):
    return calculate_expression(req.expression)