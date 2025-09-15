This document provides a comprehensive overview of the Mindhive AI Chatbot Engineer Assessment project. This multi-tool AI chatbot, built with FastAPI, LangChain, and Google Gemini, demonstrates various functionalities, including a Calculator tool, a Product Knowledge Base (RAG), and Outlet Info (Text-to-SQL).

-----

## Requirements

To set up and run this project, ensure you have the following:

  * **Python:** Version 3.10.10
  * **Virtual Environment:** `virtualenv` or `venv`
  * **Google Generative AI API Key:** For Gemini integration

-----

## Getting Started

Follow these steps to get the chatbot up and running:

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/mindhive-chatbot-assessment.git
cd mindhive-chatbot-assessment
```

### 2\. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Setup Environment Variables

Create a `.env` file in the root directory and add your Gemini API key:

```
GOOGLE_API_KEY=your_google_gemini_api_key
```

For Streamlit Cloud, set the same key in Secrets via the dashboard UI.

### 5\. Run the Application

You have several options to run the application:

#### Option A: Run API Server (FastAPI)

```bash
python -m uvicorn app.main:app --reload
```

  * **API available at:** `http://localhost:8000`
  * **Swagger UI (API Docs):** `http://localhost:8000/docs`

#### Option B: Run Local Chatbot Demo (Streamlit)

```bash
streamlit run streamlit_demo.py
```

  * Opens at: `http://localhost:8501`

#### Option C: Deploy to Streamlit Cloud

1.  Push your code to GitHub.
2.  Visit `https://streamlit.io/cloud`.
3.  Select your repository, configure secrets, and set the main file to `streamlit_demo.py`.

#### Option D: Run Individual Chatbot Parts (via Terminal)

Each chatbot part can be run directly for testing purposes. Example commands:

```bash
python -m chatbot_app.chatbot_part1
python -m chatbot_app.chatbot_part2
python -m chatbot_app.chatbot_part3
python -m chatbot_app.chatbot_part4
```

Each script simulates a sample interaction or uses test inputs to demonstrate its specific feature (e.g., memory, planning, tool use, RAG/text2sql).

-----

## Architecture Overview

This chatbot system is designed with modularity and extensibility in mind, implemented in Python using LangChain and FastAPI. It is structured around five core parts, each demonstrating a specific capability:

```
├── streamlit_demo.py       # Unified chatbot demo interface
├── chatbot_app/
│   └── __pycache__
│   └── __init__.py
│   └── chatbot_part1.py    # Sequential conversation with memory
│   └── chatbot_part2.py    # Agentic planning logic (calculator + rag placeholder)
│   └── chatbot_part3.py    # Tool integration (calculator)
│   └── chatbot_part4.py    # Full agent integration (Calculator, Retrieval-augmented generation (RAG) (product) + text-to-SQL (outlets))
│   └── streamlit_demo.py
│   └── tools/
│   │   └── calculator.py
│   │   └── rag_placeholder.py
│   │   └── outlets.py
│   │   └── products.py
```

  * **Backend:** FastAPI for serving API endpoints (`/products`, `/outlets`, etc.)
  * **Frontend (Demo):** Streamlit app for interactive exploration
  * **LLM Interface:** Google Gemini via `langchain-google-genai`
  * **Memory:** LangChain `ConversationBufferMemory`
  * **Agent Framework:** LangChain's tool and planner integration
  * **Vector Store:** FAISS for product RAG
  * **Database Engine:** SQLite
  * **Querying Interface:** Text-to-SQL via Gemini Flash

### Tools

  * **Calculator:** Basic arithmetic tool.
  * **Product RAG:** Retrieves information from a vector DB (FAISS).
  * **Outlet Text2SQL:** Converts natural language to SQL queries over the outlet database.

### Architectural Choices and Trade-offs

| Area                          | Choice Made                       | Trade-off / Rationale                                                                                                                                                                             |
| :---------------------------- | :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM Provider                  | Google Gemini                     | Fast + cheap (Gemini Flash), but limited control vs. local models.                                                                                                                        |
| LangChain Usage               | Yes                               | Rapid development & tool abstraction, but can be verbose and fragile at times.                                                                                                            |
| Streamlit UI                  | Yes                               | Simple to build demos, but not suitable for production-scale interfaces.                                                                                                                  |
| Product Info Retrieval        | RAG via FAISS + Gemini            | Enables flexible answers based on knowledge retrieval.                                                                                                                                    |
| Vector Store                  | FAISS (in-memory)                 | Lightweight and easy to use for small datasets, but not scalable to millions of documents.                                                                                                |
| SQL Backend                   | SQLite + Text-to-SQL via Gemini   | No need to manage a full database server, but depends on LLM's reliability and structured prompt.                                                                                         |
| Error Handling (Part 5)       | Manual fallback responses         | Good for demo robustness, but limited resilience under real-world failures.                                                                                                               |
| File Structure                | One script per feature (`chatbot_partX`)| Easy to test/debug in isolation                                                                                                                          |

-----

## Chatbot Flow

```
[User]
   |
   v
[Streamlit Frontend / Terminal]
   |
   v
[Chatbot Agent (LangChain)]
   |
   v
[Intent Router / Planner]
   |
   |---> Is intent clear?
   |           |
   |           |-- No --> Ask follow-up for clarity
   |           |
   |           |-- Yes --> Take action:
   |                     |
   |                     +--> Call Calculator Tool (/calculator)
   |                     +--> Query Product Info via RAG (/products)
   |                     +--> Query Outlet Info via Text2SQL (/outlets)
   |
   v
[LLM (Gemini)] --> processes + formats response
   |
   v
[Final Answer Returned to User]
```

-----

## Part 1: Sequential Conversation with Memory

### 1\. Code-First Chatbot with Memory

  * **Implemented in:** `chatbot_app/chatbot_part1.py`
  * Uses `ConversationBufferMemory` from LangChain to track chat history.
  * Powered by Google Gemini (`ChatGoogleGenerativeAI`).
  * Responds contextually to user follow-ups.

### 2\. Automated Unit Tests

  * **Located in:** `tests/test_part1_sequential_conversation.py`
  * Includes:
      * **Happy Path Test:** Verifies multi-turn memory works (e.g., outlet \> location \> opening hours).
      * **Interrupted Path Test:** Handles incomplete context gracefully by prompting for clarification.

To run the tests:

```bash
pytest tests/test_part1_sequential_conversation.py
```

### 3\. Example Conversation Flow

  * **User:** Is there an outlet in Petaling Jaya?
  * **Bot:** Yes\! Which outlet are you referring to?
  * **User:** SS 2, what's the opening time?
  * **Bot:** Ah yes, the SS 2 outlet opens at 9:00 AM.

-----

## Part 2: Agentic Planning

### Objective

Implement a controller that determines the chatbot's next action based on user input — whether to ask for clarification, call a tool, or provide a final answer.

### 1\. Planner/Controller Logic

  * **Implemented in:** `chatbot_app/chatbot_part2.py`
  * Custom logic parses the user’s input for:
      * **Intent detection** (e.g., ask a question, perform calculation, retrieve outlet info).
      * **Missing slot detection** (e.g., missing location, outlet name, product detail).
  * Based on intent and completeness:
      * Asks a follow-up question.
      * Invokes the calculator tool.
      * Calls RAG/Text2SQL endpoints.
      * Or directly returns a final response.

### 2\. Automated Unit Tests

  * **Located in:** `tests/test_part2_agentic_planning.py`
  * Tests planner behavior:
      * Clarification logic for incomplete inputs.
      * Tool routing and API usage.
      * Correct fallback handling for unsupported queries.

### 3\. Decision Flow Description

A brief explanation of decision points is documented in the code and also summarized below:

  * If the user asks a math question → use `calculator_tool`.
  * If an outlet-related query with no location → ask for location.
  * If a product-related query → trigger RAG tool.
  * If all required info is present → answer directly.
  * If intent is ambiguous → clarify with the user.

-----

## Part 3: Tool Calling

### Objective

Integrate a Calculator Tool to perform simple arithmetic operations via the chatbot. The agent must:

  * Detect when the user input expresses a mathematical intent.
  * Invoke the calculator endpoint with a valid expression.
  * Parse the response and return the result conversationally.
  * Gracefully handle invalid inputs or tool failures without crashing.

### 1\. Calculator API Endpoint (Backend Integration)

This is the code powering the actual `/calculator` HTTP endpoint. It lives in:

#### a. Calculator endpoint in `app/main.py`


### For Testing:

To run the tests:

```bash
pytest -s tests/test_part3_tool_calling.py
```

To interact with `chatbot_part3`:

```bash
python -m chatbot_app.chatbot_part3
```

### Example/Transcript:

**a. User:** 3+1
**Bot:** 4

**b. User:** 5 plus 10
**Bot:** 15

**d. User:** cat plus 1
**Bot:** Sorry, something went wrong: An output parsing error occurred. I cannot perform this operation. "Cat" is not a numerical value that can be added to 1.

-----

## Part 4: Custom API & RAG Integration

### 1\. FastAPI Backend with OpenAPI Specification

A complete FastAPI application exposing the following endpoints with full OpenAPI (Swagger) documentation:

  * **POST /chatbot****

      * Processes incoming natural language questions from the user's request body.
      * A LangChain-based agent analyzes the query, determining intent and routing the request to the appropriate backend tool:
      * Calculator Tool for arithmetic operations.
      * Product Retrieval Tool (RAG) for knowledge base lookups from a FAISS vector store.
      * Outlet Tool (Text-to-SQL) for querying ZUS Coffee outlet information.
      * Returns an AI-generated natural language response derived from the tool's execution.
      * The core chatbot logic and routing are managed within app/main.py (via run_chatbot_logic) and chatbot_app/tools/.

  * **GET `/products?query=<user_question>`**

      * Retrieves relevant product knowledge base (KB) documents from a FAISS vector store based on the user's query.
      * Returns an AI-generated summary combining the top-k most relevant product entries.
      * Powered by semantic search and summarization logic in `app/rag.py`.

  * **POST `/outlets?query=<nl_query>`**

      * Translates natural language queries into SQL commands using a Text2SQL pipeline.
      * Executes the generated SQL on a relational database containing ZUS Coffee outlet information.
      * Returns structured query results (e.g., outlet names, locations, hours).
      * Implemented via `app/text2sql_outlets.py` and `app/llm_sql_generator.py`.

  * **GET `/calculator?expression=<math_expression>`**

      * Safely evaluates mathematical expressions submitted in the query.
      * Returns the calculated result (e.g., for price comparisons or total cost scenarios).
      * The core logic is implemented in `app/calculator_logic.py`.

### 2\. Vector-Store Ingestion and Retrieval for Product KB

  * Scripts (`data_ingestion/drinkware_scrapper.py`) to scrape and ingest ZUS Coffee drinkware product documents from `https://shop.zuscoffee.com/` (Drinkware category only).
  * Preprocessing and embedding generation of product documents to populate the vector store (FAISS, `data_ingestion/build_product_vector_store.py`).
  * Retrieval code that performs similarity search in the vector store to return top-k relevant documents for any user query by `app/rag.py`.

### 3\. Text2SQL Pipeline and Outlets Database

  * Schema design and creation of a relational SQL database storing ZUS Coffee outlet information, including:
      * Outlet name, location/address, opening hours, services offered.
  * Scripts or code to scrape outlet data is located in `data_ingestion/outlet_scrapper.py`.
  * A prompt engineering setup and pipeline that converts natural language queries into executable SQL commands targeting the outlets database. SQL query executor that runs the generated SQL safely and returns results to the API. Logic is stored in `app/text2sql_outlets.py` and `app/llm_sql_generator.py`.

### 4\. Chatbot Integration

#### 1\. For Terminal

```
├── chatbot_app/
│   └── __pycache__
│   └── __init__.py
│   └── chatbot_part1.py  <-- for sequential conversations
│   └── chatbot_part2.py  <-- for agentic planning (calculator and rag_placeholder)
│   └── chatbot_part3.py  <-- for (calculator) tool calling only
│   └── chatbot_part4.py  <-- integrates 3 tools (calculator, products and outlets)
│   └── tools/
│   │   └── calculator.py
│   │   └── rag_placeholder.py
│   │   └── outlets.py
│   │   └── products.py
```

To run:

```bash
python -m chatbot_app.chatbot_part4
```

This launches the chatbot interface, which demonstrates calling all three backend endpoints interactively.

#### 2\. Running the FastAPI Backend with Swagger UI

The FastAPI backend source files are organized as follows:

```
app/
├── main.py              # FastAPI app with endpoint routes
├── calculator_logic.py  # Calculator endpoint logic
├── rag.py               # Product endpoint logic
├── text2sql_outlets.py  # Text2SQL logic
└── llm_sql_generator.py # SQL generation using LLM for Text2SQL conversion
```

To start the API server/Swagger UI:

```bash
python -m uvicorn app.main:app --reload
```

Open the Swagger UI documentation at `http://localhost:8000/docs` to interact with the `/products`, `/outlets`, and `/calculator` endpoints.

If you cannot load, try changing the port:

```bash
uvicorn app.main:app --reload --port 8001
```

#### Chatbot Endpoint (for API interaction)

  * **Located in:** `app/main.py`
  * This endpoint accepts natural language questions from the user. It delegates the question to the LangChain agent, which determines and calls the correct tool (Calculator, ProductTool, or OutletTool) before returning an LLM-generated natural language answer.

#### Calculator Tool

  * **Tool Wrapper:** `chatbot_app/tools/calculator.py`
  * **Functionality:** Integrates with the `/calculator` FastAPI endpoint.
  * **Flow:** User Query \> Chatbot Agent decides \> API Request to `/calculator` \> Calculation Processing \> Response \> Chatbot Response

#### Products Tool (RAG Integration)

  * **Tool Wrapper:** `chatbot_app/tools/products.py`
  * **Functionality:** Integrates with the `/products` FastAPI endpoint to perform RAG-based knowledge base lookups.
  * **Flow:** User Query \> Chatbot Agent decides \> API Request to `/products` \> Vector Search & Summarization \> Response \> Chatbot Response

#### Outlets Tool (Text2SQL Integration)

  * **Tool Wrapper:** `chatbot_app/tools/outlets.py`
  * **Functionality:** Integrates with the `/outlets` FastAPI endpoint to translate natural language into SQL and query outlet data.
  * **Flow:** User Query \> Chatbot Agent decides \> API Request to `/outlets` \> Text2SQL Conversion \> SQL Execution \> Response \> Chatbot Response

### 5\. Example Transcripts and Testing

#### SUCCESS CASES

1. To run the tests:

```bash
pytest -s 'tests/test_part4_chatbot_integration.py::test_success_cases[What\'s 25 * 4 + 100?-expected_keywords0]'
```

**User:** What is 25 \* 4 + 100?
**Bot:** 25 \* 4 + 100 is 200.

**Reasoning:**
Action: Calculator
Action Input: 25 \* 4 + 100200 I now know the final answer.
Final Answer: 25 \* 4 + 100 is 200.

-----

2. To run the tests:

```bash
pytest -s 'tests/test_part4_chatbot_integration.py::test_success_cases[Which bottles are BPA-free?-expected_keywords1]'
```

**Transcript for:** “Which bottles are BPA-free?”
**User:** Which bottles are BPA-free?

**Bot:**
All of the ZUS All Day Cups listed are BPA-free. This includes:
ZUS All Day Cup 500ml (17oz) – Sundaze Collection
ZUS All Day Cup 500ml (17oz) – Aqua Collection
ZUS All Day Cup 500ml (17oz) – Mountain Collection

**Reasoning:**
Thought: The user is asking about a specific characteristic (BPA-free) of a product type (bottles). This is a product-related query that can be answered using the `product_search_tool`.
Action: `product_search_tool`
Action Input: Which bottles are BPA-free?

Observation: [Retrieved relevant product data from vector store including product\_info with 'BPA Free' for all 3 ZUS All Day Cup collections.]

Thought: I now know the final answer.
Final Answer: All of the ZUS All Day Cups listed are BPA-free. This includes:

  - ZUS All Day Cup 500ml (17oz) – Sundaze Collection
  - ZUS All Day Cup 500ml (17oz) – Aqua Collection
  - ZUS All Day Cup 500ml (17oz) – Mountain Collection

-----

3. To run the tests:

```bash
pytest -s 'tests/test_part4_chatbot_integration.py::test_success_cases[List all outlets in Selangor-expected_keywords2]'
```

**Transcript:** "List all outlets in Selangor"
**User:** List all outlets in Selangor.

**Bot:**
Here are the ZUS Coffee outlets located in Selangor:
ZUS Coffee @ IOI Mall Puchong
ZUS Coffee @ SS15 Subang Jaya
ZUS Coffee @ Sunway Pyramid
ZUS Coffee @ Setia City Mall
ZUS Coffee @ Subang Parade
(...list continues based on your actual outlet data returned)

**Reasoning:**
Thought: The user is asking for a list of outlets in Selangor. This is an outlet-related query and the `outlet_search_tool` is appropriate for this. The query is specific enough.
Action: `outlet_search_tool`
Action Input: List all outlets in Selangor

Observation: [Retrieved relevant outlet data from database, addresses from Selangor.]

Thought: I now know the final answer.
Final Answer: Here are the ZUS Coffee outlets located in Selangor:
ZUS Coffee @ IOI Mall Puchong
ZUS Coffee @ SS15 Subang Jaya
ZUS Coffee @ Sunway Pyramid
ZUS Coffee @ Setia City Mall
ZUS Coffee @ Subang Parade
(...list continues based on your actual outlet data returned)

#### FAILURE CASES

1. To run the tests:

```bash
pytest -s 'tests/test_part4_chatbot_integration.py::test_failure_cases[Calculate apple + 5-expected_keywords0]'
```

**User:** Calculate apple + 5
**Bot:** I can only calculate numbers. "Apple" is not a number.

**Reasoning:**
Thought: The user is asking to perform a calculation with "apple + 5". The Calculator tool can only handle numerical expressions. "Apple" is not a number, so this calculation cannot be performed.
Final Answer: I can only calculate numbers. "Apple" is not a number.

-----

2. To run the tests:

```bash
pytest -s 'tests/test_part4_chatbot_integration.py::test_failure_cases[Any ZUS products made of gold?-expected_keywords1]'
```

**User:** Any ZUS products made of gold?
**Bot:** No, ZUS Coffee products are not made of gold. They are primarily made of Stainless Steel or other BPA Free materials.

**Reasoning:**
Thought: The user is asking about ZUS products and if any are made of gold. This is a product-related query. I need to use the `product_search_tool` to determine if any ZUS products are made of gold.
Action: `product_search_tool`
Action Input: products made of gold

Observation: Based on the product information provided, none of the ZUS Coffee products listed (ZUS All Day Cup, ZUS Frozee Cold Cup, ZUS All-Can Tumbler) are made of gold. They are primarily made of **Stainless Steel** or other **BPA Free** materials.
Final Answer: No, ZUS Coffee products are not made of gold. They are primarily made of Stainless Steel or other BPA Free materials.

Thought: I now know the final answer.
Final Answer: products made of gold

-----

3. To run the tests:

```bash
pytest -s 'tests/test_part4_chatbot_integration.py::test_failure_cases[Are there any ZUS outlets in Antarctica?-expected_keywords2]'
```

**User:** Are there any ZUS outlets in Antarctica?
**Bot:** There are no ZUS Coffee outlets in Antarctica.

**Reasoning:**
Thought: The user is asking about the existence of ZUS outlets in Antarctica. This is a query about outlet locations, which can be answered using the `outlet_search_tool`.
Action: `outlet_search_tool`
Action Input: Are there any ZUS outlets in Antarctica?
DEBUG - Raw result from query\_outlets\_from\_db: {'result': [{'COUNT(\*)': 0}]}

Observation: There are 0 ZUS Coffee outlets matching your query.
Final Answer: There are no ZUS Coffee outlets in Antarctica.

Thought: I now know the final answer.
Final Answer: There are no ZUS Coffee outlets in Antarctica.

-----

## Part 5: Unhappy Flows

| Scenarios                                   | Tool Used                          | Bot Response                                                                       |
| :------------------------------------------ | :--------------------------------- | :--------------------------------------------------------------------------------- |
| Missing parameters                          | Missing expression (`/calculator`) | `"Error: No expression provided. Please enter a valid mathematical expression."`   |
|                                             | Missing query (`/outlets`)         | `"Error: No query provided. Please ask something like 'Show all outlets in Selangor'."` |
| API downtime                                | Product tool (RAG)                 | `"Sorry, the ZUS server is currently unavailable. Please try again later."`        |
| Malicious payload (SQL injection attempt)   | Outlet tool (Text2SQL)             | `"Your query looks suspicious. Please ask about outlets using natural language."`  |

To run the test in the terminal:

```bash
pytest tests/test_part5_unhappy_flows.py -v
```

### Summary of Error-Handling and Security Strategy

#### Strategy Highlights

  * **Input Validation at Every Layer**

      * Missing or empty fields return helpful, actionable error messages.
      * Input schemas are manually checked for required keys (`query`, `expression`).

  * **Safe Failure & Graceful Degradation**

      * Each tool returns a human-friendly message instead of crashing or exposing internal errors.
      * API downtime is simulated using mocks to ensure the agent fails gracefully.

  * **SQL Injection & Prompt Injection Prevention**

      * Natural language is filtered and responses are sanitized to detect suspicious queries.
      * Example: SQL injection attempts trigger: "Your query looks suspicious..."
      * No raw SQL, stack traces, or backend details are ever exposed to the user.

  * **LLM Prompt Safety**

      * System prompts are reinforced to prevent prompt hijacking.
      * Agent/tool outputs are always checked before returning to the user.

  * **Tool Isolation**

      * Calculator, RAG, and Text2SQL tools are isolated, so one failure won’t affect the rest.
      * Errors from one tool don’t crash the entire chatbot.


### ⚠️ Note on Agent Output Format & Testing Consistency

This implementation uses the Gemini 2.5 Flash model via LangChain's `create_react_agent()` to handle tool-based queries. While the solution adheres to LangChain’s expected `action`/`observation`/`final-answer` format, Gemini’s output can sometimes be inconsistent due to:

  * Variability in generation output (e.g., occasionally omitting `Action:` or `Final Answer:`).
  * Internal temperature or token sampling behavior.
  * Non-determinism across runs or environments.

As a result, the integration tests may pass or fail unpredictably across different machines or runs, even if no code changes are made.

-----

**Disclaimer:**

This project was developed solely for the purpose of a technical assessment and educational demonstration. It includes data scraped from the publicly available ZUS Coffee website strictly for non-commercial, research, and learning purposes.

All trademarks, logos, and brand names are the property of their respective owners. 