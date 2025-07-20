import streamlit as st
import os
from dotenv import load_dotenv

print("--- Streamlit App Start ---")

# --- LangChain Imports (common for all parts) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_react_agent, AgentExecutor, initialize_agent
from langchain_core.tools import Tool

# --- Import Chatbot Parts from their respective files (using relative imports) ---

from chatbot_app.chatbot_part1 import MindhiveChatbot as MindhiveChatbotPart1
from chatbot_app.chatbot_part2 import MindhiveChatbot as MindhiveChatbotPart2
from chatbot_app.chatbot_part3 import run_calculator_agent
from chatbot_app.chatbot_part4 import run_chatbot_logic

# Import your tools
from chatbot_app.tools.calculator import calculate
from chatbot_app.tools.products import rag_tool
from chatbot_app.tools.outlets import outlet_tool
from chatbot_app.tools.rag_placeholder import zus_info_retriever

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()
print("Environment variables loaded.")

# --- Wrapper Classes for Part 3 and Part 4 ---
class MindhiveChatbotPart3Wrapper:
    def __init__(self, llm: BaseChatModel = None, memory_obj: ConversationBufferMemory = None):
        print("Initializing MindhiveChatbotPart3Wrapper (no direct LLM/memory usage here)...")
        pass

    def chat(self, user_input: str) -> str:
        return run_calculator_agent(user_input)

class MindhiveChatbotPart4Wrapper:
    def __init__(self, llm: BaseChatModel = None, memory_obj: ConversationBufferMemory = None):
        print("Initializing MindhiveChatbotPart4Wrapper (no direct LLM/memory usage here)...")
        pass

    def chat(self, user_input: str) -> str:
        return run_chatbot_logic(user_input)


# --- Streamlit App Setup ---
print("Setting Streamlit page config...")
st.set_page_config(page_title="MindHive Chatbot Demo", layout="centered")
st.title("🧠 MindHive Chatbot Demo")

st.markdown("""
Welcome to the MindHive Chatbot Demo!
Use the dropdown below to switch between different chatbot functionalities.
""")
print("Streamlit title and markdown displayed.")

# --- Define available chatbot modes and their corresponding classes/wrappers ---
CHATBOT_MODES = {
    "Part 1: Simple Conversation": MindhiveChatbotPart1,
    "Part 2: Agent with Tools (Calculator & ZUS Info)": MindhiveChatbotPart2,
    "Part 3: Dedicated Calculator Agent": MindhiveChatbotPart3Wrapper,
    "Part 4: Advanced Agent with Multiple Tools": MindhiveChatbotPart4Wrapper,
}
print("Chatbot modes defined.")

# --- Chatbot Mode Selection ---
print("Attempting to render selectbox...")
selected_mode_name = st.selectbox(
    "Select Chatbot Mode:",
    list(CHATBOT_MODES.keys()),
    key="chatbot_mode_selector",
    help="Choose which part of the chatbot functionality you want to demo."
)
selected_chatbot_class = CHATBOT_MODES[selected_mode_name]
print(f"Selected chatbot mode: {selected_mode_name}")
st.info(f"You are currently interacting with: **{selected_mode_name}**")
print("Info message displayed.")


# --- Initialize LLM and Chatbot Instance (per selected mode) ---
# Use st.cache_resource to cache the LLM and chatbot instance
# This ensures they are initialized only once per session, preventing repeated heavy loads.
@st.cache_resource(hash_funcs={ChatGoogleGenerativeAI: lambda _: None, ConversationBufferMemory: lambda _: None})
def get_llm_and_chatbot_instance(mode_name: str, chatbot_class: type) -> tuple[BaseChatModel, ConversationBufferMemory, object]:
    print(f"Caching LLM and chatbot instance for {mode_name}. Initializing...")
    # Determine temperature based on the selected mode, as per your original parts
    if "Part 1" in mode_name:
        temperature = 0.7
    elif "Part 2" in mode_name:
        temperature = 0.3
    elif "Part 3" in mode_name or "Part 4" in mode_name:
        temperature = 0.2
    else:
        temperature = 0.5 # Default for other parts if not specified yet

    print(f"Initializing ChatGoogleGenerativeAI with model='gemini-2.5-flash', temperature={temperature}...")
    llm_instance = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature
    )
    print("ChatGoogleGenerativeAI initialized.")

    print(f"Initializing ConversationBufferMemory for {mode_name}...")
    memory_instance = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    print("ConversationBufferMemory initialized.")

    print(f"Initializing selected chatbot class: {chatbot_class.__name__}...")
    chatbot_instance = chatbot_class(
        llm=llm_instance,
        memory_obj=memory_instance
    )
    print(f"Chatbot instance for {mode_name} initialized.")
    return llm_instance, memory_instance, chatbot_instance

# Call the cached function
llm_instance, memory_instance, current_chatbot_instance = get_llm_and_chatbot_instance(
    selected_mode_name, selected_chatbot_class
)
print("Current chatbot instance retrieved (from cache or new initialization).")


# --- Initialize Chat History for the current mode ---
print(f"Checking chat history for {selected_mode_name}...")
if f"chat_history_{selected_mode_name}" not in st.session_state:
    st.session_state[f"chat_history_{selected_mode_name}"] = []
    print(f"Chat history for {selected_mode_name} initialized as empty.")
else:
    print(f"Chat history for {selected_mode_name} already exists.")

# --- Display Chat Messages ---
print("Displaying chat messages...")
for speaker, message in st.session_state[f"chat_history_{selected_mode_name}"]:
    with st.chat_message(name=speaker.lower()):
        st.markdown(message)
print("Chat messages displayed.")

# --- User Input and Chat Logic ---
print("Rendering chat input...")
if prompt := st.chat_input("Ask me anything...", key=f"user_input_{selected_mode_name}"):
    # Add user message to chat history
    st.session_state[f"chat_history_{selected_mode_name}"].append(("You", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = current_chatbot_instance.chat(prompt)
            # Ensure the response is a string before passing to st.markdown
            if not isinstance(response, str):
                response = str(response)
            st.markdown(response)
            st.session_state[f"chat_history_{selected_mode_name}"].append(("Bot", response))
        except Exception as e:
            error_message = f"An error occurred while processing your request: {e}"
            st.error(error_message)
            st.session_state[f"chat_history_{selected_mode_name}"].append(("Bot", error_message))

# --- Clear Chat Button ---
print("Rendering clear chat button...")
if st.button("Clear Chat History", key=f"clear_button_{selected_mode_name}"):
    st.session_state[f"chat_history_{selected_mode_name}"] = []
    st.experimental_rerun()
    print("Clear chat button clicked. Rerunning app.")

# --- Instructions and Important Warning ---
print("Rendering sidebar instructions...")
st.sidebar.header("Setup & Important Notes")
st.sidebar.markdown("""
1.  **File Structure:** Ensure your project has the following structure. Pay close attention to the placement of `__init__.py` files and where `streamlit_demo.py` resides.
    ```
    your_project_root/
    └── chatbot_app/
        ├── __pycache__/      # Created automatically by Python
        ├── __init__.py       <-- IMPORTANT: This empty file must exist to make 'chatbot_app' a package
        ├── chatbot_part1.py
        ├── chatbot_part2.py
        ├── chatbot_part3.py
        ├── chatbot_part4.py
        ├── streamlit_demo.py # This file
        └── tools/
            ├── __init__.py   <-- IMPORTANT: This empty file must exist to make 'tools' a package
            ├── calculator.py   (contains `calculate` function)
            ├── rag_placeholder.py (contains `zus_info_retriever` tool)
            ├── products.py     (contains `rag_tool` tool)
            └── outlets.py      (contains `outlet_tool` tool)
    ```
    * **Crucially, the `__init__.py` files must exist** in `chatbot_app/` and `chatbot_app/tools/` to make them valid Python packages.
2.  **Dependencies:** Install necessary libraries:
    ```bash
    pip install streamlit langchain langchain-google-genai python-dotenv pydantic
    ```
3.  **API Key:** Create a `.env` file in `your_project_root/` (the directory *above* `chatbot_app/`) and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```
4.  **Run Locally (CRUCIAL STEP):**
    * Open your terminal.
    * **Navigate to `your_project_root/`** (the directory *containing* the `chatbot_app` folder).
        * For example, if your `your_project_root` is `C:/Users/Acer/Documents/mindhive_assessment`, you would do:
            ```bash
            cd C:/Users/Acer/Documents/mindhive_assessment
            ```
    * Once in `your_project_root/`, run the Streamlit command using Python's module execution:
        ```bash
        python -m streamlit run chatbot_app/streamlit_demo.py
        ```
    * This command explicitly tells Python to run `streamlit_demo.py` as a module within the `chatbot_app` package, allowing it to correctly resolve all relative imports.

---
### ⚠️ **IMPORTANT WARNING: Memory Usage for Cloud Deployment**

This setup still loads the entire LLM and potentially other resources (like FAISS indices or SQLite databases if your tools use them) directly within the Streamlit application's process.

* **Local PC:** This will likely work fine if your PC has sufficient RAM.
* **Cloud Deployment (e.g., Streamlit Cloud, Render Free Tier):** This approach is **highly likely to cause "Out of memory" errors** on free-tier services (which typically offer 512MB - 1GB RAM). Large language models and embedding models consume significant memory.

For robust cloud deployment, especially on free tiers, the recommended architecture is to:
1.  **Deploy your core chatbot logic (LLM, RAG, Text2SQL) as a FastAPI backend** on a platform like Render.
2.  **Have the Streamlit app act purely as a frontend**, making API calls to that deployed backend. This offloads the heavy memory usage from the Streamlit app.

If you encounter memory issues when trying to deploy this combined Streamlit app to a cloud service, please refer back to the FastAPI backend architecture we discussed previously.
""")