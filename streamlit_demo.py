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
from chatbot_app.chatbot_part3 import MindhiveChatbot as MindhiveChatbotPart3
from chatbot_app.chatbot_part4 import MindhiveChatbot as MindhiveChatbotPart4

# Import your tools
from chatbot_app.tools.calculator import calculate
from chatbot_app.tools.products import rag_tool
from chatbot_app.tools.outlets import outlet_tool
from chatbot_app.tools.rag_placeholder import zus_info_retriever

# Load environment variables (like GOOGLE_API_KEY)
# Only try to load dotenv if running locally (st.secrets is empty)
if not st.secrets:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env locally")
    except ImportError:
        print("⚠️ dotenv not available — skipping .env load")

# Works on both local and Streamlit Cloud
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in .env (local) or secrets (Streamlit Cloud).")
    st.stop()

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
    "Part 2: Agent with Tools (Calculator & Info placeholder)": MindhiveChatbotPart2,
    "Part 3: Dedicated Calculator Agent": MindhiveChatbotPart3,
    "Part 4: Advanced Agent with Multiple Tools": MindhiveChatbotPart4,
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
        temperature = 0.5
    elif "Part 2" in mode_name:
        temperature = 0.3
    elif "Part 3" in mode_name or "Part 4" in mode_name:
        temperature = 0.2
    else:
        temperature = 0.2 # Default for other parts if not specified yet

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
            # Try to find the right chat method
            method = (
                getattr(current_chatbot_instance, "chat", None)
                or getattr(current_chatbot_instance, "chat_2", None)
                or getattr(current_chatbot_instance, "chat_3", None)
                or getattr(current_chatbot_instance, "chat_4", None)
            )

            if callable(method):
                response = method(prompt)
            else:
                response = "❌ No available chat method on this chatbot instance."

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