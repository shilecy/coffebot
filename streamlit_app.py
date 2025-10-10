import streamlit as st
import os
from dotenv import load_dotenv

print("--- Streamlit App Start (Final Stable Part 4 Agent) ---")

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel

# --- Import ONLY the required Chatbot Part 4 ---
from chatbot_app.chatbot_part4 import MindhiveChatbot as MindhiveChatbotPart4

# Import your tools (Tools are essential for Part 4)
from chatbot_app.tools.calculator import calculate
from chatbot_app.tools.products import rag_tool
from chatbot_app.tools.outlets import outlet_tool
from chatbot_app.tools.rag_placeholder import zus_info_retriever # Included if used by Part 4 logic

# --- API Key Setup (Kept for completeness) ---
if not st.secrets:
    try:
        load_dotenv()
        print("✅ Loaded .env locally")
    except NameError:
        print("⚠️ dotenv not available — skipping .env load") 

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in .env (local) or secrets (Streamlit Cloud).")
    st.stop()

# ===============================================
# === CUSTOM CSS FOR COFFEE SHOP AESTHETICS & LAYOUT (RESTORED) ===
# ===============================================

# --- Custom Theme and Styling ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* Background and Overlay */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.10), rgba(0, 0, 0, 0.6)), 
                        url('https://images.unsplash.com/photo-1509042239860-f550ce710b93?auto=format&fit=crop&w=1740&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Chat container styling */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-radius: 15px !important;
            padding: 15px !important;
        }

        /* Bot message styling */
        [data-testid="stChatMessageAssistant"] {
            background-color: rgba(255, 255, 255, 0.15) !important;
        }

        /* User message styling - Keeping your requested alignment attempt */
        [data-testid="stChatMessageUser"] {
            background-color: #B08968 !important;
            position: relative !important;
            left: 20% !important;
            width: 80% !important;
        }

        /* Text colors */
        .stMarkdown, .stChatMessage {
            color: white !important;
        }

        /* Title styling */
        h1 {
            color: white !important;
            text-align: center !important;
        }

        /* Button styling */
        .stButton > button {
            background-color: #B08968 !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 25px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply custom styling
apply_custom_style()

# --- Streamlit App Setup ---
print("Setting Streamlit page config...")
st.set_page_config(page_title="Advanced CoffeeBot Agent", layout="centered") 

# ===============================================
# === APP CONTENT: TITLE ===
# ===============================================
st.title("☕ CoffeBot Agent")


# ===============================================
# === CHATBOT INITIALIZATION & LOGIC (From streamlit_demo.py) ===
# ===============================================

# Define the single mode name and class for consistency
SELECTED_MODE_NAME = "Part 4: Advanced Agent with Multiple Tools"
SELECTED_CHATBOT_CLASS = MindhiveChatbotPart4

@st.cache_resource(hash_funcs={ChatGoogleGenerativeAI: lambda _: None, ConversationBufferMemory: lambda _: None})
def get_llm_and_chatbot_instance() -> tuple[BaseChatModel, ConversationBufferMemory, object]:
    """Caches and initializes the LLM and the Part 4 Chatbot instance."""
    
    # Use the stable temperature from the demo's Part 4 logic
    temperature = 0.2 
    
    print(f"Initializing ChatGoogleGenerativeAI with model='gemini-2.5-flash', temperature={temperature}...")
    llm_instance = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature
    )

    print("Initializing ConversationBufferMemory...")
    memory_instance = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    print(f"Initializing selected chatbot class: {SELECTED_CHATBOT_CLASS.__name__}...")
    chatbot_instance = SELECTED_CHATBOT_CLASS(
        llm=llm_instance,
        memory_obj=memory_instance
    )
    return llm_instance, memory_instance, chatbot_instance

# Call the cached function
llm_instance, memory_instance, current_chatbot_instance = get_llm_and_chatbot_instance()


# --- START OF GLASS BOX WRAPPER (st.container() defines the wrapped area) ---
with st.container():

    st.markdown("""
    <div style="text-align: center; font-weight: bold;">
    Welcome! I'm your CoffeBot Agent. Ask me about our specials, locations, or even solve a quick calculation!
    <br><br>
    Features: 🧮 Calculations | ☕ Product Info (drinkwares) | 📍 Store Locations
    <br><br>
    Try me: Is the store open now? Do you have any stainless steel products? What is 23 * 47?
    </div>
    """, unsafe_allow_html=True)

    # --- Initialize Chat History ---
    CHAT_HISTORY_KEY = f"chat_history_{SELECTED_MODE_NAME}"
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []

    # --- Display Chat Messages ---
    for speaker, message in st.session_state[CHAT_HISTORY_KEY]:
        with st.chat_message(name=speaker.lower()):
            st.markdown(message)

    # --- User Input and Chat Logic --- 
    if prompt := st.chat_input("Place your order or ask a question...", key=f"user_input_{SELECTED_MODE_NAME}"):
        
        # Add user message to chat history and display it
        st.session_state[CHAT_HISTORY_KEY].append(("You", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Trigger Rerun to avoid double input display
        st.rerun()

    # Process the new user message (always the last one if it's from 'You')
    if st.session_state[CHAT_HISTORY_KEY] and st.session_state[CHAT_HISTORY_KEY][-1][0] == "You":
        prompt = st.session_state[CHAT_HISTORY_KEY][-1][1]
        
        with st.chat_message("assistant"):
            with st.spinner("Brewing a response..."):
                try:
                    # Explicitly look for the 'chat_4' method
                    method = getattr(current_chatbot_instance, "chat_4", None) or getattr(current_chatbot_instance, "chat", None)
                    
                    if callable(method):
                        # Ensure all tool imports the Part 4 agent relies on are available.
                        response = method(prompt) 
                    else:
                        response = "❌ Configuration Error: Could not find 'chat_4' or 'chat' method on the chatbot instance."

                    if not isinstance(response, str):
                        response = str(response)

                    st.markdown(response)
                    st.session_state[CHAT_HISTORY_KEY].append(("Bot", response))

                except Exception as e:
                    error_message = f"An error occurred while processing your request: {e}"
                    st.error(error_message)
                    st.session_state[CHAT_HISTORY_KEY].append(("Bot", error_message))
                
                # Rerun to clear the input box and stabilize chat
                st.rerun()

# --- END OF GLASS BOX WRAPPER ---

# --- Clear Chat Button (Positioned outside the container) ---
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1])
with col2:
    if st.button("Clear Chat History", key=f"clear_button_{SELECTED_MODE_NAME}"):
        st.session_state[CHAT_HISTORY_KEY] = []
        st.rerun()