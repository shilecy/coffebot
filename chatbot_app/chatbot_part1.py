import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel # Import for type hinting of LLM

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

class MindhiveChatbot:
    def __init__(self, llm: BaseChatModel = None, memory_obj: ConversationBufferMemory = None):
        # Initialize the LLM (Large Language Model)
        # If an LLM is provided, use it. Otherwise, create a default one.
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        else:
            self.llm = llm # Use the provided LLM (e.g., a mock)

        # Initialize memory to keep track of conversation history
        # If memory_obj is provided, use it. Otherwise, create a default one.
        if memory_obj is None:
            self.memory = ConversationBufferMemory(memory_key="chat_history")
        else:
            self.memory = memory_obj # Use the provided memory

        # Define the prompt template for the chatbot
        self.prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)

        # Create an LLMChain, which combines the LLM, prompt, and memory
        self.conversation = LLMChain(
            llm=self.llm, # This will now be the provided (or default) LLM
            prompt=self.prompt,
            verbose=False,
            memory=self.memory
        )

    def chat(self, user_input: str) -> str:
        """
        Processes user input and returns the chatbot's response.
        The invoke method handles passing inputs to the chain and retrieving the output.
        """
        try:
            response = self.conversation.invoke({"human_input": user_input})
            # The invoke method returns a dictionary; the actual response is in the 'text' key.
            return response['text'].strip() # .strip() to remove leading/trailing whitespace
        except Exception as e:
            # Basic error handling for now. We'll improve this in Part 5.
            print(f"An error occurred during chat: {e}")
            return "I'm sorry, I encountered an issue. Please try again."

# Define the prompt template *outside* the class for clarity if it's static
template = """You are a helpful assistant for ZUS Coffee.
You are designed to answer questions about ZUS Coffee outlets and products,
and to perform simple calculations.
Maintain a friendly and polite tone.

Current conversation:
{chat_history}
Human: {human_input}
AI:"""


# CLI testing
if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set in your .env file in the project root
    # e.g., GOOGLE_API_KEY="your_api_key_here"
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
        exit()

    # When run directly, MindhiveChatbot() will use default LLM and memory
    chatbot = MindhiveChatbot()
    print("Mindhive Chatbot initialized. Type 'exit' to end the conversation.")
    print("You can try: 'Is there an outlet in Petaling Jaya?'")
    print("Then: 'SS 2, whats the opening time?'")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting conversation. Goodbye!")
            break
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
