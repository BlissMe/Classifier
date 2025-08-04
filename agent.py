import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_groq.chat_models import ChatGroq

# Load environment variables
load_dotenv()

# ------------------------------
# 1. Initialize Groq LLM
# ------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------------------
# 2. Prompts
# ------------------------------
summary_prompt = PromptTemplate(
    input_variables=["conversation"],
    template="Summarize the following chat:\n{conversation}\nSummary:"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

detect_prompt = PromptTemplate(
    input_variables=["summary"],
    template=(
        "You are a mental health AI assistant. Based on this chat summary, "
        "determine if the user shows signs of depression. "
        "Respond with:\n"
        "- 'Depression Signs Detected (Confidence: XX%)' or\n"
        "- 'No Depression Signs Detected (Confidence: XX%)'\n\n"
        "Summary:\n{summary}"
    )
)
detect_chain = LLMChain(llm=llm, prompt=detect_prompt)

# ------------------------------
# 3. Conversation Memory
# ------------------------------
conversation_history = []

# ------------------------------
# 4. Functions
# ------------------------------
def chat_with_user(user_input: str) -> str:
    """Handles regular conversation with the user."""
    conversation_history.append(f"User: {user_input}")
    response = llm.invoke([HumanMessage(content=user_input)]).content
    conversation_history.append(f"Agent: {response}")
    return response

def summarize_conversation(_: str = "") -> str:
    """Summarizes the conversation so far."""
    conversation = "\n".join(conversation_history)
    result = summary_chain.invoke({"conversation": conversation})
    return result["text"] if isinstance(result, dict) else result

def detect_depression(_: str = "") -> str:
    """Detects depression signs based on conversation summary."""
    summary = summarize_conversation()
    result = detect_chain.invoke({"summary": summary})
    return f"Summary:\n{summary}\n\nDetection Result: {result['text'] if isinstance(result, dict) else result}"

def detect_from_summary(summary: str) -> str:
    """Detect depression signs from a manually provided chat summary."""
    result = detect_chain.invoke({"summary": summary})
    return f"Summary:\n{summary}\n\nDetection Result: {result['text'] if isinstance(result, dict) else result}"

# ------------------------------
# 5. Tools and Agent
# ------------------------------
tools = [
    Tool(name="Summarizer", func=summarize_conversation, description="Summarizes chat so far."),
    Tool(name="Depression Detector", func=detect_depression, description="Detects depression signs from conversation summary.")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Export functions and agent
__all__ = ["chat_with_user", "agent", "detect_depression", "detect_from_summary"]
