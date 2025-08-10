import os
import re
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq.chat_models import ChatGroq

# Load environment variables
load_dotenv()

# ------------------------------
# 1. Initialize Groq LLM
# ------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,  # slightly lower for more deterministic JSON
    api_key=os.getenv("GROQ_API_KEY"),
)

# ------------------------------
# 2. Detection Prompt & Chain
# ------------------------------
detect_prompt = PromptTemplate(
    input_variables=["summary"],
    template=(
        "You are a mental health AI assistant. Based on this chat summary, "
        "determine if the user shows signs of depression. "
        "Respond with EXACTLY one of the following formats:\n"
        "- 'Depression Signs Detected (Confidence: XX%)'\n"
        "- 'No Depression Signs Detected (Confidence: XX%)'\n\n"
        "Summary:\n{summary}"
    ),
)
detect_chain = LLMChain(llm=llm, prompt=detect_prompt)

# ------------------------------
# 3. Emotion Prompt & Chain
# ------------------------------
emotion_prompt = PromptTemplate(
    input_variables=["summary"],
    template=(
        "You are an emotion classification assistant. Read the chat summary and "
        "choose the single most likely prevailing emotion from this fixed set: "
        "sad, happy, neutral, angry, fearful.\n"
        "Return STRICT JSON ONLY with the following schema:\n"
        '{{"emotion":"<one of: sad|happy|neutral|angry|fearful>","confidence":<integer 0-100>}}'
        "\nNo extra text.\n\n"
        "Summary:\n{summary}"
    ),
)
emotion_chain = LLMChain(llm=llm, prompt=emotion_prompt)

# ------------------------------
# 4. Public Functions
# ------------------------------
def detect_from_summary(summary: str) -> str:
    """Detect depression signs from a manually provided chat summary."""
    result = detect_chain.invoke({"summary": summary})
    text = result["text"] if isinstance(result, dict) else str(result)
    return f"Summary:\n{summary}\n\nDetection Result: {text}"

def _extract_json(text: str):
    """
    Attempts to parse JSON from a model response.
    Tries direct json.loads first; if it fails, attempts to find the first {...} block.
    """
    # Direct attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    # Regex fallback to first balanced-looking JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

def detect_emotion_from_summary(summary: str) -> dict:
    """
    Classify prevailing emotion (sad, happy, neutral, angry, fearful) with confidence 0-100.
    Returns a dict: {"emotion": <str>, "confidence": <int>}
    """
    raw = emotion_chain.invoke({"summary": summary})
    text = raw["text"] if isinstance(raw, dict) else str(raw)

    data = _extract_json(text) or {}
    emotion = str(data.get("emotion", "neutral")).lower()
    confidence = data.get("confidence", 50)

    # Sanitize outputs
    allowed = {"sad", "happy", "neutral", "angry", "fearful"}
    if emotion not in allowed:
        emotion = "neutral"
    try:
        confidence = int(confidence)
    except Exception:
        confidence = 50
    confidence = max(0, min(100, confidence))

    return {"emotion": emotion, "confidence": confidence}

__all__ = ["detect_from_summary", "detect_emotion_from_summary"]
