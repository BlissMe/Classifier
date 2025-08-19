import os
import re
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq  # ✅ correct import

# Load environment variables
load_dotenv()

# ------------------------------
# 1. Initialize Groq LLM
# ------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,  # slightly lower for more deterministic outputs
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
# 3. Emotion Prompt & Chain (label only)
# ------------------------------
emotion_prompt = PromptTemplate(
    input_variables=["summary"],
    template=(
        "You are an emotion classification assistant. Read the chat summary and "
        "choose the single most likely prevailing emotion from this fixed set: "
        "sad, happy, neutral, angry, fearful.\n"
        "Return STRICT JSON ONLY with the following schema:\n"
        "{{\"emotion\":\"<one of: sad|happy|neutral|angry|fearful>\"}}\n"
        "No extra text.\n\n"
        "Summary:\n{summary}"
    ),
)
emotion_chain = LLMChain(llm=llm, prompt=emotion_prompt)

# ------------------------------
# Helpers
# ------------------------------
_CONF_RE = re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%", flags=re.IGNORECASE)

def _invert_confidence_if_no_depression(text: str) -> str:
    """
    If the model output is 'No Depression Signs Detected (Confidence: XX%)',
    replace the confidence with (100 - XX) to reflect 'depression likelihood'.
    """
    if re.search(r"\bNo\s+Depression\s+Signs\s+Detected\b", text, flags=re.IGNORECASE):
        m = _CONF_RE.search(text)
        if m:
            try:
                raw = int(m.group(1))
                raw = max(0, min(raw, 100))
                inverted = 100 - raw
                start, end = m.span(1)
                text = text[:start] + str(inverted) + text[end:]
            except Exception:
                pass
    return text

def _extract_json(text: str):
    """Parse JSON from a model response, with a regex fallback."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

# ------------------------------
# 4. Public Functions
# ------------------------------
def detect_from_summary(summary: str) -> str:
    """Detect depression signs from a manually provided chat summary."""
    result = detect_chain.invoke({"summary": summary})
    text = result["text"] if isinstance(result, dict) else str(result)
    text = _invert_confidence_if_no_depression(text)
    return f"Summary:\n{summary}\n\nDetection Result: {text}"

def detect_emotion_from_summary(summary: str):
    """
    Classify prevailing emotion (sad, happy, neutral, angry, fearful).
    Returns the emotion label as a lowercase string, e.g., 'sad'.
    """
    raw = emotion_chain.invoke({"summary": summary})
    text = raw["text"] if isinstance(raw, dict) else str(raw)
    data = _extract_json(text) or {}
    emotion = str(data.get("emotion", "neutral")).lower()
    allowed = {"sad", "happy", "neutral", "angry", "fearful"}
    if emotion not in allowed:
        emotion = "neutral"
    return emotion

__all__ = ["detect_from_summary", "detect_emotion_from_summary"]
