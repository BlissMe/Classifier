from dotenv import load_dotenv
from colorama import init, Fore, Style
from agent import detect_from_summary, detect_emotion_from_summary
import os
import sys

# Initialize colorama
init(autoreset=True)

def check_api_key():
    """Ensure Groq API key is available before running."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(Fore.RED + "❌ ERROR: GROQ_API_KEY not found.")
        print("Please set it in your environment or .env file.")
        sys.exit(1)

def display_detection_result(result: str):
    """Displays depression detection result in colored format."""
    if "Depression Signs Detected" in result:
        print(Fore.YELLOW + result)
    elif "No Depression Signs Detected" in result:
        # In your pipeline, this confidence number is inverted to represent
        # "depression likelihood". So green = low likelihood.
        print(Fore.GREEN + result)
    else:
        print(result)

def display_emotion_result(emotion):
    """
    Displays emotion classification result with color mapping.
    Accepts either a string or a dict like {"emotion": "sad"}.
    """
    if isinstance(emotion, dict):
        emotion = emotion.get("emotion", "neutral")
    if not isinstance(emotion, str) or not emotion:
        emotion = "neutral"

    color_map = {
        "happy": Fore.GREEN,
        "neutral": Fore.GREEN,
        "sad": Fore.GREEN,
        "angry": Fore.GREEN,
        "fearful": Fore.GREEN,
    }
    color = color_map.get(emotion.lower(), Fore.WHITE)
    print(color + f"Emotion: {emotion.capitalize()}" + Style.RESET_ALL)

def main():
    check_api_key()
    print(Fore.CYAN + "🧠 Classifier Agent")
    print("Paste a chat summary to analyze for depression signs and emotion.")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            summary_input = input("Paste chat summary (or 'exit' to quit): ").strip()
            if summary_input.lower() == "exit":
                print("Agent: Goodbye!")
                break
            if not summary_input:
                print(Fore.RED + "Please paste a non-empty summary.\n")
                continue

            # Depression detection (with adjusted confidence for 'No Depression' case)
            dep_result = detect_from_summary(summary_input)
            display_detection_result(dep_result)

            # Emotion detection (label only)
            emotion_label = detect_emotion_from_summary(summary_input)
            display_emotion_result(emotion_label)
            print()  # spacer
    except KeyboardInterrupt:
        print("\nAgent: Goodbye!")

if __name__ == "__main__":
    main()
