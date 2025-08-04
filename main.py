
from dotenv import load_dotenv
from colorama import init, Fore
from agent import chat_with_user, detect_depression, detect_from_summary
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
    """Displays detection result in colored format."""
    if "Depression Signs Detected" in result:
        print(Fore.YELLOW + result)
    elif "No Depression Signs Detected" in result:
        print(Fore.GREEN + result)
    else:
        print(result)

def main():
    check_api_key()
    print(Fore.CYAN + "🧠 Depression Detection AI Agent")
    print("Options:")
    print("1. Chat Mode (type messages to talk)")
    print("2. Summary Mode (type 'summary' to provide chat summary)")
    print("Type 'analyze' in chat mode to detect depression. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break
        elif user_input.lower() == "summary":
            summary_input = input("Paste chat summary: ")
            result = detect_from_summary(summary_input)
            display_detection_result(result)
        elif user_input.lower() == "analyze":
            result = detect_depression()
            display_detection_result(result)
        else:
            response = chat_with_user(user_input)
            print("Agent:", response)

if __name__ == "__main__":
    main()
