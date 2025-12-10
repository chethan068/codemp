import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_available_models():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    try:
        genai.configure(api_key=api_key)
        
        print("--- Available Gemini Models ---")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Name: {m.name}")
                print(f"Display Name: {m.display_name}")
                print(f"Description: {m.description}")
                print("-" * 30)

    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_available_models()