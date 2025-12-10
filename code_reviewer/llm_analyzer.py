import os
import google.generativeai as genai
import json
import re
import time

def analyze_with_llm(code_snippet):
    """
    Analyzes a code snippet using the Google Gemini model (Gemini 2.5 Flash).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Warning: GOOGLE_API_KEY not found in .env file. Skipping LLM analysis.")
        return []

    try:
        genai.configure(api_key=api_key)
        
        # UPDATED: Using 'gemini-2.5-flash' which has higher rate limits for free tier
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are an expert Senior Software Engineer performing a critical code review on a Java code snippet.
        Your task is to identify potential logical errors, security vulnerabilities, performance issues, or significant violations of best practices.
        Do NOT comment on simple style issues that a linter would catch. Focus only on major and critical issues.

        For each distinct issue you find, provide a response as a JSON object within a list.
        Each JSON object MUST have exactly three keys:
        1. "line": an integer representing the line number where the issue occurs (or your best guess if it spans multiple lines).
        2. "severity": a string, which must be either "major" or "critical".
        3. "message": a string, providing a clear and concise explanation of the issue and why it is a problem.

        If you find no issues, you MUST return an empty list: [].

        Here is the code to review:
        ```java
        {code_snippet}
        ```
        """

        # Simple retry logic for transient errors
        try:
            response = model.generate_content(prompt)
        except Exception as e:
            if "429" in str(e):
                print("⚠️ Quota exceeded. Waiting 30 seconds before retrying...")
                time.sleep(30)
                response = model.generate_content(prompt)
            else:
                raise e

        text = response.text

        # --- Robust JSON Cleaning ---
        # 1. Try to find the JSON list inside the text using Regex
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            json_response_str = json_match.group(0)
        else:
            # If regex fails, try simple stripping
            json_response_str = text.strip().replace('```json', '').replace('```', '').strip()
        
        # 2. Parse the JSON
        issues = json.loads(json_response_str)
        
        for issue in issues:
            issue['source'] = 'LLM'
            
        return issues

    except Exception as e:
        print(f"❌ An error occurred during LLM analysis: {e}")
        return []