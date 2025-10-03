import os
import google.generativeai as genai
import json

class LLMAnalyzer:
    """
    Handles sending code to a Large Language Model (Google's Gemini) for deep analysis.
    """
    def __init__(self):
        self.model = None
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print("Google Gemini model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Gemini model: {e}")

    def analyze(self, code_string):
        """
        Sends the code to the Gemini model for a review and parses the JSON response.
        
        Args:
            code_string (str): The raw source code of the file to analyze.
            
        Returns:
            list: A list of dictionaries, where each dictionary is an issue found by the LLM.
        """
        if not self.model:
            print("LLM model not available. Skipping deep analysis.")
            return []

        prompt = self._build_prompt(code_string)
        
        try:
            # Tell the model to generate a JSON response
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            return self._parse_llm_response(response.text)

        except Exception as e:
            print(f"An error occurred while communicating with the LLM: {e}")
            return []

    def _build_prompt(self, code_string):
        """
        Constructs the detailed prompt to send to the LLM.
        """
        return f"""
        You are an expert senior Java software engineer and a world-class code reviewer.
        Your task is to analyze the following Java code snippet for potential issues.

        Please identify a wide range of problems, including but not limited to:
        1.  **Critical Bugs:** NullPointerExceptions, resource leaks, race conditions, infinite loops.
        2.  **Security Vulnerabilities:** SQL injection, insecure direct object references, cross-site scripting risks.
        3.  **Performance Issues:** Inefficient algorithms (e.g., O(n^2) loops), unnecessary object creation, slow I/O operations.
        4.  **Best Practice Violations:** Poor naming, lack of comments, overly complex methods, mutable state being passed where it shouldn't be.

        Analyze the code below:
        ```java
        {code_string}
        ```

        Provide your feedback in a strict JSON format. The output MUST be a JSON array of objects.
        Each object in the array should represent a single issue and have the following structure:
        {{
            "line": <line_number_integer>,
            "severity": "<critical|major|minor>",
            "message": "<A short, one-sentence summary of the issue>",
            "description": "<A detailed, two-to-three sentence explanation of the issue, its impact, and a suggestion for how to fix it.>"
        }}

        If you find no issues, return an empty JSON array [].
        Do not include any text or explanations outside of the main JSON array.
        """

    def _parse_llm_response(self, response_text):
        """
        Safely parses the JSON response from the LLM.
        """
        try:
            issues = json.loads(response_text)
            # Add the 'type' field to each issue
            for issue in issues:
                issue['type'] = 'llm'
            return issues
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from LLM response.")
            print(f"Received: {response_text}")
            return []
        except Exception as e:
            print(f"An error occurred while parsing the LLM response: {e}")
            return []

# Create a single, reusable instance for the app to import
llm_analyzer = LLMAnalyzer()
