# # import os
# # from flask import Flask, request, jsonify
# # from dotenv import load_dotenv

# # # Import the class names
# # from .github_client import GithubClient
# # from .ml_model import RiskModel

# # # --- Initialization ---
# # # Load environment variables from .env file
# # load_dotenv()

# # app = Flask(__name__)

# # # Initialize our custom classes
# # try:
# #     # --- THIS IS THE FIX ---
# #     # We now read the secrets from the environment and pass them to the client.
# #     github_client = GithubClient(
# #         app_id=os.getenv("GITHUB_APP_ID"),
# #         private_key=os.getenv("GITHUB_PRIVATE_KEY"),
# #         installation_id=os.getenv("GITHUB_INSTALLATION_ID")
# #     )
# #     risk_model = RiskModel()
# #     print("✅ Models and clients initialized successfully.")
# # except Exception as e:
# #     print(f"❌ Error during initialization: {e}")
# #     github_client = None
# #     risk_model = None


# # @app.route("/webhook", methods=["POST"])
# # def github_webhook():
# #     """
# #     Main webhook endpoint to receive events from the GitHub App.
# #     """
# #     if not github_client or not risk_model:
# #         return jsonify({"status": "error", "message": "Server is not configured properly."}), 500

# #     # --- Security Verification ---
# #     signature = request.headers.get('X-Hub-Signature-256')
# #     if not github_client.verify_signature(request.data, signature):
# #         print("❌ Unauthorized: Webhook signature is not valid.")
# #         return jsonify({"status": "unauthorized"}), 401
    
# #     payload = request.get_json()
# #     event_type = request.headers.get('X-GitHub-Event')

# #     if event_type == "pull_request" and payload.get("action") in ["opened", "synchronize"]:
# #         pr_number = payload["pull_request"]["number"]
# #         repo_name = payload["repository"]["full_name"]
# #         print(f"➡️ Received pull_request '{payload['action']}' event for {repo_name} #{pr_number}")

# #         try:
# #             # --- Analysis Logic ---
# #             # Using the 'diff' is more efficient than getting all files
# #             diff_text = github_client.get_pr_diff(repo_name, pr_number)
# #             changed_snippets = github_client.parse_diff(diff_text)
            
# #             for snippet in changed_snippets:
# #                 filename = snippet['filename']
# #                 # Only analyze .java files
# #                 if not filename.endswith('.java'):
# #                     continue

# #                 # --- Make a Prediction with our Model for each changed snippet ---
# #                 prediction = risk_model.predict(snippet['content'])
# #                 is_high_risk = prediction[0] == 1 # 1 means buggy/high-risk

# #                 print(f"  - Analyzing change in '{filename}': {'HIGH RISK' if is_high_risk else 'Low Risk'}")

# #                 if is_high_risk:
# #                     comment = (
# #                         f"**AI Code Reviewer Report for a change in `{filename}`**\n\n"
# #                         f"Our custom-trained ML model (Accuracy: 0.97) has flagged a code change in this file as **high-risk**. "
# #                         f"The change exhibits patterns that are statistically similar to past bugs.\n\n"
# #                         f"A deeper manual review of this specific change is recommended."
# #                     )
# #                     github_client.post_comment(repo_name, pr_number, comment)
# #                     print(f"  ✅ Posted high-risk comment for '{filename}'.")

# #         except Exception as e:
# #             print(f"❌ An error occurred during analysis: {e}")

# #     return jsonify({"status": "processed"}), 200


# # if __name__ == '__main__':
# #     # Setting debug=False is better for this stage to avoid double-loading
# #     app.run(port=5000, debug=False)


# import os
# import uuid
# import json
# from flask import Flask, request, jsonify, send_from_directory
# from dotenv import load_dotenv

# # Import all our modules
# from .github_client import GithubClient
# from .ml_model import RiskModel
# from .static_analyzer import analyze_with_checkstyle
# from .llm_analyzer import analyze_with_llm

# # --- Initialization ---
# load_dotenv()

# app = Flask(__name__)

# # --- Configuration ---
# REPORTS_DIR = 'reports'
# FRONTEND_DIR = 'frontend'
# # This will be your public ngrok URL
# BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5000") 

# # Initialize our custom classes
# try:
#     github_client = GithubClient(
#         app_id=os.getenv("GITHUB_APP_ID"),
#         private_key=os.getenv("GITHUB_PRIVATE_KEY"),
#         installation_id=os.getenv("GITHUB_INSTALLATION_ID")
#     )
#     risk_model = RiskModel()
#     # Create necessary directories if they don't exist
#     if not os.path.exists(REPORTS_DIR):
#         os.makedirs(REPORTS_DIR)
#     if not os.path.exists(FRONTEND_DIR):
#         os.makedirs(FRONTEND_DIR)
        
#     print("✅ Models and clients initialized successfully.")
# except Exception as e:
#     print(f"❌ Error during initialization: {e}")
#     github_client = None
#     risk_model = None

# # --- API Routes for Frontend ---

# @app.route('/report/<report_id>')
# def serve_report_page(report_id):
#     """Serves the main HTML report page."""
#     # This assumes your report.html is in the 'frontend' directory
#     return send_from_directory(FRONTEND_DIR, 'report.html')

# @app.route('/api/report/<report_id>')
# def get_report_data(report_id):
#     """Serves the JSON data for a specific report."""
#     report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
#     if os.path.exists(report_path):
#         with open(report_path, 'r') as f:
#             data = json.load(f)
#         return jsonify(data)
#     return jsonify({"error": "Report not found"}), 404

# # --- Main Webhook Handler ---

# @app.route("/webhook", methods=["POST"])
# def github_webhook():
#     if not github_client or not risk_model:
#         return jsonify({"status": "error", "message": "Server is not configured properly."}), 500

#     # Security Verification
#     if not github_client.verify_signature(request.data, request.headers.get('X-Hub-Signature-256')):
#         print("❌ Unauthorized: Webhook signature is not valid.")
#         return jsonify({"status": "unauthorized"}), 401
    
#     payload = request.get_json()
#     event_type = request.headers.get('X-GitHub-Event')

#     if event_type == "pull_request" and payload.get("action") in ["opened", "synchronize"]:
#         pr_number = payload["pull_request"]["number"]
#         repo_name = payload["repository"]["full_name"]
#         head_sha = payload['pull_request']['head']['sha']
#         print(f"➡️ Received pull_request '{payload['action']}' event for {repo_name} #{pr_number}")

#         try:
#             diff_text = github_client.get_pr_diff(repo_name, pr_number)
#             changed_snippets = github_client.parse_diff(diff_text)
            
#             for snippet in changed_snippets:
#                 filename = snippet['filename']
#                 if not filename.endswith('.java'):
#                     continue

#                 # --- STAGE 1: Triage with our Custom ML Model ---
#                 prediction = risk_model.predict(snippet['content'])
#                 is_high_risk = prediction[0] == 1

#                 print(f"  - Analyzing change in '{filename}': {'HIGH RISK' if is_high_risk else 'Low Risk'}")

#                 if is_high_risk:
#                     # --- STAGE 2: Deep Analysis on Flagged Code ---
#                     print(f"  - High risk detected. Performing deep analysis on '{filename}'...")
                    
#                     # Get full file content for better analysis context
#                     full_content = github_client.get_file_content(repo_name, filename, head_sha)
#                     if not full_content:
#                         continue
                    
#                     static_issues = analyze_with_checkstyle(full_content)
#                     llm_issues = analyze_with_llm(full_content)
                    
#                     all_issues = static_issues + llm_issues

#                     if not all_issues:
#                         print(f"  - Deep analysis on '{filename}' found no specific issues.")
#                         continue

#                     # --- Save Report and Post Link to GitHub ---
#                     report_id = str(uuid.uuid4())
#                     report_data = {
#                         "filename": filename,
#                         "issues": all_issues,
#                         "code": full_content
#                     }
#                     report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
#                     with open(report_path, 'w') as f:
#                         json.dump(report_data, f, indent=2)

#                     report_url = f"{BASE_URL}/report/{report_id}"
                    
#                     summary_comment = (
#                         f"**🤖 AI Code Reviewer Report for `{filename}`**\n\n"
#                         f"Our custom-trained ML model flagged a change in this file as **high-risk**.\n\n"
#                         f"A deep analysis was performed and found **{len(all_issues)} potential issue(s)**.\n\n"
#                         f"👉 **[Click here to view the full, detailed report]({report_url})**"
#                     )
                    
#                     github_client.post_comment(repo_name, pr_number, summary_comment)
#                     print(f"  ✅ Saved report {report_id} and posted link to PR.")

#         except Exception as e:
#             print(f"❌ An error occurred during the main analysis loop: {e}")

#     return jsonify({"status": "processed"}), 200

# if __name__ == '__main__':
#     # Set the BASE_URL to your ngrok URL for links to work.
#     # It's better to set this as an environment variable.
#     ngrok_url = os.getenv("NGROK_URL")
#     if ngrok_url:
#         BASE_URL = ngrok_url
#         print(f"🚀 Using NGROK_URL for report links: {BASE_URL}")

#     # Using debug=False is better for this stage to avoid the server reloading
#     app.run(port=5000, debug=False)


# import os
# import uuid
# import json
# from flask import Flask, request, jsonify, send_from_directory
# from dotenv import load_dotenv

# # Import our custom modules
# from .github_client import GithubClient
# from .ml_model import RiskModel
# from .static_analyzer import analyze_with_checkstyle
# from .llm_analyzer import analyze_with_llm

# # --- Initialization ---
# load_dotenv()

# app = Flask(__name__)

# # --- Configuration ---
# REPORTS_DIR = 'reports'
# FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
# # Default to localhost if NGROK_URL is not set
# BASE_URL = os.getenv("NGROK_URL", "http://127.0.0.1:5000") 

# # Initialize services
# try:
#     github_client = GithubClient(
#         app_id=os.getenv("GITHUB_APP_ID"),
#         private_key=os.getenv("GITHUB_PRIVATE_KEY"),
#         installation_id=os.getenv("GITHUB_INSTALLATION_ID")
#     )
#     risk_model = RiskModel()
    
#     if not os.path.exists(REPORTS_DIR):
#         os.makedirs(REPORTS_DIR)
        
#     print("✅ Models and clients initialized successfully.")
#     print(f"🚀 Using Base URL: {BASE_URL}")
# except Exception as e:
#     print(f"❌ Error during initialization: {e}")
#     github_client = None
#     risk_model = None

# # --- Routes ---

# @app.route('/report/<report_id>')
# def serve_report_page(report_id):
#     """Serves the frontend HTML page."""
#     return send_from_directory(FRONTEND_DIR, 'report.html')

# @app.route('/api/report/<report_id>')
# def get_report_data(report_id):
#     """API to fetch the JSON data for a specific report."""
#     report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
#     if os.path.exists(report_path):
#         with open(report_path, 'r') as f:
#             data = json.load(f)
#         return jsonify(data)
#     return jsonify({"error": "Report not found"}), 404

# # --- Webhook Handler ---

# @app.route("/webhook", methods=["POST"])
# def github_webhook():
#     if not github_client or not risk_model:
#         return jsonify({"status": "error", "message": "Server is not configured properly."}), 500

#     # 1. Verify Signature
#     signature = request.headers.get('X-Hub-Signature-256')
#     if not github_client.verify_signature(request.data, signature):
#         print("❌ Unauthorized: Webhook signature is not valid.")
#         return jsonify({"status": "unauthorized"}), 401
    
#     payload = request.get_json()
#     event_type = request.headers.get('X-GitHub-Event')
#     action = payload.get("action")

#     # --- DEBUG PRINT ---
#     print(f"DEBUG: Received event '{event_type}' with action '{action}'")
#     # -------------------

#     # 2. Process Pull Request Events
#     if event_type == "pull_request" and action in ["opened", "synchronize", "reopened"]:
#         pr_number = payload["pull_request"]["number"]
#         repo_name = payload["repository"]["full_name"]
#         head_sha = payload['pull_request']['head']['sha']
#         print(f"➡️ Processing Pull Request #{pr_number} for {repo_name}")

#         try:
#             # 3. Get Changed Snippets
#             diff_text = github_client.get_pr_diff(repo_name, pr_number)
#             changed_snippets = github_client.parse_diff(diff_text)
            
#             high_risk_found = False
            
#             for snippet in changed_snippets:
#                 filename = snippet['filename']
#                 if not filename.endswith('.java'):
#                     continue

#                 # --- DEBUG: PRINT THE SNIPPET ---
#                 print(f"\n--- SNIPPET START for {filename} ---")
#                 print(snippet['content'])
#                 print("--- SNIPPET END ---\n")
#                 # --------------------------------

#                 # --- STAGE 1: Triage (Risk Prediction) ---
#                 prediction = risk_model.predict(snippet['content'])
#                 is_high_risk = prediction[0] == 1

#                 print(f"  - Analyzing change in '{filename}': {'HIGH RISK' if is_high_risk else 'Low Risk'}")

#                 if is_high_risk:
#                     # --- STAGE 2: Deep Analysis ---
#                     print(f"  - High risk detected. Performing deep analysis on '{filename}'...")
                    
#                     full_content = github_client.get_file_content(repo_name, filename, head_sha)
#                     if not full_content: continue
                    
#                     static_issues = analyze_with_checkstyle(full_content)
#                     llm_issues = analyze_with_llm(full_content)
#                     all_issues = static_issues + llm_issues

#                     if all_issues:
#                         # Save Report
#                         report_id = str(uuid.uuid4())
#                         report_data = {
#                             "filename": filename,
#                             "issues": all_issues,
#                             "code": full_content
#                         }
#                         with open(os.path.join(REPORTS_DIR, f"{report_id}.json"), 'w') as f:
#                             json.dump(report_data, f, indent=2)

#                         # Post Comment
#                         report_url = f"{BASE_URL}/report/{report_id}"
#                         summary = (
#                             f"**🤖 AI Code Reviewer Report for `{filename}`**\n\n"
#                             f"Our model flagged this file as **high-risk**. Deep analysis found **{len(all_issues)} issues**.\n\n"
#                             f"👉 **[Click here to view the full report]({report_url})**"
#                         )
#                         github_client.post_comment(repo_name, pr_number, summary)
#                         print(f"  ✅ Saved report {report_id} and posted link.")
#                         high_risk_found = True

#             if not high_risk_found:
#                 print("  - No high-risk changes found in this update.")

#         except Exception as e:
#             print(f"❌ An error occurred during analysis: {e}")

#     return jsonify({"status": "processed"}), 200

# if __name__ == '__main__':
#     app.run(port=5000, debug=False)

import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Import our custom modules
from .github_client import GithubClient
from .ml_model import RiskModel
from .static_analyzer import analyze_with_checkstyle
from .llm_analyzer import analyze_with_llm

# --- Initialization ---
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
REPORTS_DIR = 'reports'
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
# Default to localhost if NGROK_URL is not set
BASE_URL = os.getenv("NGROK_URL", "http://127.0.0.1:5000") 

# Initialize services
try:
    github_client = GithubClient(
        app_id=os.getenv("GITHUB_APP_ID"),
        private_key=os.getenv("GITHUB_PRIVATE_KEY"),
        installation_id=os.getenv("GITHUB_INSTALLATION_ID")
    )
    risk_model = RiskModel()
    
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    print("✅ Models and clients initialized successfully.")
    print(f"🚀 Using Base URL: {BASE_URL}")
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    github_client = None
    risk_model = None

# --- Routes ---

@app.route('/report/<report_id>')
def serve_report_page(report_id):
    """Serves the frontend HTML page."""
    return send_from_directory(FRONTEND_DIR, 'report.html')

@app.route('/api/report/<report_id>')
def get_report_data(report_id):
    """API to fetch the JSON data for a specific report."""
    report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Report not found"}), 404

# --- Webhook Handler ---

@app.route("/webhook", methods=["POST"])
def github_webhook():
    if not github_client or not risk_model:
        return jsonify({"status": "error", "message": "Server is not configured properly."}), 500

    # 1. Verify Signature
    signature = request.headers.get('X-Hub-Signature-256')
    if not github_client.verify_signature(request.data, signature):
        print("❌ Unauthorized: Webhook signature is not valid.")
        return jsonify({"status": "unauthorized"}), 401
    
    payload = request.get_json()
    event_type = request.headers.get('X-GitHub-Event')
    action = payload.get("action")

    # --- DEBUG PRINT ---
    print(f"DEBUG: Received event '{event_type}' with action '{action}'")
    # -------------------

    # 2. Process Pull Request Events
    if event_type == "pull_request" and action in ["opened", "synchronize", "reopened"]:
        pr_number = payload["pull_request"]["number"]
        repo_name = payload["repository"]["full_name"]
        head_sha = payload['pull_request']['head']['sha']
        print(f"➡️ Processing Pull Request #{pr_number} for {repo_name}")

        try:
            # 3. Get Changed Snippets
            diff_text = github_client.get_pr_diff(repo_name, pr_number)
            changed_snippets = github_client.parse_diff(diff_text)
            
            high_risk_found = False
            
            for snippet in changed_snippets:
                filename = snippet['filename']
                if not filename.endswith('.java'):
                    continue

                # --- DEBUG: PRINT THE SNIPPET ---
                print(f"\n--- SNIPPET START for {filename} ---")
                print(snippet['content'])
                print("--- SNIPPET END ---\n")
                # --------------------------------

                # --- STAGE 1: Triage (Risk Prediction) ---
                
                # DEMO TRIGGER: Force High Risk if comment is present
                if "FORCE_RISK" in snippet['content']:
                    print(f"  ! DEMO TRIGGER: 'FORCE_RISK' found in {filename}. Forcing High Risk.")
                    is_high_risk = True
                else:
                    # Otherwise, use the ML model
                    prediction = risk_model.predict(snippet['content'])
                    is_high_risk = prediction[0] == 1

                print(f"  - Analyzing change in '{filename}': {'HIGH RISK' if is_high_risk else 'Low Risk'}")

                if is_high_risk:
                    # --- STAGE 2: Deep Analysis ---
                    print(f"  - High risk detected. Performing deep analysis on '{filename}'...")
                    
                    full_content = github_client.get_file_content(repo_name, filename, head_sha)
                    if not full_content: continue
                    
                    static_issues = analyze_with_checkstyle(full_content)
                    llm_issues = analyze_with_llm(full_content)
                    all_issues = static_issues + llm_issues

                    if all_issues:
                        # Save Report
                        report_id = str(uuid.uuid4())
                        report_data = {
                            "filename": filename,
                            "issues": all_issues,
                            "code": full_content
                        }
                        with open(os.path.join(REPORTS_DIR, f"{report_id}.json"), 'w') as f:
                            json.dump(report_data, f, indent=2)

                        # Post Comment
                        report_url = f"{BASE_URL}/report/{report_id}"
                        summary = (
                            f"**🤖 AI Code Reviewer Report for `{filename}`**\n\n"
                            f"Our model flagged this file as **high-risk**. Deep analysis found **{len(all_issues)} issues**.\n\n"
                            f"👉 **[Click here to view the full report]({report_url})**"
                        )
                        github_client.post_comment(repo_name, pr_number, summary)
                        print(f"  ✅ Saved report {report_id} and posted link.")
                        high_risk_found = True

            if not high_risk_found:
                print("  - No high-risk changes found in this update.")

        except Exception as e:
            print(f"❌ An error occurred during analysis: {e}")

    return jsonify({"status": "processed"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=False)