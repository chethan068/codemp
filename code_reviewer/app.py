import os
import tempfile
from flask import Flask, request, jsonify

# Import all the necessary components of our application
from .github_client import github_client
from .ml_model import risk_model
from .static_analyzer import static_analyzer
from .llm_analyzer import llm_analyzer

app = Flask(__name__)

# --- Main Webhook Endpoint ---
@app.route("/webhook", methods=["POST"])
def github_webhook():
    """
    Main endpoint to receive webhook events from the GitHub App.
    """
    try:
        payload = request.get_json()
        
        # Verify that the request is a valid pull request event
        if "pull_request" not in payload or payload["action"] not in ["opened", "synchronize"]:
            return jsonify({"status": "event ignored, not an opened or updated pull request"}), 200

        pr_data = payload["pull_request"]
        repo_name = payload["repository"]["full_name"]
        pr_number = pr_data["number"]
        
        print(f"\n--- Received PR Event for {repo_name} #{pr_number} ---")

        # Get the list of modified files from the pull request
        pr_files = github_client.get_pr_files(repo_name, pr_number)
        
        for file in pr_files:
            # We only want to analyze Java files that were modified, not deleted
            if not file['filename'].endswith('.java') or file['status'] == 'removed':
                continue

            print(f"Analyzing file: {file['filename']}")
            
            # --- STAGE 1: RISK ASSESSMENT (The Gatekeeper) ---
            file_content = github_client.get_file_content(file['raw_url'])
            if not file_content:
                continue
            
            risk_prediction = risk_model.predict(file_content)

            if risk_prediction == 1: # '1' means high-risk
                print(f"  - STAGE 1: File '{file['filename']}' flagged as HIGH-RISK.")
                
                # --- STAGE 2: DEEP ANALYSIS (The Specialist) ---
                # We need to save the file content to a temporary file for Checkstyle
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.java', delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                print(f"  - STAGE 2: Running deep analysis...")
                static_issues = static_analyzer.analyze(temp_file_path)
                llm_issues = llm_analyzer.analyze(file_content)

                os.unlink(temp_file_path) # Clean up the temporary file

                # Combine and format the results
                if static_issues or llm_issues:
                    report = format_report(file['filename'], static_issues, llm_issues)
                    github_client.post_comment(repo_name, pr_number, report)
                    print(f"  - REPORT: Posted a detailed review for '{file['filename']}'.")
                else:
                    print("  - REPORT: Deep analysis found no specific issues to report.")

            else:
                print(f"  - STAGE 1: File '{file['filename']}' assessed as LOW-RISK. Skipping deep analysis.")

        return jsonify({"status": "processed"}), 200

    except Exception as e:
        print(f"An error occurred in the webhook handler: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def format_report(filename, static_issues, llm_issues):
    """
    Formats the findings from both analyzers into a clean Markdown report.
    """
    report = [f"### 🚨 AI-Powered Analysis for `{filename}`\n"]
    report.append("Our automated analysis has identified this file as high-risk and found the following potential issues:\n")
    
    # Add Static Analysis Issues
    if static_issues:
        report.append("---")
        report.append("#### 📝 Static Analysis Findings (Checkstyle)")
        for issue in static_issues:
            report.append(f"- **Line {issue['line']} ({issue['severity']}):** {issue['message']}")
    
    # Add LLM Analysis Issues
    if llm_issues:
        report.append("---")
        report.append("#### 🧠 Deep Analysis Findings (AI)")
        for issue in llm_issues:
            report.append(f"- **Line {issue['line']} ({issue['severity']}):** {issue['message']}\n  - *AI Explanation:* {issue['description']}")

    return "\n".join(report)

if __name__ == '__main__':
    # This allows running the Flask app directly for local testing
    # Note: For production, use a proper WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000)

