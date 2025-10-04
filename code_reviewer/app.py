import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Import the class names
from .github_client import GithubClient
from .ml_model import RiskModel

# --- Initialization ---
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize our custom classes
try:
    # --- THIS IS THE FIX ---
    # We now read the secrets from the environment and pass them to the client.
    github_client = GithubClient(
        app_id=os.getenv("GITHUB_APP_ID"),
        private_key=os.getenv("GITHUB_PRIVATE_KEY"),
        installation_id=os.getenv("GITHUB_INSTALLATION_ID")
    )
    risk_model = RiskModel()
    print("✅ Models and clients initialized successfully.")
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    github_client = None
    risk_model = None


@app.route("/webhook", methods=["POST"])
def github_webhook():
    """
    Main webhook endpoint to receive events from the GitHub App.
    """
    if not github_client or not risk_model:
        return jsonify({"status": "error", "message": "Server is not configured properly."}), 500

    # --- Security Verification ---
    signature = request.headers.get('X-Hub-Signature-256')
    if not github_client.verify_signature(request.data, signature):
        print("❌ Unauthorized: Webhook signature is not valid.")
        return jsonify({"status": "unauthorized"}), 401
    
    payload = request.get_json()
    event_type = request.headers.get('X-GitHub-Event')

    if event_type == "pull_request" and payload.get("action") in ["opened", "synchronize"]:
        pr_number = payload["pull_request"]["number"]
        repo_name = payload["repository"]["full_name"]
        print(f"➡️ Received pull_request '{payload['action']}' event for {repo_name} #{pr_number}")

        try:
            # --- Analysis Logic ---
            # Using the 'diff' is more efficient than getting all files
            diff_text = github_client.get_pr_diff(repo_name, pr_number)
            changed_snippets = github_client.parse_diff(diff_text)
            
            for snippet in changed_snippets:
                filename = snippet['filename']
                # Only analyze .java files
                if not filename.endswith('.java'):
                    continue

                # --- Make a Prediction with our Model for each changed snippet ---
                prediction = risk_model.predict(snippet['content'])
                is_high_risk = prediction[0] == 1 # 1 means buggy/high-risk

                print(f"  - Analyzing change in '{filename}': {'HIGH RISK' if is_high_risk else 'Low Risk'}")

                if is_high_risk:
                    comment = (
                        f"**AI Code Reviewer Report for a change in `{filename}`**\n\n"
                        f"Our custom-trained ML model (Accuracy: 0.97) has flagged a code change in this file as **high-risk**. "
                        f"The change exhibits patterns that are statistically similar to past bugs.\n\n"
                        f"A deeper manual review of this specific change is recommended."
                    )
                    github_client.post_comment(repo_name, pr_number, comment)
                    print(f"  ✅ Posted high-risk comment for '{filename}'.")

        except Exception as e:
            print(f"❌ An error occurred during analysis: {e}")

    return jsonify({"status": "processed"}), 200


if __name__ == '__main__':
    # Setting debug=False is better for this stage to avoid double-loading
    app.run(port=5000, debug=False)

