# import time
# import jwt
# import requests
# import os
# from github import Github

# class GithubClient:
#     """
#     Handles all interactions with the GitHub API, including authentication
#     as a GitHub App.
#     """
#     def __init__(self, app_id, private_key, installation_id):
#         if not all([app_id, private_key, installation_id]):
#             raise ValueError("GitHub App credentials (app_id, private_key, installation_id) must be provided.")
        
#         self.app_id = app_id
#         # The private key might have escaped newlines, so we fix them
#         self.private_key = private_key.replace('\\n', '\n')
#         self.installation_id = installation_id
#         self.github_api = self._authenticate()

#     def _get_jwt(self):
#         """Generates a JSON Web Token for app authentication."""
#         payload = {
#             # Issued at time
#             'iat': int(time.time()),
#             # JWT expiration time (10 minutes maximum)
#             'exp': int(time.time()) + (10 * 60),
#             # GitHub App's identifier
#             'iss': self.app_id
#         }
#         return jwt.encode(payload, self.private_key, algorithm='RS256')

#     def _get_installation_access_token(self):
#         """Gets a temporary access token for a specific installation."""
#         jwt_token = self._get_jwt()
#         headers = {
#             'Authorization': f'Bearer {jwt_token}',
#             'Accept': 'application/vnd.github.v3+json',
#         }
#         url = f'https://api.github.com/app/installations/{self.installation_id}/access_tokens'
        
#         try:
#             response = requests.post(url, headers=headers)
#             response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
#             return response.json()['token']
#         except requests.exceptions.RequestException as e:
#             print(f"Error getting installation access token: {e}")
#             print(f"Response body: {response.text}")
#             raise

#     def _authenticate(self):
#         """Authenticates as the app installation and returns a PyGithub instance."""
#         try:
#             token = self._get_installation_access_token()
#             return Github(token)
#         except Exception as e:
#             print(f"Failed to authenticate with GitHub: {e}")
#             return None

#     def get_pull_request(self, repo_full_name, pr_number):
#         """Retrieves a pull request object from a given repository."""
#         if not self.github_api:
#             print("Error: GithubClient is not authenticated.")
#             return None
#         try:
#             repo = self.github_api.get_repo(repo_full_name)
#             return repo.get_pull(pr_number)
#         except Exception as e:
#             print(f"Error getting pull request '{repo_full_name}#{pr_number}': {e}")
#             return None

#     def post_comment(self, pull_request, comment_body):
#         """Posts a comment on a given pull request."""
#         if not pull_request:
#             print("Error: Invalid pull request object provided for commenting.")
#             return
#         try:
#             pull_request.create_issue_comment(comment_body)
#             print(f"Successfully posted comment to PR #{pull_request.number}")
#         except Exception as e:
#             print(f"Error posting comment to PR #{pull_request.number}: {e}")
import time
import jwt
import requests
import hmac
import hashlib
import os
from github import Github

class GithubClient:
    """
    Handles all interactions with the GitHub API, including authentication
    as a GitHub App.
    """
    def __init__(self, app_id, private_key, installation_id):
        if not all([app_id, private_key, installation_id]):
            raise ValueError("GitHub App credentials (app_id, private_key, installation_id) must be provided.")
        
        self.app_id = app_id
        # The private key might have escaped newlines, so we fix them
        self.private_key = private_key.replace('\\n', '\n')
        self.installation_id = installation_id
        self.github_api = self._authenticate()
        self.webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")

    # --- Authentication Methods ---

    def _get_jwt(self):
        """Generates a JSON Web Token for app authentication."""
        payload = {
            'iat': int(time.time()),
            'exp': int(time.time()) + (10 * 60), # 10 minutes expiration
            'iss': self.app_id
        }
        return jwt.encode(payload, self.private_key, algorithm='RS256')

    def _get_installation_access_token(self):
        """Gets a temporary access token for a specific installation."""
        jwt_token = self._get_jwt()
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Accept': 'application/vnd.github.v3+json',
        }
        url = f'https://api.github.com/app/installations/{self.installation_id}/access_tokens'
        
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status() 
            return response.json()['token']
        except requests.exceptions.RequestException as e:
            print(f"Error getting installation access token: {e}")
            raise

    def _authenticate(self):
        """Authenticates as the app installation and returns a PyGithub instance."""
        try:
            token = self._get_installation_access_token()
            return Github(token)
        except Exception as e:
            print(f"Failed to authenticate with GitHub: {e}")
            return None

    # --- Security Methods ---

    def verify_signature(self, payload_body, signature_header):
        """
        Verify that the webhook payload was sent from GitHub.
        """
        if not signature_header or not self.webhook_secret:
            return False
        
        secret = self.webhook_secret.encode('utf-8')
        hash_object = hmac.new(secret, msg=payload_body, digestmod=hashlib.sha256)
        expected_signature = "sha256=" + hash_object.hexdigest()
        
        return hmac.compare_digest(expected_signature, signature_header)

    # --- Data Fetching Methods ---

    def get_pr_diff(self, repo_full_name, pr_number):
        """Fetches the raw diff of a pull request."""
        try:
            # We use requests directly here to get the raw diff easily
            token = self._get_installation_access_token()
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3.diff' 
            }
            url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error getting PR diff: {e}")
            return ""

    def parse_diff(self, diff_text):
        """
        Parses a git diff text into a list of changed code snippets.
        Each item is a dict with 'filename' and 'content' (the added/changed lines).
        """
        changes = []
        current_file = None
        current_content = []
        
        for line in diff_text.splitlines():
            if line.startswith('diff --git'):
                # Save previous file if exists
                if current_file and current_content:
                    changes.append({'filename': current_file, 'content': "\n".join(current_content)})
                
                # Start new file
                current_content = []
                # Extract filename (simple parsing, assumes standard git diff format)
                parts = line.split(' ')
                if len(parts) >= 4:
                    current_file = parts[2][2:] # remove 'a/' prefix
            
            elif line.startswith('+++') or line.startswith('---') or line.startswith('index'):
                continue
            
            elif line.startswith('+') and not line.startswith('+++'):
                # This is an added line of code
                current_content.append(line[1:]) # Remove the '+'
        
        # Save the last file
        if current_file and current_content:
            changes.append({'filename': current_file, 'content': "\n".join(current_content)})
            
        return changes

    def get_file_content(self, repo_full_name, file_path, ref):
        """Fetches the full content of a file at a specific commit."""
        try:
            repo = self.github_api.get_repo(repo_full_name)
            content_file = repo.get_contents(file_path, ref=ref)
            return content_file.decoded_content.decode('utf-8')
        except Exception as e:
            print(f"Error getting file content: {e}")
            return None

    # --- Interaction Methods ---

    def post_comment(self, repo_full_name, pr_number, comment_body):
        """Posts a comment on a pull request."""
        try:
            repo = self.github_api.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            pr.create_issue_comment(comment_body)
            print(f"Successfully posted comment to PR #{pr_number}")
        except Exception as e:
            print(f"Error posting comment: {e}")