import time
import jwt
import requests
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

    def _get_jwt(self):
        """Generates a JSON Web Token for app authentication."""
        payload = {
            # Issued at time
            'iat': int(time.time()),
            # JWT expiration time (10 minutes maximum)
            'exp': int(time.time()) + (10 * 60),
            # GitHub App's identifier
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
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()['token']
        except requests.exceptions.RequestException as e:
            print(f"Error getting installation access token: {e}")
            print(f"Response body: {response.text}")
            raise

    def _authenticate(self):
        """Authenticates as the app installation and returns a PyGithub instance."""
        try:
            token = self._get_installation_access_token()
            return Github(token)
        except Exception as e:
            print(f"Failed to authenticate with GitHub: {e}")
            return None

    def get_pull_request(self, repo_full_name, pr_number):
        """Retrieves a pull request object from a given repository."""
        if not self.github_api:
            print("Error: GithubClient is not authenticated.")
            return None
        try:
            repo = self.github_api.get_repo(repo_full_name)
            return repo.get_pull(pr_number)
        except Exception as e:
            print(f"Error getting pull request '{repo_full_name}#{pr_number}': {e}")
            return None

    def post_comment(self, pull_request, comment_body):
        """Posts a comment on a given pull request."""
        if not pull_request:
            print("Error: Invalid pull request object provided for commenting.")
            return
        try:
            pull_request.create_issue_comment(comment_body)
            print(f"Successfully posted comment to PR #{pull_request.number}")
        except Exception as e:
            print(f"Error posting comment to PR #{pull_request.number}: {e}")

