# import json
# import os
# from pydriller import Repository

# # --- Configuration ---
# # List of repository URLs to clone and mine. Add more for a better dataset.
# # Note: This will clone the full repositories, which can take up a lot of disk space.
# REPO_URLS = [
#     "https://github.com/apache/kafka.git",
#     "https://github.com/elastic/elasticsearch.git",
#     # "https://github.com/spring-projects/spring-framework.git", # Example of another large repo
#     # "https://github.com/google/guava.git", # Example of another large repo
# ]

# # Keywords to identify bug-fixing commits
# BUG_KEYWORDS = ['fix', 'bug', 'resolve', 'patch', 'defect', 'error', 'issue']

# # Output directory and file for the dataset
# DATA_DIR = 'data'
# OUTPUT_FILE = os.path.join(DATA_DIR, 'bugfix_commits.json')

# def mine_repositories():
#     """
#     Clones repositories, traverses their commit history, and builds the dataset.
#     """
#     if not os.path.exists(DATA_DIR):
#         os.makedirs(DATA_DIR)

#     all_data_pairs = []
    
#     for repo_url in REPO_URLS:
#         repo_name = repo_url.split('/')[-1].replace('.git', '')
#         print(f"\n--- Mining Repository: {repo_name} ---")

#         try:
#             # PyDriller will clone the repo if it doesn't exist locally
#             for commit in Repository(repo_url).traverse_commits():
#                 # Check if the commit message contains any of the bug-fix keywords
#                 if any(keyword in commit.msg.lower() for keyword in BUG_KEYWORDS):
                    
#                     for modification in commit.modified_files:
#                         # Process only Java files that were modified (not added/deleted)
#                         # and have both 'before' and 'after' states.
#                         if modification.filename.endswith('.java') and modification.source_code_before and modification.source_code:
                            
#                             data_point = {
#                                 "project": repo_name,
#                                 "commit_hash": commit.hash,
#                                 "file_path": modification.new_path,
#                                 "buggy_code": modification.source_code_before,
#                                 "fixed_code": modification.source_code
#                             }
#                             all_data_pairs.append(data_point)
#                             print(f"  [+] Found bug-fix pair in {commit.hash[:7]} for {modification.new_path}")

#         except Exception as e:
#             print(f"Could not process repository {repo_url}. Error: {e}")

#     # Save the collected data to a JSON file
#     with open(OUTPUT_FILE, 'w') as f:
#         json.dump(all_data_pairs, f, indent=2)
        
#     print(f"\n--- Dataset creation complete. Found {len(all_data_pairs)} pairs. ---")
#     print(f"Dataset saved to {OUTPUT_FILE}")

# if __name__ == '__main__':
#     mine_repositories()


import json
import os
from pydriller import Repository

# --- Configuration ---
REPO_URLS = [
    "https://github.com/apache/kafka.git",
    "https://github.com/elastic/elasticsearch.git",
]
BUG_KEYWORDS = ['fix', 'bug', 'resolve', 'patch', 'defect', 'error']
OUTPUT_DATASET_FILE = 'data/bugfix_changes.json'
DATA_DIR = 'data'

def mine_repositories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    all_changes = []

    for repo_url in REPO_URLS:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        print(f"--- Mining Repository: {repo_name} ---")

        try:
            for commit in Repository(repo_url, only_no_merge=True).traverse_commits():
                if any(keyword in commit.msg.lower() for keyword in BUG_KEYWORDS):
                    for modification in commit.modified_files:
                        if modification.filename.endswith('.java'):
                            diff_lines = modification.diff_parsed
                            buggy_change_lines = [line for _, line in diff_lines['deleted']]
                            fixed_change_lines = [line for _, line in diff_lines['added']]

                            change_size = len(buggy_change_lines) + len(fixed_change_lines)
                            if change_size < 5:
                                continue  # skip trivial changes

                            buggy_snippet = "\n".join(buggy_change_lines)
                            fixed_snippet = "\n".join(fixed_change_lines)

                            # Contextual risk labeling
                            if change_size < 10:
                                buggy_label = 0.5
                                fixed_label = 0.5
                            else:
                                buggy_label = 1.0
                                fixed_label = 0.0

                            all_changes.append({
                                "project": repo_name,
                                "commit_hash": commit.hash,
                                "file_path": modification.new_path,
                                "buggy_change": buggy_snippet,
                                "fixed_change": fixed_snippet,
                                "buggy_label": buggy_label,
                                "fixed_label": fixed_label
                            })
                            print(f"  [+] Found bug-fix change in {commit.hash[:7]} for {modification.new_path}")

        except Exception as e:
            print(f"Could not process repository {repo_url}. Error: {e}")

    with open(OUTPUT_DATASET_FILE, 'w') as f:
        json.dump(all_changes, f, indent=2)

    print(f"\n--- Dataset creation complete. Found {len(all_changes)} changes. ---")
    print(f"Dataset saved to {OUTPUT_DATASET_FILE}")

if __name__ == '__main__':
    mine_repositories()








