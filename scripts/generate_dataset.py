import os
import re
import csv
from git import Repo

REPO_PATH = os.path.abspath(".")
OUTPUT_FILE = "diffs_labeled.csv"
BUG_PATTERNS = [r"TODO", r"FIXME", r"null", r"==", r"System\.exit", r"print\(.*\)"]

def extract_features(diff_text):
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    warnings = sum(len(re.findall(p, diff_text)) for p in BUG_PATTERNS)
    return added, removed, warnings

def generate_dataset(repo_path, output_file, max_commits=200):
    repo = Repo(repo_path)
    commits = list(repo.iter_commits("main", max_count=max_commits))
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["diff_text", "added_loc", "removed_loc", "lint_warnings", "label"])
        
        for commit in commits:
            if not commit.parents:
                continue
            parent = commit.parents[0]
            diff_index = parent.diff(commit, create_patch=True)
            
            for diff in diff_index:
                if diff.diff:
                    diff_text = diff.diff.decode("utf-8", errors="ignore")
                    added, removed, warnings = extract_features(diff_text)
                    label = 1 if warnings > 0 else 0  # dummy labeling
                    writer.writerow([diff_text[:5000], added, removed, warnings, label])

    print(f"✅ Dataset saved to {output_file} with {len(commits)} commits processed.")

if __name__ == "__main__":
    generate_dataset(REPO_PATH, OUTPUT_FILE)
