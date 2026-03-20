#!/bin/bash
# Create GitHub repo and push this project.
# Run from project root: bash scripts/github_setup.sh
# Prerequisites: gh CLI installed and authenticated, or create repo manually on github.com

set -e
cd "$(dirname "$0")/.."

REPO_NAME="${REPO_NAME:-fraud-synth-icml}"
GITHUB_USER="${GITHUB_USER:-}"  # Set your GitHub username, or use gh's default

echo "=== Step 1: Add all files (respecting .gitignore) ==="
git add -A
git status

echo ""
echo "=== Step 2: Commit ==="
git commit -m "Initial commit: fraud detection with synthetic oversampling (SMOTE, CTGAN, TabDDPM)" || true

echo ""
echo "=== Step 3: Create GitHub repo and push ==="
if command -v gh &>/dev/null; then
  echo "Using GitHub CLI (gh)..."
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
  echo "Done! Repo: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')"
else
  echo "GitHub CLI (gh) not found. Create the repo manually:"
  echo "  1. Go to https://github.com/new"
  echo "  2. Create repo named: $REPO_NAME"
  echo "  3. Do NOT initialize with README (we have one)"
  echo "  4. Run: git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
  echo "  5. Run: git push -u origin master"
  echo "  (or: git push -u origin main   if your default branch is main)"
fi
