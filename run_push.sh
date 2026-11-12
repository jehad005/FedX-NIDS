#!/usr/bin/env bash
# Quick helper to stage, commit, and push changes to GitHub
set -e

MSG="${1:-Auto update FedX-NIDS}"
cd "$(dirname "$0")"

git add -A
if git diff --staged --quiet; then
    echo "Working tree is clean. Nothing new to commit."
else
    git commit -m "$MSG"
fi

git push origin main
echo "✓ Successfully pushed to https://github.com/jehad005/FedX-NIDS (main)"
