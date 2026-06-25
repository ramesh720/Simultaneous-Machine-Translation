#!/usr/bin/env bash
# =============================================================================
# Git Workflow — Run on GCP VM
#
# Usage:
#   bash git_workflow.sh setup     # Create dev branch, configure remote
#   bash git_workflow.sh commit    # Commit all changes to dev branch
#   bash git_workflow.sh merge     # Merge dev branch into main and push
# =============================================================================
set -euo pipefail

REMOTE_URL="https://github.com/chiranjeevi-sagi/Simultaneous-Machine-Translation.git"
BRANCH="dev/gcp-multilang"

case "${1:-help}" in
    setup)
        echo "=== Setting up git ==="
        git config user.email "chiranjeevi@simul-mt.dev"
        git config user.name "Chiranjeevi Sagi"
        git remote set-url origin "${REMOTE_URL}" 2>/dev/null || git remote add origin "${REMOTE_URL}"
        git checkout -b "${BRANCH}" 2>/dev/null || git checkout "${BRANCH}"
        echo "On branch: $(git branch --show-current)"
        echo ""
        echo "Next: Make your changes, then run: bash git_workflow.sh commit"
        ;;
    commit)
        echo "=== Committing changes to ${BRANCH} ==="
        git add -A
        git status
        echo ""
        read -rp "Commit message: " MSG
        git commit -m "${MSG:-Update SimulMask + multi-lang training + frontend}"
        echo ""
        echo "Pushing to remote..."
        git push -u origin "${BRANCH}"
        echo "Done! Branch pushed to ${REMOTE_URL}"
        ;;
    merge)
        echo "=== Merging ${BRANCH} into main ==="
        git checkout main
        git pull origin main 2>/dev/null || true
        git merge "${BRANCH}" -m "Merge ${BRANCH}: SimulMask, COMET, LAAL, multi-lang, frontend"
        git push origin main
        echo ""
        echo "Done! Changes merged to main and pushed."
        ;;
    *)
        echo "Usage: bash git_workflow.sh {setup|commit|merge}"
        echo ""
        echo "  setup   - Create dev branch, configure remote"
        echo "  commit  - Stage, commit, and push all changes"
        echo "  merge   - Merge dev branch into main and push"
        ;;
esac
