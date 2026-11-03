#!/usr/bin/env bash
# Copy code and processed data to a GPU server and record the git commit.
# Usage: REMOTE=user@host REMOTE_DIR=~/Projects/motorphen-dev scripts/remote/sync.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${REMOTE:?set REMOTE=user@host}"
REMOTE_DIR=${REMOTE_DIR:-'~/Projects/motorphen-dev'}
if [[ -n "$(git status --porcelain -- src scripts pyproject.toml requirements*.txt)" ]]; then
  echo "uncommitted changes in code; commit first so COMMIT is meaningful" >&2; exit 1
fi
ssh -o BatchMode=yes "$REMOTE" "mkdir -p $REMOTE_DIR/data/processed $REMOTE_DIR/results/e1 $REMOTE_DIR/checkpoints $REMOTE_DIR/logs"
rsync -a --delete src scripts tests pyproject.toml requirements.txt requirements-gpu.txt "$REMOTE:$REMOTE_DIR/"
rsync -a data/processed/vc2023.npz data/processed/vc2023_events.tsv "$REMOTE:$REMOTE_DIR/data/processed/"
git rev-parse HEAD | ssh -o BatchMode=yes "$REMOTE" "cat > $REMOTE_DIR/COMMIT"
echo "synced $(git rev-parse --short HEAD) to $REMOTE:$REMOTE_DIR"
