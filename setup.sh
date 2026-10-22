#!/usr/bin/env bash
# Fetch nanoGPT at a known-good commit and link this project's files into it.
# nanoGPT's scripts expect to be run from its own folder, with datasets under
# data/ and configs under config/, so we symlink rather than copy: edits you
# make in this repo are picked up immediately, and nanoGPT itself stays untouched.
set -euo pipefail
shopt -s nullglob

NANOGPT_REPO="https://github.com/karpathy/nanoGPT.git"
NANOGPT_COMMIT="3adf61e"   # pin so results stay reproducible

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d nanoGPT ]; then
  git clone "$NANOGPT_REPO" nanoGPT
fi
git -C nanoGPT checkout -q "$NANOGPT_COMMIT"

for d in data/*/; do
  name="$(basename "$d")"
  ln -sfn "$ROOT/data/$name" "nanoGPT/data/$name"
done
for f in config/*.py; do
  ln -sf "$ROOT/$f" "nanoGPT/$f"
done
ln -sf  "$ROOT/render.py" nanoGPT/render.py
ln -sfn "$ROOT/prompts"   nanoGPT/prompts

echo "nanoGPT ready at $ROOT/nanoGPT (commit $NANOGPT_COMMIT)."
echo "Next: cd nanoGPT, then prepare data and train as described in README.md."
