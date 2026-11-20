#!/usr/bin/env bash
# Join a diverged upstream branch without discarding fork-only changes.
# Exit 0 when the branch is updated or already contains upstream.
# Exit 2 when fork-only paths overlap upstream edits; the branch is left unchanged.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: sync_upstream_git.sh <repo> <branch> <upstream-ref>" >&2
  exit 1
fi

repo=$1
branch=$2
upstream_ref=$3

cd "$repo"
start=$(git rev-parse --verify "$branch^{commit}")
upstream=$(git rev-parse --verify "$upstream_ref^{commit}")

if git merge-base --is-ancestor "$upstream" "$start"; then
  exit 0
fi
if git merge-base --is-ancestor "$start" "$upstream"; then
  git update-ref "refs/heads/$branch" "$upstream" "$start"
  exit 0
fi

declare -A fork_trees=()
while read -r tree; do
  [ -n "$tree" ] || continue
  fork_trees["$tree"]=1
done < <(git log --format='%T' "$start")

u_eq=
while read -r sha tree; do
  if [ -n "${fork_trees[$tree]+x}" ]; then
    u_eq=$sha
    break
  fi
done < <(git log --format='%H %T' "$upstream")

if [ -z "$u_eq" ]; then
  echo "No upstream commit matches a tree already in the fork; refusing to sync." >&2
  exit 2
fi

declare -A upstream_paths=()
while IFS= read -r -d '' path; do
  upstream_paths["$path"]=1
done < <(git diff --name-only --no-renames -z "$u_eq" "$upstream")

overlap=0
while IFS= read -r -d '' path; do
  if [ -n "${upstream_paths[$path]+x}" ]; then
    echo "Fork-only path also changed upstream: $path" >&2
    overlap=1
  fi
done < <(git diff --name-only --no-renames -z "$u_eq" "$start")
if [ "$overlap" -ne 0 ]; then
  exit 2
fi

if ! git config user.email >/dev/null; then
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git config user.name "github-actions[bot]"
fi

wt=$(mktemp -d)
rmdir "$wt"
cleanup() {
  git worktree remove --force "$wt" >/dev/null 2>&1 || true
  rm -rf "$wt"
}
trap cleanup EXIT
git worktree add --detach "$wt" "$start" >/dev/null
git -C "$wt" read-tree --reset -u "$upstream^{tree}"

while IFS= read -r -d '' status && IFS= read -r -d '' path; do
  case "$status" in
    A|M|T)
      git -C "$wt" checkout "$start" -- "$path"
      ;;
    D)
      git -C "$wt" rm -f --ignore-unmatch -- "$path" >/dev/null
      ;;
    *)
      echo "Unsupported diff status '$status' for $path" >&2
      exit 1
      ;;
  esac
done < <(git diff --name-status --no-renames -z "$u_eq" "$start")

tree=$(git -C "$wt" write-tree)
message="Merge upstream ${branch}, keeping fork-only changes"
new=$(git commit-tree "$tree" -p "$start" -p "$upstream" -m "$message")
git update-ref "refs/heads/$branch" "$new" "$start"
