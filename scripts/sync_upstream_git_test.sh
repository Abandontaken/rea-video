#!/usr/bin/env bash
# Regression tests for non-force recovery when upstream rewrites history.
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
script="$root/scripts/sync_upstream_git.sh"
workflow="$root/.github/workflows/sync-upstream.yml"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_eq() {
  local got=$1 expected=$2 msg=$3
  if [ "$got" != "$expected" ]; then
    fail "$msg: got $(printf %q "$got") expected $(printf %q "$expected")"
  fi
}

init_repo() {
  git init -q -b main "$1"
  git -C "$1" config user.email test@example.com
  git -C "$1" config user.name test
  git -C "$1" config commit.gpgsign false
}

commit_file() {
  local repo=$1 path=$2 content=$3 message=$4
  mkdir -p "$repo/$(dirname -- "$path")"
  printf '%s\n' "$content" > "$repo/$path"
  git -C "$repo" add -- "$path"
  git -C "$repo" commit -qm "$message"
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# --- rewritten upstream, fork-only workflow file, no overlapping edits ---
up="$tmp/rewrite-up"
fork="$tmp/rewrite-fork"
init_repo "$up"
commit_file "$up" README base base
commit_file "$up" README v2 v2
commit_file "$up" keep.txt keep keep
v2=$(git -C "$up" rev-parse HEAD)
v2_tree=$(git -C "$up" rev-parse 'HEAD^{tree}')

git clone -q "$up" "$fork"
git -C "$fork" config user.email test@example.com
git -C "$fork" config user.name test
git -C "$fork" config commit.gpgsign false
commit_file "$fork" .github/workflows/sync-upstream.yml 'name: Sync upstream' 'fork workflow'
rm -f "$fork/keep.txt"
git -C "$fork" add -u -- keep.txt
git -C "$fork" commit -qm 'fork deletes keep.txt'
fork_before=$(git -C "$fork" rev-parse HEAD)
workflow_blob=$(git -C "$fork" rev-parse 'HEAD:.github/workflows/sync-upstream.yml')

git -C "$up" checkout -q --orphan rewritten
git -C "$up" rm -rfq .
git -C "$up" checkout "$v2" -- .
git -C "$up" commit -qm 'rewritten equivalent snapshot'
assert_eq "$(git -C "$up" rev-parse 'HEAD^{tree}')" "$v2_tree" 'rewritten snapshot keeps the same tree'
commit_file "$up" app.txt 'new feature' 'upstream feature'
git -C "$up" branch -D main >/dev/null
git -C "$up" branch -M main
upstream_tip=$(git -C "$up" rev-parse HEAD)

git -C "$fork" fetch -q "$up" "+refs/heads/main:refs/remotes/upstream/main"
set +e
"$script" "$fork" main refs/remotes/upstream/main
rewrite_status=$?
set -e
assert_eq "$rewrite_status" 0 'rewritten history with only fork-only paths should sync'
new=$(git -C "$fork" rev-parse HEAD)
assert_eq "$(git -C "$fork" rev-parse "${new}^1")" "$fork_before" 'merge keeps the previous fork commit as first parent'
assert_eq "$(git -C "$fork" rev-parse "${new}^2")" "$upstream_tip" 'merge records the upstream tip as second parent'
assert_eq "$(git -C "$fork" rev-parse 'HEAD:.github/workflows/sync-upstream.yml')" "$workflow_blob" 'fork workflow file is preserved'
assert_eq "$(git -C "$fork" show 'HEAD:app.txt')" 'new feature' 'new upstream file is taken'
assert_eq "$(git -C "$fork" show 'HEAD:README')" v2 'rewritten upstream content is taken'
if git -C "$fork" cat-file -e 'HEAD:keep.txt' 2>/dev/null; then
  fail 'fork-only deletion was lost'
fi
git -C "$fork" merge-base --is-ancestor "$fork_before" HEAD
git -C "$fork" merge-base --is-ancestor "$v2" HEAD

# --- real overlap stays a conflict and does not move the branch ---
up="$tmp/conflict-up"
fork="$tmp/conflict-fork"
init_repo "$up"
commit_file "$up" README base base
git clone -q "$up" "$fork"
git -C "$fork" config user.email test@example.com
git -C "$fork" config user.name test
git -C "$fork" config commit.gpgsign false
commit_file "$fork" README 'fork edit' 'fork edit'
commit_file "$up" README 'upstream edit' 'upstream edit'
git -C "$fork" fetch -q "$up" "+refs/heads/main:refs/remotes/upstream/main"
conflict_before=$(git -C "$fork" rev-parse HEAD)
set +e
"$script" "$fork" main refs/remotes/upstream/main
conflict_status=$?
set -e
assert_eq "$conflict_status" 2 'overlapping fork and upstream edits are a conflict'
assert_eq "$(git -C "$fork" rev-parse HEAD)" "$conflict_before" 'conflict leaves the fork branch unchanged'

# --- upstream changed the fork-only workflow path ---
up="$tmp/workflow-up"
fork="$tmp/workflow-fork"
init_repo "$up"
commit_file "$up" README base base
git clone -q "$up" "$fork"
git -C "$fork" config user.email test@example.com
git -C "$fork" config user.name test
git -C "$fork" config commit.gpgsign false
commit_file "$fork" .github/workflows/sync-upstream.yml 'fork workflow' 'fork workflow'
base=$(git -C "$up" rev-parse HEAD)
base_tree=$(git -C "$up" rev-parse 'HEAD^{tree}')
git -C "$up" checkout -q --orphan rewritten
git -C "$up" rm -rfq .
git -C "$up" checkout "$base" -- .
git -C "$up" commit -qm 'rewritten base'
assert_eq "$(git -C "$up" rev-parse 'HEAD^{tree}')" "$base_tree" 'workflow fixture rewrite keeps the tree'
commit_file "$up" .github/workflows/sync-upstream.yml 'upstream workflow' 'upstream workflow'
git -C "$up" branch -D main >/dev/null
git -C "$up" branch -M main
git -C "$fork" fetch -q "$up" "+refs/heads/main:refs/remotes/upstream/main"
workflow_before=$(git -C "$fork" rev-parse HEAD)
set +e
"$script" "$fork" main refs/remotes/upstream/main
workflow_status=$?
set -e
assert_eq "$workflow_status" 2 'upstream edits to a fork-only path are a conflict'
assert_eq "$(git -C "$fork" rev-parse HEAD)" "$workflow_before" 'workflow conflict leaves the branch unchanged'

# The workflow must keep failing real conflicts, must not force-push, and must
# use the recovery script only for the diverging-changes error from gh.
grep -q 'sync_upstream_git.sh' "$workflow" || fail 'workflow does not call the recovery script'
grep -q "can't sync because there are diverging changes" "$workflow" || fail 'recovery is not limited to the diverging-changes error'
grep -q 'Fail job if sync failed' "$workflow" || fail 'workflow no longer fails the job after a real sync failure'
grep -q 'Upstream sync needs manual merge' "$workflow" || fail 'workflow no longer opens the conflict issue'
if grep -E -q 'git push([^#\n]*)--force|repo sync([^#\n]*)--force|reset --hard' "$workflow" "$script"; then
  fail 'sync path uses force or hard reset'
fi

echo "ok"
