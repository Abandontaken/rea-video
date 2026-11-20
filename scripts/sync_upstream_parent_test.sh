#!/usr/bin/env bash
# The recovery path must parse `gh repo view --json parent` (GraphQL) and
# fetch the public upstream without the checkout credential header.
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
parent_script="$root/scripts/sync_upstream_parent.sh"
fetch_script="$root/scripts/sync_upstream_fetch.sh"
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

# Captured from `gh repo view Abandontaken/rea-video --json parent` (gh 2.103.0).
# This object has owner.login and name, and no full_name.
cli_json='{"parent":{"id":"R_kgDOVcA9Ag","name":"rea-video","owner":{"id":"MDQ6VXNlcjYwODE1NTEw","login":"JuneYaooo"}}}'
if printf '%s' "$cli_json" | grep -q full_name; then
  fail 'CLI fixture unexpectedly contains full_name'
fi

got=$(printf '%s\n' "$cli_json" | "$parent_script")
assert_eq "$got" "JuneYaooo/rea-video" 'GraphQL parent slug'

rest_json='{"parent":{"full_name":"JuneYaooo/rea-video","name":"rea-video","owner":{"login":"JuneYaooo"}}}'
got=$(printf '%s\n' "$rest_json" | "$parent_script")
assert_eq "$got" "JuneYaooo/rea-video" 'REST parent slug'

set +e
printf '%s\n' '{"parent":null}' | "$parent_script" >/dev/null
null_status=$?
printf '%s\n' '{"parent":{"name":"rea-video"}}' | "$parent_script" >/dev/null
partial_status=$?
set -e
assert_eq "$null_status" 1 'null parent is rejected'
assert_eq "$partial_status" 1 'parent without an owner login is rejected'

if grep -q '\.parent\.full_name' "$workflow"; then
  fail 'workflow still reads .parent.full_name from gh repo view'
fi
grep -q 'sync_upstream_parent.sh' "$workflow" || fail 'workflow does not parse parent JSON with the tested script'
grep -q 'sync_upstream_fetch.sh' "$workflow" || fail 'workflow does not fetch upstream through the tested helper'
grep -q 'http.https://github.com/.extraheader=' "$fetch_script" || fail 'upstream fetch keeps the checkout authorization header'

# actions/checkout stores a token in http.https://github.com/.extraheader.
# A bad or scoped token makes a public upstream fetch fail until that key is cleared.
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
git init -q -b main "$probe"
git -C "$probe" config http.https://github.com/.extraheader "AUTHORIZATION: basic bm90LWEtcmVhbC10b2tlbg=="
set +e
git -C "$probe" ls-remote https://github.com/JuneYaooo/rea-video.git HEAD >/dev/null 2>"$probe/bad.err"
bad_status=$?
set -e
if [ "$bad_status" -eq 0 ]; then
  fail 'bogus checkout header unexpectedly fetched upstream'
fi
"$fetch_script" "$probe" https://github.com/JuneYaooo/rea-video.git '+refs/heads/main:refs/remotes/upstream/main'
git -C "$probe" rev-parse --verify refs/remotes/upstream/main >/dev/null

if command -v gh >/dev/null 2>&1; then
  live=$(gh repo view Abandontaken/rea-video --json parent | "$parent_script")
  assert_eq "$live" "JuneYaooo/rea-video" 'live gh repo view parent'
fi

echo "ok"
