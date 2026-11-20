#!/usr/bin/env bash
# Fetch a public upstream ref. actions/checkout stores the fork token in
# http.https://github.com/.extraheader, and GitHub rejects that credential
# for a repository the token cannot access, including a public parent.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: sync_upstream_fetch.sh <repo> <url> <refspec>" >&2
  exit 1
fi

repo=$1
url=$2
refspec=$3

git -C "$repo" -c http.https://github.com/.extraheader= fetch --no-tags "$url" "$refspec"
