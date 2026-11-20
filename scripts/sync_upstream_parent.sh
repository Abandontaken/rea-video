#!/usr/bin/env bash
# Print owner/name for a `gh repo view --json parent` document on stdin.
# gh's JSON is the GraphQL parent (owner.login + name). The REST parent
# object instead carries full_name. Exit 1 when neither form is present.
set -euo pipefail

json=$(cat)
python3 - "$json" <<'PY'
import json
import sys

doc = json.loads(sys.argv[1])
parent = doc.get("parent") if isinstance(doc, dict) else None
if not isinstance(parent, dict):
    sys.exit(1)
owner = parent.get("owner")
login = owner.get("login") if isinstance(owner, dict) else None
name = parent.get("name")
if isinstance(login, str) and login and isinstance(name, str) and name:
    slug = f"{login}/{name}"
else:
    full = parent.get("full_name")
    if not isinstance(full, str) or not full.strip():
        sys.exit(1)
    slug = full.strip()
if "/" not in slug or slug.startswith("/") or slug.endswith("/") or " " in slug:
    sys.exit(1)
print(slug)
PY
