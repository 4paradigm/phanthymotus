#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  echo "Usage: bash deploy/deploy-approval/upload-cos-evidence.sh OWNER/REPO PR [SSH_HOST_ALIAS]" >&2
}

if (( $# < 2 || $# > 3 )); then usage; exit 2; fi
repo=$1
pr_number=$2
ssh_target=${3:-root@172.21.80.40}
if [[ ! $repo =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ||
      ! $pr_number =~ ^[1-9][0-9]*$ ||
      ! $ssh_target =~ ^[A-Za-z0-9][A-Za-z0-9_.@:-]*$ ]]; then
  usage; exit 2
fi
for command in gh ssh python3 curl open; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command missing: $command" >&2
    exit 2
  fi
done
if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not logged in." >&2
  exit 2
fi
gh_login=$(gh api user --jq .login 2>/dev/null) || {
  echo "Could not read the logged-in GitHub user." >&2; exit 2;
}
if [[ ! $gh_login =~ ^[A-Za-z0-9-]+$ ]]; then
  echo "GitHub returned an invalid current user." >&2; exit 2
fi

get_context() {
  local pr_json comments_json
  pr_json=$(gh api "repos/$repo/pulls/$pr_number" 2>/dev/null) || return 1
  comments_json=$(gh api --paginate --slurp "repos/$repo/issues/$pr_number/comments" 2>/dev/null) || return 1
  python3 -c '
import json, re, sys
repo=sys.argv[1].lower(); pr_number=int(sys.argv[2])
pr=json.loads(sys.argv[3]); pages=json.loads(sys.argv[4])
comments=[c for page in pages for c in (page if isinstance(page,list) else [])]
head=(pr.get("head") or {}).get("sha", "")
base=((pr.get("base") or {}).get("repo") or {}).get("full_name", "")
if (pr.get("number") != pr_number or base.lower() != repo
        or pr.get("state") != "open" or pr.get("merged") or pr.get("draft")
        or not re.fullmatch(r"[0-9a-f]{40}", head)):
    raise SystemExit("PR must be open, non-draft, and have a valid HEAD")
marker="<!-- deploy-approval-state:v1\n"
found=[]
for comment in comments:
    body=comment.get("body", "")
    app=comment.get("performed_via_github_app")
    if (not isinstance(body,str) or body.count(marker) != 1
            or not isinstance(app,dict) or type(app.get("id")) is not int or app["id"] <= 0
            or not isinstance(comment.get("user"),dict)
            or comment["user"].get("type") != "Bot"):
        continue
    start=body.index(marker)+len(marker)
    end=body.find("\n-->", start)
    if end < 0: continue
    try: state=json.loads(body[start:end])
    except (TypeError, ValueError): continue
    if isinstance(state,dict): found.append(state)
if len(found) != 1: raise SystemExit("Could not identify one trusted Lifecycle state")
state=found[0]
round_id=state.get("round_id")
if (state.get("status") != "testing" or state.get("head_sha") != head
        or not isinstance(round_id,str)
        or not re.fullmatch(re.escape(head)+r"-[0-9]+",round_id)):
    raise SystemExit("PR is not in testing for its current HEAD and round")
print(json.dumps({"head":head,"round":round_id},separators=(",",":")))
' "$repo" "$pr_number" "$pr_json" "$comments_json"
}

if ! context=$(get_context); then
  echo "PR context check failed: expected an open, non-draft testing PR with one trusted Lifecycle state." >&2
  exit 2
fi
head=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["head"])' "$context") || exit 2
round_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["round"])' "$context") || exit 2

if ! ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
    -o ExitOnForwardFailure=yes "$ssh_target" true </dev/null >/dev/null 2>&1; then
  echo "NETWORK_UNREACHABLE: existing SSH host verification or connection failed." >&2
  exit 2
fi

existing_pids=$(lsof -nP -t -iTCP:25001 -sTCP:LISTEN 2>/dev/null || true)
if [[ -n $existing_pids ]]; then
  while IFS= read -r pid; do
    [[ $pid =~ ^[1-9][0-9]*$ ]] || { echo "Port 25001 is occupied by an unknown process." >&2; exit 2; }
    process=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if [[ $process != *ssh* || $process != *"127.0.0.1:25001:127.0.0.1:25001"* ||
          $process != *"$ssh_target"* ]]; then
      echo "Port 25001 is occupied by an unknown process; refusing to connect." >&2
      exit 2
    fi
  done <<< "$existing_pids"
else
  if ! ssh -fNT -o BatchMode=yes -o StrictHostKeyChecking=yes \
      -o ExitOnForwardFailure=yes -L 127.0.0.1:25001:127.0.0.1:25001 \
      "$ssh_target" >/dev/null 2>&1; then
    echo "NETWORK_UNREACHABLE: could not establish the local SSH tunnel." >&2
    exit 2
  fi
fi

local_health=
for _ in {1..15}; do
  local_health=$(curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:25001/healthz 2>/dev/null || true)
  if [[ $local_health == '{"status":"ok"}' ]]; then break; fi
  sleep 1
done
controller_health=$(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$ssh_target" 'docker exec deploy-approval curl -fsS --max-time 3 http://127.0.0.1:25001/healthz' 2>/dev/null || true)
if [[ $local_health != '{"status":"ok"}' || $controller_health != "$local_health" ]]; then
  echo "NETWORK_UNREACHABLE: local health check did not match the Controller on the SSH host." >&2
  exit 2
fi

# The Controller signs a page-scoped, read-only capability inside the existing
# container. Only the token is returned; configuration and COS keys stay there.
capability=$(printf '%s' "$(python3 -c 'import json,sys; print(json.dumps({"repo":sys.argv[1],"pr_number":int(sys.argv[2]),"head_sha":sys.argv[3],"round_id":sys.argv[4]},separators=(",",":")))' "$repo" "$pr_number" "$head" "$round_id")" |
  ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes "$ssh_target" \
  'docker exec -i deploy-approval python3 -c '\''import json,sys; from agents.deploy_approval.config import load_config; from agents.deploy_approval.evidence_auth import issue_capability,DOWNLOAD_PURPOSE; d=json.load(sys.stdin); c=load_config(); print(issue_capability(c,d["repo"],d["pr_number"],d["head_sha"],d["round_id"],purpose=DOWNLOAD_PURPOSE,resource="page"))'\'' ' 2>/dev/null) || {
  echo "Could not obtain the read-only page capability from the running Controller." >&2
  exit 2
}
if [[ ! $capability =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Controller returned an invalid page capability." >&2
  exit 2
fi

secret=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || exit 2
proof=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(bytes.fromhex(sys.argv[1])).hexdigest())' "$secret") || exit 2
proof_body="/authorize_evidence head=$head round=$round_id proof=$proof"

# A failed POST can have an unknown result. Always list and confirm by exact
# body and logged-in identity before deciding success; never blindly retry POST.
post_response=$(gh api "repos/$repo/issues/$pr_number/comments" \
  -f "body=$proof_body" 2>/dev/null) || post_response=
comments_after=$(gh api --paginate --slurp "repos/$repo/issues/$pr_number/comments" 2>/dev/null) || {
  unset secret proof capability
  echo "Authorization comment could not be confirmed; no upload page was opened." >&2
  exit 2
}
proof_comment_id=$(EXPECTED_BODY=$proof_body EXPECTED_LOGIN=$gh_login python3 -c '
import json,os,sys
pages=json.loads(sys.argv[1]); rows=[c for page in pages for c in (page if isinstance(page,list) else [])]
matches=[c for c in rows if c.get("body")==os.environ["EXPECTED_BODY"]
         and ((c.get("user") or {}).get("login","")).casefold()==os.environ["EXPECTED_LOGIN"].casefold()
         and type(c.get("id")) is int and c["id"]>0]
print(matches[0]["id"] if len(matches)==1 else "")
' "$comments_after")
if [[ ! $proof_comment_id =~ ^[1-9][0-9]*$ ]]; then
  unset secret proof capability
  echo "Authorization comment POST was not confirmed; no upload page was opened." >&2
  exit 2
fi

if ! context_after=$(get_context) ||
   [[ $(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["head"])' "$context_after" 2>/dev/null || true) != "$head" ||
      $(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["round"])' "$context_after" 2>/dev/null || true) != "$round_id" ]]; then
  unset secret proof capability
  echo "PR HEAD or testing round changed; refusing to open the upload page." >&2
  exit 2
fi

url="http://127.0.0.1:25001/evidence/${repo}/pulls/${pr_number}?cap=${capability}#proof=${secret}"
if ! open "$url" >/dev/null 2>&1; then
  unset secret proof capability url
  echo "Could not open the local Evidence page." >&2
  exit 2
fi
unset secret proof capability url proof_body post_response comments_after
echo "PR=$repo#$pr_number"
echo "HEAD=$head"
echo "ROUND=$round_id"
echo "UPLOAD_PAGE_OPENED=YES"
