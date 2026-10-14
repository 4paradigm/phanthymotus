# Deploy Approval Agent — Deployment Guide

## Configuration Architecture

Deploy Approval uses **four distinct configuration layers**. Understanding the
separation is critical to safe, repeatable deployments across release upgrades.

### 1. GitHub Versioned Source Code

| What | Where | Managed by |
|------|-------|------------|
| Python source (`agents/deploy_approval/`) | Git repository | Source control |
| `docker-compose.yml` | Git repository | Source control |
| `deploy.sh` | Git repository | Source control |
| `Dockerfile` | Git repository | Source control |

These files are versioned in Git and change only through pull requests.
They define **how** the service runs but **not what** machines it deploys or
**which** cloud bucket stores evidence.

### 2. Server Private Persistent Configuration

| What | Host Path | Container Mount | Stored in Git? |
|------|-----------|-----------------|----------------|
| Machine policy | `$DEPLOY_APPROVAL_CONFIG_DIR/machines.yaml` | `/run/deploy-approval/machines.yaml` (ro) | **NO** (.gitignore) |
| COS secrets | `$DEPLOY_APPROVAL_CONFIG_DIR/secrets.yaml` | `/run/deploy-approval/secrets.yaml` (ro) | **NO** (.gitignore) |

The configuration directory is specified by the **`DEPLOY_APPROVAL_CONFIG_DIR`**
environment variable. This makes the deployment portable across development,
test, and production machines without hardcoding any server-specific path.

- **Persisted across releases**: When a new GitHub SHA is deployed, the
  container reads the same directory regardless of which `deploy.sh` or
  `docker-compose.yml` version is running.
- **Private**: Never committed to Git. Example templates exist as
  `machines.example.yaml` and `secrets.example.yaml` for reference only.
- **Read-only inside the container**: Mounted with `:ro` flag. The service
  cannot modify them at runtime.
- **Required**: `DEPLOY_APPROVAL_CONFIG_DIR` must be set to an absolute
  directory. It cannot be omitted, relative, or fall back to a hardcoded path.

**Critical**: `DEPLOY_APPROVAL_CONFIG_DIR` is the **only** way to specify the
configuration directory. Hardcoded paths like `/var/lib/deploy-approval` are
NOT used by the source code. Each machine sets its own directory value.

### 3. GitHub PR Hidden State

| What | Where | Persisted in |
|------|-------|-------------|
| Lifecycle comment body | GitHub PR comment (#5908741249 style) | GitHub API |
| Hidden state block | `<!-- deploy-approval-state:v1 ... -->` | GitHub PR comment |
| Visible history | `<!-- deploy-approval-visible-history:start -->` | GitHub PR comment |

This state is managed entirely by the Deploy Approval Agent through GitHub PR
comments. It is **not stored in any local database** on the server. The service
is stateless — all state lives in GitHub.

### 4. COS Evidence Storage

| What | Where | Managed by |
|------|-------|-----------|
| Evidence archive | Tencent COS bucket | Deploy Approval Agent (upload) |
| Object key pattern | `phanthymotus_pr/.../pr-{N}/evidence-{SHA}.log.gz` | COS |
| SHA256 reference | Hidden state `cos.sha256` field | GitHub PR comment |

COS artifacts are uploaded as part of the record_test lifecycle phase. The
COS `object_key`, `sha256`, and `size` are persisted in the PR's hidden state
block, providing verifiable evidence that can be validated independently.

## Deployment

### Prerequisites

1. Set the configuration directory:
   ```bash
   export DEPLOY_APPROVAL_CONFIG_DIR="/path/to/your/config"
   mkdir -p "$DEPLOY_APPROVAL_CONFIG_DIR"
   ```

2. Place private configuration files in that directory:
   ```bash
   # Machine policy (copy from example, then fill in real values)
   cp machines.example.yaml "$DEPLOY_APPROVAL_CONFIG_DIR/machines.yaml"
   # COS secrets (copy from example, then fill in real values)
   cp secrets.example.yaml "$DEPLOY_APPROVAL_CONFIG_DIR/secrets.yaml"
   ```

3. File permissions (owner-only, no group/world access):
   ```bash
   chmod 600 "$DEPLOY_APPROVAL_CONFIG_DIR/machines.yaml"
   chmod 600 "$DEPLOY_APPROVAL_CONFIG_DIR/secrets.yaml"
   ```

4. GitHub App credentials set as environment variables:
   ```bash
   export GITHUB_APP_ID="..."
   export GITHUB_INSTALLATION_ID="..."
   export GITHUB_APP_PRIVATE_KEY_FILE="/absolute/path/to/key.pem"
   ```

### Running

```bash
# Pre-flight checks + start
./deploy.sh up

# Rebuild from source + restart
./deploy.sh rebuild

# Stop / start / restart
./deploy.sh stop
./deploy.sh start
./deploy.sh restart

# Tear down
./deploy.sh down

# Check health
./deploy.sh status
./deploy.sh logs
```

### Pre-flight Validation

`deploy.sh` validates **before** starting or rebuilding the container:

- `DEPLOY_APPROVAL_CONFIG_DIR` is set and is an absolute path
- Files exist at `$DEPLOY_APPROVAL_CONFIG_DIR/machines.yaml` and
  `$DEPLOY_APPROVAL_CONFIG_DIR/secrets.yaml`
- Files are regular files (not symlinks)
- Files are owned by the invoking user
- Files have no group/world permissions
- `machines.yaml` is a valid version-1 mapping with real values (no `<placeholder>`)
- `secrets.yaml` has valid COS region/bucket format, real secret values,
  correct review_comment_trust settings
- GitHub App inputs are valid

**If any check fails, the container is NOT started or modified.**

### Cross-Release Persistence

When you deploy a new GitHub SHA, the container continues to read the
same `$DEPLOY_APPROVAL_CONFIG_DIR` without any reconfiguration:

```bash
export DEPLOY_APPROVAL_CONFIG_DIR="/path/to/your/config"
./deploy.sh rebuild
```

Your machine configuration and COS secrets are preserved automatically because
they live at the configured directory, not inside the Git repository.

## Security

- **No secrets in Git**: `machines.yaml` and `secrets.yaml` are in `.gitignore`.
  Example files contain `<placeholder>` values and must never be committed.
- **Read-only mounts**: Both config files are mounted `:ro` inside the container.
- **File permissions**: deploy.sh rejects files with group/world access.
- **No symlink attacks**: deploy.sh rejects symlinked config files.
- **No database**: All runtime state lives in GitHub PR comments and COS.
- **Container security**: `read_only: true`, `cap_drop: ALL`, `no-new-privileges`.
