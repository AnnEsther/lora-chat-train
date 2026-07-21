#!/usr/bin/env bash
# =============================================================================
# How to fix it
# Step 1 — Check how much space you have:
# df -h /var/lib/containerd
# df -h /var/lib/docker   # if using Docker daemon instead of containerd
# df -h /                 # overall root partition
# Step 2 — Free up Docker/containerd cache:
# Remove unused images, containers, volumes, build cache
# docker system prune -af --volumes
# Check how much build cache specifically is taking up
# docker system df
# Step 3 — If still not enough, remove old images manually:
# List all images sorted by size
# docker images --format "{{.Size}}\t{{.Repository}}:{{.Tag}}" | sort -rh | head -20
# Remove specific old/unused ones
# docker rmi <image_id>
# Step 4 — Retry the build:
# bash start.sh

# start.sh — LoRA Chat & Train
# Idempotent setup + deploy script. Safe to run on a fresh EC2 or for updates.
# Usage:
#   First time : ./start.sh --setup
#   Updates    : ./start.sh
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✔${NC}  $*"; }
info() { echo -e "${BLUE}▸${NC}  $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
fail() { echo -e "${RED}✖${NC}  $*"; exit 1; }
header() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }

SETUP_MODE=false
if [[ "${1:-}" == "--setup" ]]; then
  SETUP_MODE=true
fi

# Flag set in Section 3.5 when the user switches models.
# Acted on in Section 6 AFTER docker compose down releases the volume.
HF_CACHE_SHOULD_CLEAR=false

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# =============================================================================
# SECTION 1 — SYSTEM DEPENDENCIES (--setup only)
# =============================================================================
if $SETUP_MODE; then
  header "Installing system dependencies"

  sudo apt-get update -qq

  # Docker
  if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    ok "Docker installed — you may need to log out and back in for group to apply"
  else
    ok "Docker already installed ($(docker --version | cut -d' ' -f3 | tr -d ','))"
  fi

  # Docker Compose plugin
  if ! docker compose version &>/dev/null 2>&1; then
    info "Installing Docker Compose plugin..."
    sudo apt-get install -y docker-compose-plugin
    ok "Docker Compose installed"
  else
    ok "Docker Compose already installed ($(docker compose version --short))"
  fi

  # Nginx
  if ! command -v nginx &>/dev/null; then
    info "Installing nginx..."
    sudo apt-get install -y nginx
    ok "Nginx installed"
  else
    ok "Nginx already installed"
  fi

  # Certbot (for SSL)
  if ! command -v certbot &>/dev/null; then
    info "Installing Certbot..."
    sudo apt-get install -y certbot python3-certbot-nginx
    ok "Certbot installed"
  else
    ok "Certbot already installed"
  fi

  # htpasswd (for basic auth)
  if ! command -v htpasswd &>/dev/null; then
    info "Installing apache2-utils (htpasswd)..."
    sudo apt-get install -y apache2-utils
    ok "htpasswd installed"
  else
    ok "htpasswd already installed"
  fi

  # NVIDIA drivers check + nvidia-container-toolkit
  header "Checking GPU & NVIDIA container toolkit"

  if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — NVIDIA drivers may not be installed"
    warn "Install drivers with: sudo apt-get install -y nvidia-driver-535"
    warn "Then reboot and run this script again"
    warn "Skipping nvidia-container-toolkit install"
  else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    ok "GPU detected: $GPU_NAME"

    if ! dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
      info "Installing nvidia-container-toolkit..."
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      sudo apt-get update -qq
      sudo apt-get install -y nvidia-container-toolkit
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker
      ok "nvidia-container-toolkit installed and Docker restarted"
    else
      ok "nvidia-container-toolkit already installed"
    fi
  fi

  # Setup basic auth if .htpasswd doesn't exist
  header "Basic auth setup"
  if [ ! -f /etc/nginx/.htpasswd ]; then
    warn ".htpasswd not found at /etc/nginx/.htpasswd"
    echo -e "${YELLOW}Enter a username for the web UI basic auth:${NC}"
    read -r HTPASSWD_USER
    sudo htpasswd -c /etc/nginx/.htpasswd "$HTPASSWD_USER"
    ok "Created /etc/nginx/.htpasswd for user: $HTPASSWD_USER"
  else
    ok "/etc/nginx/.htpasswd already exists — skipping"
  fi
fi

# =============================================================================
# SECTION 2 — ALWAYS: Pre-flight checks
# =============================================================================
header "Pre-flight checks"

# Docker running
if ! docker info &>/dev/null; then
  fail "Docker daemon is not running. Start it with: sudo systemctl start docker"
fi
ok "Docker daemon is running"

# Docker Compose
if ! docker compose version &>/dev/null 2>&1; then
  fail "Docker Compose not found. Run: ./start.sh --setup"
fi
ok "Docker Compose available"

# GPU check
if command -v nvidia-smi &>/dev/null; then
  if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM)"
  else
    fail "nvidia-smi found but GPU check failed — driver issue? Try: sudo nvidia-smi"
  fi
else
  warn "nvidia-smi not found — GPU unavailable. Model server will run on CPU (slow)."
  warn "If you expect a GPU, install drivers and run: ./start.sh --setup"
  echo -e "${YELLOW}Continue without GPU? [y/N]:${NC} " && read -r CONTINUE_NO_GPU
  [[ "$CONTINUE_NO_GPU" =~ ^[Yy]$ ]] || fail "Aborted — install GPU drivers first"
fi

# =============================================================================
# SECTION 3 — ALWAYS: .env check
# =============================================================================
header "Environment configuration"

if [ ! -f "$REPO_DIR/.env" ]; then
  if [ -f "$REPO_DIR/.env.example" ]; then
    warn ".env not found — copying from .env.example"
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    fail ".env created from example. Fill in all values then re-run: ./start.sh"
  else
    fail ".env not found and no .env.example to copy from"
  fi
fi

# Check required vars are set and non-empty
REQUIRED_VARS=(
  "DATABASE_URL"
  "REDIS_URL"
  "HF_TOKEN"
#   "HF_ENDPOINT_URL"
  "BASE_MODEL"
  "MODEL_SERVER_URL"
  "NEXT_PUBLIC_API_URL"
  "NEXT_PUBLIC_MODEL_SERVER_URL"
  "EXTERNAL_SITE_ORIGIN"
)

MISSING_VARS=()
for VAR in "${REQUIRED_VARS[@]}"; do
  VALUE=$(grep -E "^${VAR}=" "$REPO_DIR/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
  if [[ -z "$VALUE" || "$VALUE" == "your_"* || "$VALUE" == "<"* ]]; then
    MISSING_VARS+=("$VAR")
  fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
  fail "The following required .env variables are missing or unfilled:\n$(printf '  • %s\n' "${MISSING_VARS[@]}")\n\nEdit .env and re-run."
fi
ok ".env present and required variables set"

# Optional vars — warn but don't block
OPTIONAL_VARS=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "S3_BUCKET" "SLACK_WEBHOOK_URL" "CHAT_API_KEY")
for VAR in "${OPTIONAL_VARS[@]}"; do
  VALUE=$(grep -E "^${VAR}=" "$REPO_DIR/.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true)
  if [[ -z "$VALUE" ]]; then
    warn "$VAR not set — related feature will be disabled"
  fi
done

# =============================================================================
# SECTION 3.5 — ALWAYS: Model selection
# =============================================================================
header "Model selection"

# Preset model list — add/remove entries here to change the menu
PRESET_MODELS=(
  "cognitivecomputations/dolphin-2.9-mistral-7b-v2"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mistral-7B-v0.3"
)

# Read current value from .env
CURRENT_MODEL=$(grep -E "^BASE_MODEL=" "$REPO_DIR/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'" | tr -d '[:space:]')
CURRENT_MODEL="${CURRENT_MODEL:-unset}"

# Skip prompt in non-interactive / CI environments
if [[ "${CI:-false}" == "true" || ! -t 0 ]]; then
  ok "Non-interactive mode — keeping current model: ${CURRENT_MODEL}"
else
  echo ""
  echo -e "  ${BOLD}Current model:${NC} ${CURRENT_MODEL}"
  echo ""
  echo -e "  ${BOLD}Available models:${NC}"
  for i in "${!PRESET_MODELS[@]}"; do
    LABEL=""
    if [[ "${PRESET_MODELS[$i]}" == "$CURRENT_MODEL" ]]; then
      LABEL=" ${GREEN}[active]${NC}"
    fi
    echo -e "    $((i+1))) ${PRESET_MODELS[$i]}${LABEL}"
  done
  echo -e "    $((${#PRESET_MODELS[@]}+1))) Enter a custom HuggingFace model ID"
  echo ""
  echo -e "  ${YELLOW}Select [1-$((${#PRESET_MODELS[@]}+1))], or press Enter to keep current:${NC} " && read -r MODEL_CHOICE

  if [[ -z "$MODEL_CHOICE" ]]; then
    ok "Keeping current model: ${CURRENT_MODEL}"
    NEW_MODEL=""
  elif [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]] && \
       [[ "$MODEL_CHOICE" -ge 1 ]] && \
       [[ "$MODEL_CHOICE" -le "${#PRESET_MODELS[@]}" ]]; then
    NEW_MODEL="${PRESET_MODELS[$((MODEL_CHOICE-1))]}"
  elif [[ "$MODEL_CHOICE" == "$((${#PRESET_MODELS[@]}+1))" ]]; then
    echo -e "  ${YELLOW}Enter HuggingFace model ID (e.g. org/model-name):${NC} " && read -r NEW_MODEL
    if [[ -z "$NEW_MODEL" ]]; then
      ok "No model entered — keeping current: ${CURRENT_MODEL}"
      NEW_MODEL=""
    fi
  else
    warn "Invalid selection '${MODEL_CHOICE}' — keeping current model: ${CURRENT_MODEL}"
    NEW_MODEL=""
  fi

  if [[ -n "$NEW_MODEL" && "$NEW_MODEL" != "$CURRENT_MODEL" ]]; then
    # Update BASE_MODEL in .env (replace the active line, leave commented-out lines untouched)
    sed -i "s|^BASE_MODEL=.*|BASE_MODEL=${NEW_MODEL}|" "$REPO_DIR/.env"
    # Update or insert HF_ENDPOINT_MODEL
    if grep -qE "^HF_ENDPOINT_MODEL=" "$REPO_DIR/.env"; then
      sed -i "s|^HF_ENDPOINT_MODEL=.*|HF_ENDPOINT_MODEL=${NEW_MODEL}|" "$REPO_DIR/.env"
    else
      echo "HF_ENDPOINT_MODEL=${NEW_MODEL}" >> "$REPO_DIR/.env"
    fi
    ok "Model updated: ${NEW_MODEL}"

    # Schedule hf_cache wipe — cannot remove the volume here because containers
    # may still be running and holding it open. The actual rm happens in Section 6
    # after docker compose down has released all volume mounts.
    HF_CACHE_SHOULD_CLEAR=true
    info "Old model weights (${CURRENT_MODEL}) will be cleared before rebuild (~14 GB freed)"
  elif [[ -n "$NEW_MODEL" ]]; then
    ok "Selected model is already active: ${CURRENT_MODEL}"
  fi
fi

# =============================================================================
# SECTION 4 — ALWAYS: Pull latest code
# =============================================================================
header "Pulling latest code"

if git rev-parse --is-inside-work-tree &>/dev/null; then
  CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  info "On branch: $CURRENT_BRANCH"
  git pull origin "$CURRENT_BRANCH"
  ok "Repository up to date"
else
  warn "Not inside a git repository — skipping git pull"
fi

# =============================================================================
# SECTION 5 — ALWAYS: Nginx config
# =============================================================================
header "Nginx configuration"

NGINX_SRC="$REPO_DIR/infra/nginx/lora-chat"
NGINX_DEST="/etc/nginx/sites-available/lora-chat"
NGINX_ENABLED="/etc/nginx/sites-enabled/lora-chat"

if [ ! -f "$NGINX_SRC" ]; then
  fail "Nginx config not found at $NGINX_SRC — add it with: sudo cp /etc/nginx/sites-available/lora-chat infra/nginx/lora-chat && git add infra/nginx/lora-chat"
fi

sudo cp "$NGINX_SRC" "$NGINX_DEST"
ok "Copied $NGINX_SRC → $NGINX_DEST"

if [ ! -L "$NGINX_ENABLED" ]; then
  sudo ln -s "$NGINX_DEST" "$NGINX_ENABLED"
  ok "Created symlink: $NGINX_ENABLED"
else
  ok "Symlink already exists: $NGINX_ENABLED"
fi

# Remove default nginx site if it exists (causes conflicts)
if [ -L /etc/nginx/sites-enabled/default ]; then
  sudo rm /etc/nginx/sites-enabled/default
  warn "Removed default nginx site (was conflicting)"
fi

info "Testing nginx config..."
if sudo nginx -t 2>&1; then
  sudo systemctl reload nginx
  ok "Nginx reloaded successfully"
else
  fail "Nginx config test failed — check $NGINX_SRC for errors"
fi

# Ensure nginx starts on reboot
sudo systemctl enable nginx &>/dev/null
ok "Nginx enabled on boot"

# =============================================================================
# SECTION 5.5 — ALWAYS: Free up disk space before build
# =============================================================================
header "Disk cleanup"

DISK_FREE=$(df / | awk 'NR==2 {print $4}')
DISK_FREE_GB=$(( DISK_FREE / 1024 / 1024 ))
info "Free disk space before cleanup: ${DISK_FREE_GB}GB"

# Remove ALL unused images (not just dangling) and all stale build cache.
# This reclaims the previous lora-chat-train-model_server / worker image layers
# (~8–15 GB) that accumulate across rebuilds.
info "Pruning all unused Docker images and build cache..."
docker image prune -af
docker builder prune -af

DISK_FREE_AFTER=$(df / | awk 'NR==2 {print $4}')
DISK_FREE_AFTER_GB=$(( DISK_FREE_AFTER / 1024 / 1024 ))
ok "Disk space after cleanup: ${DISK_FREE_AFTER_GB}GB free"

# The model_server image requires ~20 GB free to build and unpack.
# If we are still below that threshold, escalate: full system prune first,
# then wipe hf_cache (model re-downloads automatically on next start).
if [ "$DISK_FREE_AFTER_GB" -lt 20 ]; then
  warn "Less than 20GB free (${DISK_FREE_AFTER_GB}GB) — running full system prune..."
  docker system prune -af
  DISK_FREE_FINAL=$(df / | awk 'NR==2 {print $4}')
  DISK_FREE_FINAL_GB=$(( DISK_FREE_FINAL / 1024 / 1024 ))
  ok "Disk space after full prune: ${DISK_FREE_FINAL_GB}GB free"

  if [ "$DISK_FREE_FINAL_GB" -lt 20 ]; then
    warn "Still only ${DISK_FREE_FINAL_GB}GB free — clearing hf_cache volume..."
    warn "Model weights will re-download on first start (~10–20 min depending on network)"
    docker volume rm lora-chat-train_hf_cache 2>/dev/null || true
    DISK_FREE_FINAL=$(df / | awk 'NR==2 {print $4}')
    DISK_FREE_FINAL_GB=$(( DISK_FREE_FINAL / 1024 / 1024 ))
    ok "Disk space after hf_cache clear: ${DISK_FREE_FINAL_GB}GB free"
  fi
fi

# =============================================================================
# SECTION 6 — ALWAYS: Build and start Docker services
# =============================================================================
header "Starting Docker services"

# Bring down cleanly if updating (not first run)
if docker compose ps --quiet 2>/dev/null | grep -q .; then
  info "Stopping existing containers..."
  docker compose down --remove-orphans
fi

# Clear hf_cache now that containers are stopped and the volume is no longer mounted.
# This runs only when the user switched models in Section 3.5.
if [ "$HF_CACHE_SHOULD_CLEAR" = "true" ]; then
  info "Clearing cached model weights for old model..."
  docker volume rm lora-chat-train_hf_cache 2>/dev/null || true
  ok "hf_cache cleared — new model weights will download on first start (~10–20 min)"
fi

info "Building and starting all services..."
docker compose build backend
docker compose build worker

# Hard disk guard — the model_server image (CUDA + PyTorch + ML libs) requires
# ~20 GB to build and unpack. Fail fast here rather than after 400 seconds.
DISK_PRE_MODEL=$(df / | awk 'NR==2 {print $4}')
DISK_PRE_MODEL_GB=$(( DISK_PRE_MODEL / 1024 / 1024 ))
if [ "$DISK_PRE_MODEL_GB" -lt 20 ]; then
  warn "Only ${DISK_PRE_MODEL_GB}GB free before model_server build — need at least 20GB"
  warn "Clearing hf_cache volume to recover space (model weights will re-download)..."
  docker volume rm lora-chat-train_hf_cache 2>/dev/null || true
  DISK_POST_CLEAR=$(df / | awk 'NR==2 {print $4}')
  DISK_POST_CLEAR_GB=$(( DISK_POST_CLEAR / 1024 / 1024 ))
  ok "Disk after hf_cache clear: ${DISK_POST_CLEAR_GB}GB free"
  if [ "$DISK_POST_CLEAR_GB" -lt 20 ]; then
    fail "Still only ${DISK_POST_CLEAR_GB}GB free after cleanup — cannot safely build model_server.\nExpand your root volume to at least 150GB in the AWS Console, then re-run ./start.sh"
  fi
fi

docker compose build model_server
docker compose build frontend
docker compose build glyph_chat
docker compose up -d

ok "All containers started"

# =============================================================================
# SECTION 7 — ALWAYS: Wait for services and health checks
# =============================================================================
header "Health checks"

wait_for() {
  local NAME=$1
  local URL=$2
  local RETRIES=${3:-20}
  local WAIT=${4:-3}

  info "Waiting for $NAME..."
  for i in $(seq 1 $RETRIES); do
    if curl -sf "$URL" &>/dev/null; then
      ok "$NAME is healthy"
      return 0
    fi
    sleep "$WAIT"
  done
  warn "$NAME did not respond after $((RETRIES * WAIT))s — check: docker compose logs $NAME"
  return 1
}

# Give postgres and redis a moment to initialise
sleep 5

wait_for "postgres"     "http://localhost:8000/health" 30 3   # backend proxy check
wait_for "backend API"  "http://localhost:8000/health" 20 3
wait_for "model server" "http://localhost:8001/health" 20 3
wait_for "frontend"     "http://localhost:3000"        20 3

# =============================================================================
# SECTION 8 — ALWAYS: DB initialisation (safe to run multiple times)
# =============================================================================
header "Database initialisation"

info "Applying schema (safe to re-run)..."
if docker compose exec -T postgres psql -U lora -d lora -c "\dt" 2>/dev/null \
    | grep -q "sessions"; then
  ok "Database schema already applied"
else
  docker compose exec -T postgres psql -U lora -d lora \
    -f /docker-entrypoint-initdb.d/01_schema.sql 2>/dev/null || true
  ok "Schema applied"
fi

# =============================================================================
# SECTION 9 — ALWAYS: Final status summary
# =============================================================================
header "Deployment summary"

echo ""
docker compose ps
echo ""

# GPU in containers
if command -v nvidia-smi &>/dev/null; then
  info "GPU access in worker container:"
  docker compose exec -T worker nvidia-smi --query-gpu=name,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/   /' || warn "  Could not query GPU inside worker"
fi

echo ""
ok "Deployment complete"
echo ""
echo -e "  ${BOLD}Frontend:${NC}     https://train.anratelier.com"
echo -e "  ${BOLD}Backend API:${NC}  https://train.anratelier.com/api/health"
echo -e "  ${BOLD}Model server:${NC} https://train.anratelier.com/model/health"
echo ""
echo -e "  ${BOLD}Logs:${NC}         docker compose logs -f"
echo -e "  ${BOLD}Status:${NC}       docker compose ps"
echo ""