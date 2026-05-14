#!/usr/bin/env bash
set -euo pipefail

echo "==> Pulling latest changes..."
git pull origin main

echo "==> Syncing nginx config..."
sudo cp infra/nginx/lora-chat /etc/nginx/sites-available/lora-chat

# Create symlink if it doesn't exist yet
if [ ! -L /etc/nginx/sites-enabled/lora-chat ]; then
  sudo ln -s /etc/nginx/sites-available/lora-chat /etc/nginx/sites-enabled/lora-chat
fi

echo "==> Testing nginx config..."
sudo nginx -t

echo "==> Reloading nginx..."
sudo systemctl reload nginx

echo "==> Rebuilding and restarting containers..."
docker compose pull --quiet || true
docker compose up --build -d

echo "==> Done."