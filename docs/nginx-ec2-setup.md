# Nginx Setup & Update Guide (EC2 — train.anratelier.com)

## Overview

This project uses nginx as a reverse proxy on the EC2 instance to route all browser traffic through a single HTTPS endpoint (`https://train.anratelier.com`). Nginx sits in front of three internal services:

| Path prefix | Internal service | Port |
|-------------|-----------------|------|
| `/` | Frontend (Next.js) | 3000 |
| `/api/` | Backend API (FastAPI) | 8000 |
| `/model/` | Model server (hf_serve) | 8001 |

This avoids mixed-content and CORS errors — the browser never talks directly to ports 8000 or 8001.

---

## File Location

The nginx config for this project lives at:

```
/etc/nginx/sites-available/lora-chat
```

It is symlinked into the active sites directory:

```
/etc/nginx/sites-enabled/lora-chat -> /etc/nginx/sites-available/lora-chat
```

---

## Opening and Editing the File

The file is owned by root, so you must use `sudo`:

```bash
sudo nano /etc/nginx/sites-available/lora-chat
```

> **Tip:** If you prefer vim: `sudo vim /etc/nginx/sites-available/lora-chat`

---

## Full Config Reference

This is the complete, correct config for this project. Use it as the source of truth when making changes:

```nginx
server {
    server_name train.anratelier.com;

    # ── Frontend (Next.js on port 3000) ───────────────────────────────────────
    location / {
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # ── Backend API + SSE streaming (FastAPI on port 8000) ───────────────────
    location /api/ {
        rewrite ^/api(/.*)$ $1 break;
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;        # required for SSE streaming
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # ── Model server (hf_serve on port 8001) ─────────────────────────────────
    location /model/ {
        rewrite ^/model(/.*)$ $1 break;
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # ── SSL (managed by Certbot — do not edit manually) ──────────────────────
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/train.anratelier.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/train.anratelier.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}

# ── HTTP → HTTPS redirect (managed by Certbot) ───────────────────────────────
server {
    if ($host = train.anratelier.com) {
        return 301 https://$host$request_uri;
    }
    listen 80;
    server_name train.anratelier.com;
    return 404;
}
```

---

## Applying Changes

After every edit, always validate before reloading — a bad config will take down the site:

```bash
# 1. Test the config syntax
sudo nginx -t

# 2. If the test passes, reload nginx (zero-downtime)
sudo systemctl reload nginx

# 3. If something is broken and you need a full restart
sudo systemctl restart nginx
```

Expected output from a passing `nginx -t`:
```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

---

## Environment Variables (.env)

After updating nginx, make sure the frontend environment variables match the proxy paths. These are in the `.env` file at the project root:

```bash
# Browser-facing URLs — must be HTTPS and use the proxy paths
NEXT_PUBLIC_API_URL=https://train.anratelier.com/api
NEXT_PUBLIC_MODEL_SERVER_URL=https://train.anratelier.com/model
```

> **Important:** `NEXT_PUBLIC_*` variables are baked into the Next.js bundle at **build time**, not runtime. You must rebuild the frontend container any time these values change.

---

## Rebuilding the Frontend After .env Changes

```bash
# From the project root on the EC2 instance
docker compose up --build -d frontend
```

To confirm the new environment variables are active in the running container:

```bash
docker compose exec frontend env | grep NEXT_PUBLIC
```

---

## Checking nginx Status

```bash
# Is nginx running?
sudo systemctl status nginx

# Watch live access/error logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# See only errors from the last 50 lines
sudo tail -50 /var/log/nginx/error.log | grep -i error
```

---

## Adding Basic Auth (htpasswd)

The frontend location block uses `auth_basic`. To add or update a user:

```bash
# Install apache2-utils if not already present
sudo apt install apache2-utils -y

# Create the file and add a user (you will be prompted for a password)
sudo htpasswd -c /etc/nginx/.htpasswd your_username

# Add a second user (omit -c to avoid overwriting the file)
sudo htpasswd /etc/nginx/.htpasswd another_user

# Verify the file exists
cat /etc/nginx/.htpasswd
```

After changing htpasswd, reload nginx:

```bash
sudo systemctl reload nginx
```

---

## SSL Certificate Renewal

SSL is managed by Certbot. Certificates auto-renew via a cron job or systemd timer. To manually trigger renewal:

```bash
sudo certbot renew --dry-run   # test renewal without actually renewing
sudo certbot renew             # renew for real
```

> **Do not manually edit** the `ssl_certificate`, `ssl_certificate_key`, or `include` lines in the nginx config — Certbot manages these.

---

## Troubleshooting

### Site shows 502 Bad Gateway
A container is down. Check:
```bash
docker compose ps
docker compose logs -f backend    # or frontend / model_server
```

### Mixed content errors in browser console
`NEXT_PUBLIC_API_URL` or `NEXT_PUBLIC_MODEL_SERVER_URL` is set to `http://` instead of `https://`. Fix `.env` and rebuild the frontend.

### CORS errors with status `null`
The browser cannot reach the backend at the specified URL at all — usually means the nginx proxy block is missing or wrong, or the Docker container isn't running. Check `docker compose ps` and `sudo nginx -t`.

### nginx won't reload after an edit
Run `sudo nginx -t` to see the exact syntax error with line number before reloading.

### Port 8000 or 8001 directly blocked
That is intentional. The EC2 security group should only expose ports 80 and 443. All traffic must go through nginx.

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-05-01 | Initial nginx setup guide created | — |