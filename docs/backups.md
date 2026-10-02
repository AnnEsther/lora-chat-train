# Backups and saved Glyph Chat conversations

## What is stored where

| Data | Where | Backed up by |
|---|---|---|
| Training sessions, passages, Q&A pairs, training runs | Postgres (`postgres_data` Docker volume on the EC2 disk) | `scripts/backup_db.sh` |
| **Every Glyph Chat message** (player and Glyph) | Postgres: `glyph_conversations`, `glyph_messages` | `scripts/backup_db.sh` |
| Training datasets, configs, adapters, eval reports | S3 (uploaded by each training run) | Already in S3 |
| Live and archived adapters | `adapter_store` Docker volume | Copies in S3; EBS snapshots |
| Base model weights | `hf_cache` Docker volume | Not needed; re-downloaded from Hugging Face |

## Back up now

```bash
cd ~/lora-chat-train
./scripts/backup_db.sh        # or: make backup
```

This writes `~/lora-backups/lora_<UTC timestamp>.sql.gz`, checks the file is a valid gzip, and
uploads it to `s3://<bucket>/backups/postgres/`. The bucket is `BACKUP_S3_BUCKET` from `.env`,
or `S3_BUCKET` if that isn't set. If neither is set, it keeps only the local copy and says so.
Local copies older than 14 days are deleted (`BACKUP_KEEP_DAYS` overrides this).
If any step fails, it posts a Slack alert.

> Check the bucket in `.env` is one **you** own before relying on it. The default from
> `.env.example` is the placeholder `your-lora-bucket`:
> `aws s3api head-bucket --bucket <bucket> --expected-bucket-owner $(aws sts get-caller-identity --query Account --output text)`

## Nightly backups (cron)

```bash
mkdir -p ~/lora-backups
crontab -e
```

Add this line to back up every night at 03:15 UTC:

```
15 3 * * * /home/ubuntu/lora-chat-train/scripts/backup_db.sh >> /home/ubuntu/lora-backups/backup.log 2>&1
```

Check that it ran:

```bash
tail -20 ~/lora-backups/backup.log
ls -lh ~/lora-backups/
```

To keep S3 from growing forever, add a lifecycle rule on the bucket that expires
`backups/postgres/` objects after, say, 90 days (S3 console → bucket → Management → Lifecycle rules).

## Restore

The dump drops and recreates every object it contains, so it restores cleanly over an existing database.

```bash
cd ~/lora-chat-train
./scripts/backup_db.sh                      # safety copy of the current state first
docker compose stop backend worker          # nothing writing during the restore
gunzip -c ~/lora-backups/lora_<timestamp>.sql.gz \
  | docker compose exec -T postgres psql -U lora -d lora -v ON_ERROR_STOP=1
docker compose start backend worker
```

To restore from S3, download the file first:

```bash
aws s3 cp s3://<bucket>/backups/postgres/lora_<timestamp>.sql.gz ~/lora-backups/
```

## Whole-disk snapshots (recommended)

The database dump doesn't cover the Docker volumes for adapters, the `.env` file or the server setup.
For those, take daily EBS snapshots of the instance's volume:
AWS console → EC2 → Lifecycle Manager → Create lifecycle policy → target the instance's volume →
daily, keep 7. You can then restore the whole server from any of the last 7 days.

## Commands that destroy data

| Command | Destroys |
|---|---|
| `docker compose down -v` | All volumes: database (sessions, Q&A, **Glyph Chat history**) and adapters |
| `docker system prune --volumes` / `docker volume prune` (while stopped) | Same |
| `make reset-all` | Training sessions, Q&A pairs, runs and adapters. Glyph Chat history is **kept** |

`make prune` is safe: it only removes images, build cache and stopped containers.
Back up before any of the commands in the table.

---

## Saved Glyph Chat conversations

Every message sent to Glyph Chat (`POST /chat/direct`) is saved, along with the Glyph's reply,
before and after the reply streams:

- `glyph_conversations`: one row per conversation (`id`, `created_at`, `updated_at`).
- `glyph_messages`: `role` (`user` or `assistant`), `content`, `adapter_id`, `adapter_run_id`
  (which trained adapter answered), and `error`, which is set when a reply failed or the player
  left mid-reply. Partial replies are still saved.

The Glyph Chat page keeps the conversation id. Closing the dialog or reloading the page
continues the same conversation; the ↻ button starts a new one.
The server keeps the history: the model sees the last `GLYPH_HISTORY_MESSAGES` (default 20)
saved messages of the conversation, and ignores any history the client sends, so a player
can't fake earlier Glyph replies.

### Reading them in /glyph

The 🕘 button in the Glyph Chat dialog lists **everyone's** saved conversations, newest first,
with the first question, message count and last activity. Clicking one loads it, and the next
message continues that conversation.

How it's protected (it exposes every conversation, so this matters):

- The browser calls `/glyph/api/conversations[/<id>]`. That path is under the host nginx's
  `/glyph` location, so the site password is required.
- The Glyph Chat container's own nginx (`glyph_chat/nginx.conf`) forwards these calls to the
  backend's `/glyph/conversations` and adds `X-Admin-Key: $GLYPH_ADMIN_KEY` server-side. The
  key is never in the browser bundle (unlike `NEXT_PUBLIC_API_KEY`).
- The backend refuses those endpoints without the admin key. If `GLYPH_ADMIN_KEY` is unset,
  they are disabled (403) and the list shows an error.
- The container's port is bound to `127.0.0.1:3001`, so the proxy can't be reached directly
  on the server's public IP, bypassing the password.

Setup (once): add a random key to `.env`, letters and digits only, then rebuild:

```bash
echo "GLYPH_ADMIN_KEY=$(openssl rand -hex 32)" >> .env
docker compose build glyph_chat && docker compose up -d backend glyph_chat
```

> This lists every player's conversation to anyone with the site password. Before real players
> use `/glyph`, switch to showing only the player's own conversations.

### Reading them on the server

```bash
make glyph-chats                    # 20 most recent conversations with message counts
make glyph-chat ID=<conversation>   # one conversation, in order
make glyph-export                   # every message → glyph_chats_<date>.csv (git-ignored)
```

### Privacy

These are players' own words, stored indefinitely and included in backups. Tell players their
chats are saved (for example, a line in the chat window). Decide how long to keep them, and add
a cleanup job if needed, for example to delete conversations older than 180 days:

```bash
docker compose exec -T postgres psql -U lora -d lora -c \
  "DELETE FROM glyph_conversations WHERE updated_at < NOW() - INTERVAL '180 days';"
```

Deleting a conversation also deletes its messages.
