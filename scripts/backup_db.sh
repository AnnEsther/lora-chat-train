#!/usr/bin/env bash
# =============================================================================
# scripts/backup_db.sh — Back up the Postgres database.
#
# Covers everything in Postgres: training sessions, passages, Q&A pairs,
# training runs and every saved Glyph Chat conversation.
#
# - Dumps the database and keeps a gzipped copy in ~/lora-backups
#   (files older than BACKUP_KEEP_DAYS, default 14, are removed).
# - Uploads the dump to s3://$BACKUP_S3_BUCKET/backups/postgres/
#   (BACKUP_S3_BUCKET defaults to S3_BUCKET from .env). The upload runs in the
#   worker container, which already has boto3 and the AWS credentials.
# - Posts a Slack alert if any step fails.
#
# Usage:   ./scripts/backup_db.sh        (or: make backup)
# Nightly: see docs/backups.md for the cron line.
# Restore: see docs/backups.md.
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

env_value() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- || true; }

BACKUP_DIR="${BACKUP_DIR:-$HOME/lora-backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-14}"
BUCKET="${BACKUP_S3_BUCKET:-$(env_value BACKUP_S3_BUCKET)}"
BUCKET="${BUCKET:-$(env_value S3_BUCKET)}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FILE="$BACKUP_DIR/lora_$STAMP.sql.gz"
KEY="backups/postgres/lora_$STAMP.sql.gz"

on_failure() {
  local msg="Database backup failed at line $1 ($STAMP) — local copy may be missing; check $BACKUP_DIR/backup.log"
  echo "ERROR: $msg" >&2
  rm -f "$FILE.tmp"
  docker compose exec -T worker python -c "
import sys
from shared.slack_notifier import backend_error
backend_error('nightly database backup', sys.argv[1], sys.argv[2])
" "$msg" "$STAMP" </dev/null >/dev/null 2>&1 || true
}
trap 'on_failure $LINENO' ERR

mkdir -p "$BACKUP_DIR"
echo "[$STAMP] dumping database → $FILE"
# --clean --if-exists: the dump drops and recreates objects, so it restores
# cleanly over an existing database
docker compose exec -T postgres pg_dump -U lora --clean --if-exists lora | gzip -9 > "$FILE.tmp"
gzip -t "$FILE.tmp"   # fail now rather than discover a corrupt backup later
mv "$FILE.tmp" "$FILE"
echo "dump ok ($(du -h "$FILE" | cut -f1))"

if [ -n "$BUCKET" ]; then
  echo "uploading → s3://$BUCKET/$KEY"
  docker compose exec -T worker python -c "
import os, sys, boto3
s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION') or None)
s3.upload_fileobj(sys.stdin.buffer, sys.argv[1], sys.argv[2])
" "$BUCKET" "$KEY" < "$FILE"
  echo "upload ok"
else
  echo "WARNING: no BACKUP_S3_BUCKET or S3_BUCKET in .env — kept the local copy only"
fi

find "$BACKUP_DIR" -name 'lora_*.sql.gz' -mtime +"$KEEP_DAYS" -delete
echo "done — $(ls "$BACKUP_DIR"/lora_*.sql.gz | wc -l) local backup(s) in $BACKUP_DIR"
