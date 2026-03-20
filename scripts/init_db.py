"""scripts/init_db.py — Initialise the Postgres database from schema.sql.

Usage:
    python -m scripts.init_db

Reads DATABASE_URL from environment (or .env file).
Safe to run multiple times — uses IF NOT EXISTS everywhere.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent / "infra" / "schema.sql"


def main() -> None:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    # asyncpg URL → psycopg2 URL
    db_url = os.environ["DATABASE_URL"].replace("+asyncpg", "")

    logger.info("connecting", extra={"url": db_url})
    conn = psycopg2.connect(db_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    sql = SCHEMA_PATH.read_text()
    logger.info("executing_schema", extra={"path": str(SCHEMA_PATH)})
    cur.execute(sql)

    logger.info("schema_applied — all tables ready")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
