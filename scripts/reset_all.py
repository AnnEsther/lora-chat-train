"""scripts/reset_all.py — Clear all sessions, turns, training data, and adapters.

Usage:
    python -m scripts.reset_all

This will:
1. Delete all sessions, turns, candidates, datasets, and training runs from the database
2. Delete all adapter files in adapters/current/ and adapters/history/
3. Optionally reset the database schema (add new columns)

Run this to start fresh.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    # Database URL
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set in environment")
        return

    # Connect to database
    sync_db_url = db_url.replace("+asyncpg", "")
    conn = psycopg2.connect(sync_db_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Delete all data from tables (in correct order due to foreign keys)
    tables = [
        "training_runs",
        "datasets",
        "synthesized_qa",
        "knowledge_records",
        "knowledge_corpus",
        "training_candidates",
        "turns",
        "sessions",
    ]

    for table in tables:
        try:
            cur.execute(f"DELETE FROM {table}")
            logger.info(f"Deleted all rows from {table}")
        except Exception as e:
            logger.warning(f"Could not delete {table}: {e}")

    # Add system_prompt and training_system_prompt columns if they don't exist
    try:
        cur.execute("""
            ALTER TABLE sessions 
            ADD COLUMN IF NOT EXISTS system_prompt TEXT
        """)
        cur.execute("""
            ALTER TABLE sessions 
            ADD COLUMN IF NOT EXISTS training_system_prompt TEXT
        """)
        logger.info("Ensured system_prompt and training_system_prompt columns exist")
    except Exception as e:
        logger.warning(f"Could not add columns: {e}")

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database cleared successfully")

    # Clear adapters directory
    root = Path(__file__).parent.parent
    adapter_dir = root / "adapters"
    current_dir = adapter_dir / "current"
    history_dir = adapter_dir / "history"

    if current_dir.exists():
        for item in current_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        logger.info(f"Cleared {current_dir}")

    if history_dir.exists():
        for item in history_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
        logger.info(f"Cleared {history_dir}")

    logger.info("Reset complete! You can now start fresh.")


if __name__ == "__main__":
    main()
