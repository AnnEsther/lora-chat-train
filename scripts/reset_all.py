"""scripts/reset_all.py — Clear all sessions, turns, training data, adapters, outputs, and Celery queues.

Usage:
    python -m scripts.reset_all          # full reset (prompts for confirmation)
    python -m scripts.reset_all --yes    # skip confirmation prompt

This will:
1. Delete all rows from all DB tables (FK-safe order)
2. Clear adapters/current/ and adapters/history/
3. Clear outputs/sessions/ and outputs/training_runs/
4. Flush Celery task queues from Redis (cancels any pending/retrying tasks)

Run this to start completely fresh.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# Tables in FK-safe delete order (children before parents)
TABLES = [
    "deployment_events",  # → training_runs
    "model_versions",  # → training_runs
    "training_runs",  # → datasets, sessions
    "datasets",  # → sessions
    "synthesized_qa",  # → sessions (CASCADE, but explicit is safer)
    "knowledge_records",  # → sessions (CASCADE)
    "knowledge_corpus",  # → sessions
    "training_candidates",  # → sessions
    "turns",  # → sessions (CASCADE)
    "sessions",
]

# outputs/ subdirs that belong to pipeline runs — cleared on reset
OUTPUT_SUBDIRS = ["sessions", "training_runs"]


def _confirm() -> bool:
    print("\n⚠  This will permanently delete:")
    print("   • All rows from every database table")
    print("   • All adapter files in adapters/current/ and adapters/history/")
    print("   • All files in outputs/sessions/ and outputs/training_runs/")
    print("   • All pending / retrying Celery tasks in Redis\n")
    answer = input("Type 'yes' to continue: ").strip().lower()
    return answer == "yes"


def reset_database() -> dict[str, int]:
    """Delete all rows from all tables. Returns {table: rows_deleted}."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set — skipping database reset")
        return {}

    sync_url = db_url.replace("+asyncpg", "")
    conn = psycopg2.connect(sync_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    results: dict[str, int] = {}
    for table in TABLES:
        try:
            cur.execute(f"DELETE FROM {table}")
            results[table] = cur.rowcount
            logger.info(f"  {table}: deleted {cur.rowcount} rows")
        except Exception as exc:
            logger.warning(f"  {table}: skipped — {exc}")
            results[table] = -1

    cur.close()
    conn.close()
    return results


def reset_adapters() -> list[str]:
    """Remove all files/dirs from adapters/current/ and adapters/history/."""
    cleared: list[str] = []

    for subdir in ["current", "history"]:
        target = ROOT / "adapters" / subdir
        if not target.exists():
            logger.info(f"  adapters/{subdir}/: does not exist, skipping")
            continue
        for item in target.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                cleared.append(str(item.relative_to(ROOT)))
            except Exception as exc:
                logger.warning(f"  could not remove {item}: {exc}")

        logger.info(
            f"  adapters/{subdir}/: cleared ({len([c for c in cleared if f'adapters/{subdir}' in c])} items)"
        )

    return cleared


def reset_outputs() -> list[str]:
    """Remove pipeline-generated files from outputs/sessions/ and outputs/training_runs/."""
    cleared: list[str] = []
    outputs_root = ROOT / "outputs"

    for subdir in OUTPUT_SUBDIRS:
        target = outputs_root / subdir
        if not target.exists():
            logger.info(f"  outputs/{subdir}/: does not exist, skipping")
            continue

        count = 0
        for item in target.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                cleared.append(str(item.relative_to(ROOT)))
                count += 1
            except Exception as exc:
                logger.warning(f"  could not remove {item}: {exc}")

        logger.info(f"  outputs/{subdir}/: cleared ({count} items)")

    return cleared


def reset_redis_queues() -> bool:
    """Flush the Celery broker DB in Redis (cancels all pending/retrying tasks)."""
    broker_url = (
        os.environ.get("CELERY_BROKER_URL")
        or os.environ.get("REDIS_URL")
        or "redis://localhost:6379/0"
    )

    try:
        import redis as redis_lib

        client = redis_lib.from_url(broker_url)
        client.ping()  # verify connection

        # Flush only the broker DB (not the whole Redis instance)
        client.flushdb()
        logger.info(f"  Redis broker DB flushed ({broker_url})")
        return True

    except ImportError:
        logger.warning("  redis package not installed — skipping queue flush")
        return False
    except Exception as exc:
        logger.warning(f"  could not connect to Redis at {broker_url} — {exc}")
        return False


def print_summary(
    db_results: dict[str, int],
    redis_ok: bool,
    adapter_count: int,
    output_count: int,
) -> None:
    total_rows = sum(v for v in db_results.values() if v >= 0)
    print("\n" + "─" * 50)
    print("Reset complete. Summary:")
    print(f"  Database rows deleted : {total_rows}")
    print(f"  Adapter files removed : {adapter_count}")
    print(f"  Output files removed  : {output_count}")
    print(f"  Redis queues flushed  : {'yes' if redis_ok else 'no (skipped)'}")
    print("─" * 50)
    print("You can now start fresh. Run 'make up' to bring services back up.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset all LoRA Chat & Train data.")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    args = parser.parse_args()

    if not args.yes and not _confirm():
        print("Aborted.")
        sys.exit(0)

    print()
    logger.info("=== Resetting database ===")
    db_results = reset_database()

    logger.info("=== Resetting adapters ===")
    adapter_items = reset_adapters()

    logger.info("=== Resetting outputs ===")
    output_items = reset_outputs()

    logger.info("=== Flushing Celery queues ===")
    redis_ok = reset_redis_queues()

    print_summary(
        db_results,
        redis_ok,
        adapter_count=len(adapter_items),
        output_count=len(output_items),
    )


if __name__ == "__main__":
    main()
