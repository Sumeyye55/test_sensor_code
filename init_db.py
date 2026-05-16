"""Create or reset the fingerprint SQLite database (sensor enrollments only)."""

import argparse
import sqlite3

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_fingerprints (
    user_id TEXT PRIMARY KEY,
    keypoints BLOB NOT NULL,
    descriptors BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init_db(db_path: str, reset: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    try:
        if reset:
            conn.execute("DROP TABLE IF EXISTS user_fingerprints")
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize SQLite storage for sensor-captured fingerprints."
    )
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing enrollments (required when migrating from old .tif-based data).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_db(args.db_path, reset=args.reset)
    action = "reset and created" if args.reset else "ready"
    print(f"Database {action}: {args.db_path}")
