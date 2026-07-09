"""
Common utilities.
Author: yezhengmaolove@gmail.com
"""

import re
import sqlite3
import sys
from pathlib import Path


# =============================================================================
# SQL
# =============================================================================

RANK_SQL = "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name = 'DeviceEnvironment'"

TABLE_EXISTS_SQL = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?"


def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""
    path = Path(db_path)
    if not path.exists():
        print(f"Error: File not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix not in (".sqlite", ".db"):
        print(f"Warning: Expected .sqlite file, got: {path.suffix}", file=sys.stderr)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_rank(conn: sqlite3.Connection) -> int | None:
    """Extract global rank from TARGET_INFO_SYSTEM_ENV DeviceEnvironment."""
    try:
        row = conn.execute(RANK_SQL).fetchone()
        if row:
            m = re.search(r"(?:^|;)RANK=(\d+)(?:;|$)", row[0])
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    """True iff a table with the given name exists in the database."""
    return conn.execute(TABLE_EXISTS_SQL, (name,)).fetchone() is not None


def require_kernel_table(conn: sqlite3.Connection, db_path: str) -> None:
    """Exit with a friendly message if CUPTI_ACTIVITY_KIND_KERNEL is missing."""
    if not has_table(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        conn.close()
        print(
            f"Error: '{db_path}' has no CUPTI_ACTIVITY_KIND_KERNEL table — "
            "profile was exported without CUPTI kernel trace.",
            file=sys.stderr,
        )
        sys.exit(1)


def human_ns(ns: float) -> str:
    """Format a nanosecond duration with an SI suffix (s / ms / us / ns)."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.0f} ns"


def truncate(s: str, w: int) -> str:
    """Truncate s to at most w visible chars; overlong strings get a trailing '…'."""
    return s if len(s) <= w else s[: w - 1] + "…"
