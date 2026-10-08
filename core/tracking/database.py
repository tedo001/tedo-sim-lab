"""SQLite with numbered SQL migrations (``migrations/NNNN_name.sql``).

The schema version lives in ``PRAGMA user_version``. Each migration runs in
one transaction together with its version bump, so a failed migration leaves
the database exactly as it was. A database written by a newer build is
refused rather than silently misread.

One :class:`Database` object per thread: SQLite connections are not shared
across threads here. The training worker opens its own; WAL mode lets it
write while the UI reads.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["Database", "MIGRATIONS_DIR", "MigrationError", "discover_migrations", "utc_now"]

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    """The migrations on disk or the database's version do not fit together."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """``[(1, 0001_initial.sql), ...]``; numbers must run 1, 2, 3 ... without gaps."""
    found = []
    for file in sorted(directory.glob("*.sql")):
        match = _MIGRATION.match(file.name)
        if not match:
            raise MigrationError(f"{file.name}: migrations are named NNNN_lower_case.sql")
        found.append((int(match.group(1)), file))
    numbers = [number for number, _ in found]
    if numbers != list(range(1, len(found) + 1)):
        raise MigrationError(f"migration numbers must be 1..{len(found)} without gaps: {numbers}")
    return found


class Database:
    def __init__(self, path: Path | str, *, migrations_dir: Path = MIGRATIONS_DIR) -> None:
        self.path = str(path)
        self.migrations_dir = migrations_dir
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        if self.path != ":memory:":
            self.connection.execute("PRAGMA journal_mode = WAL")

    @property
    def version(self) -> int:
        return int(self.connection.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> int:
        """Apply every migration newer than the database; return the resulting version."""
        available = discover_migrations(self.migrations_dir)
        latest = available[-1][0] if available else 0
        current = self.version
        if current > latest:
            raise MigrationError(f"this database is at schema {current}, but this build only knows "
                                 f"up to {latest}: open it with a newer version of the lab")
        for number, file in available:
            if number <= current:
                continue
            script = file.read_text(encoding="utf-8")
            try:
                self.connection.executescript(
                    f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
            except sqlite3.Error as exc:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise MigrationError(f"{file.name} failed: {exc}") from exc
        return self.version

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT``, rolled back on any exception."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        self.connection.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, params).fetchone()

    def tables(self) -> set[str]:
        rows = self.query("SELECT name FROM sqlite_master WHERE type = 'table' "
                          "AND name NOT LIKE 'sqlite_%'")
        return {row["name"] for row in rows}

    def close(self) -> None:
        self.connection.close()
