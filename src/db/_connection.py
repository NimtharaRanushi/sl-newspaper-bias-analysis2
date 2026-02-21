"""Database connection management base class."""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from src.config import load_config


class DatabaseConnection:
    """PostgreSQL database connection manager."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()["database"]
        self._conn = None

    def connect(self):
        """Establish database connection."""
        self._conn = psycopg2.connect(
            host=self.config["host"],
            port=self.config["port"],
            dbname=self.config["name"],
            user=self.config["user"],
            password=self.config["password"]
        )
        self._conn.autocommit = False
        return self

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self, dict_cursor: bool = True):
        """Context manager for database cursor."""
        cursor_factory = RealDictCursor if dict_cursor else None
        cur = self._conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
