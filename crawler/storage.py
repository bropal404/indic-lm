import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class StateStore:
    def __init__(self, path: str = "crawl_state.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS urls (
                normalized_url TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'DISCOVERED',
                http_status INTEGER,
                last_attempt TEXT,
                attempt_count INTEGER DEFAULT 0,
                content_hash TEXT,
                source TEXT,
                rejection_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                normalized_url TEXT,
                source TEXT,
                char_count INTEGER,
                language_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def add_urls(self, rows: Iterable[Dict[str, str]]) -> int:
        count = 0
        with self._lock:
            for row in rows:
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO urls (normalized_url, url, domain, status, source)
                    VALUES (?, ?, ?, 'QUEUED', ?)
                    """,
                    (row["normalized_url"], row["url"], row["domain"], row.get("source")),
                )
                count += cur.rowcount
            self.conn.commit()
        return count

    def queued(self, domain: str, limit: int) -> List[str]:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT normalized_url FROM urls WHERE domain = ? AND status = 'QUEUED'
                ORDER BY
                    CASE
                        WHEN normalized_url GLOB '*/[0-9][0-9][0-9][0-9]/[0-9][0-9]/*' THEN 0
                        WHEN normalized_url GLOB '*/[0-9][0-9][0-9][0-9][0-9]*' THEN 1
                        WHEN normalized_url LIKE '%/news/%' OR normalized_url LIKE '%/story/%'
                             OR normalized_url LIKE '%/article/%' OR normalized_url LIKE '%/samachar/%' THEN 2
                        ELSE 9
                    END,
                    discovered_at
                LIMIT ?
                """,
                (domain, limit),
            )
            return [row["normalized_url"] for row in cur.fetchall()]

    def mark_attempt(self, url: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE urls SET status='FETCHED', attempt_count=attempt_count+1, last_attempt=CURRENT_TIMESTAMP WHERE normalized_url=?",
                (url,),
            )
            self.conn.commit()

    def mark(self, url: str, status: str, http_status: Optional[int] = None, content_hash: Optional[str] = None, rejection_reason: Optional[str] = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE urls
                SET status=?, http_status=COALESCE(?, http_status), content_hash=COALESCE(?, content_hash), rejection_reason=?
                WHERE normalized_url=?
                """,
                (status, http_status, content_hash, rejection_reason, url),
            )
            self.conn.commit()

    def add_document(self, doc_id: str, url: str, source: str, char_count: int, language_score: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO documents (id, normalized_url, source, char_count, language_score) VALUES (?, ?, ?, ?, ?)",
                (doc_id, url, source, char_count, language_score),
            )
            self.conn.commit()

    def stats(self) -> Dict[str, int]:
        stats = {}
        for row in self.conn.execute("SELECT status, COUNT(*) AS n FROM urls GROUP BY status"):
            stats[row["status"]] = row["n"]
        return stats

    def close(self) -> None:
        self.conn.close()


class JsonlCorpusWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a", encoding="utf-8")

    def write(self, row: Dict[str, object]) -> None:
        self.fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
