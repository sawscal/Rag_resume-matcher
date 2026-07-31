"""
MySQL Storage Layer for Candidate Metadata
Stores structured resume metadata (skills, experience, education, TF-IDF score)
in a relational MySQL schema for rapid querying and candidate filtering.

This module gracefully degrades — if DATABASE_URL is not set or MySQL is
unreachable, all methods become no-ops so the rest of the app still works.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS candidates (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    resume_id       VARCHAR(64)   NOT NULL UNIQUE,
    filename        VARCHAR(512)  NOT NULL,
    skills          JSON,
    experience_years FLOAT        DEFAULT 0.0,
    education       VARCHAR(256),
    tfidf_score     FLOAT         DEFAULT 0.0,
    raw_text        LONGTEXT,
    uploaded_at     DATETIME      DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_resume_id (resume_id),
    FULLTEXT INDEX ft_raw_text (raw_text)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ---------------------------------------------------------------------------
# Helper: parse DATABASE_URL → connection kwargs
# ---------------------------------------------------------------------------
def _parse_db_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Parses a MySQL connection URL of the form:
        mysql://user:password@host:3306/dbname
    Returns a dict of kwargs for mysql.connector.connect(), or None on failure.
    """
    try:
        # Strip scheme
        url = url.strip()
        if url.startswith("mysql://"):
            url = url[len("mysql://"):]
        elif url.startswith("mysql+mysqlconnector://"):
            url = url[len("mysql+mysqlconnector://"):]

        # user:password@host:port/dbname
        user_pass, rest = url.split("@", 1)
        if ":" in user_pass:
            user, password = user_pass.split(":", 1)
        else:
            user, password = user_pass, ""

        if "/" in rest:
            host_port, dbname = rest.rsplit("/", 1)
        else:
            host_port, dbname = rest, ""

        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        else:
            host, port = host_port, 3306

        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": dbname,
        }
    except Exception as exc:
        logger.warning(f"[CandidateDB] Failed to parse DATABASE_URL: {exc}")
        return None


# ---------------------------------------------------------------------------
# CandidateDB class
# ---------------------------------------------------------------------------
class CandidateDB:
    """
    Thin wrapper around mysql-connector-python.
    All public methods return safely even when the DB is unavailable.
    """

    def __init__(self):
        self._conn = None
        self._available = False
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self):
        """Attempt to connect to MySQL using DATABASE_URL env var."""
        db_url = os.environ.get("DATABASE_URL", "").strip()
        if not db_url:
            logger.info("[CandidateDB] DATABASE_URL not set — MySQL storage disabled.")
            return

        kwargs = _parse_db_url(db_url)
        if not kwargs:
            return

        try:
            import mysql.connector  # type: ignore
            self._conn = mysql.connector.connect(**kwargs, autocommit=True)
            self._available = True
            self._ensure_table()
            logger.info("[CandidateDB] Connected to MySQL successfully.")
        except Exception as exc:
            logger.warning(f"[CandidateDB] Could not connect to MySQL: {exc}")
            self._available = False

    def _ensure_table(self):
        """Create the candidates table if it doesn't already exist."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(CREATE_TABLE_SQL)
            cursor.close()
        except Exception as exc:
            logger.warning(f"[CandidateDB] Table creation failed: {exc}")

    def _cursor(self):
        if not self._available or self._conn is None:
            return None
        try:
            # Reconnect if connection dropped
            self._conn.ping(reconnect=True, attempts=2, delay=1)
            return self._conn.cursor(dictionary=True)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available

    def insert_candidate(
        self,
        resume_id: str,
        filename: str,
        skills: List[str],
        experience_years: float,
        education: Optional[str],
        tfidf_score: float,
        raw_text: str,
    ) -> bool:
        """
        Inserts or replaces a candidate record.
        Returns True on success, False otherwise.
        """
        cursor = self._cursor()
        if cursor is None:
            return False
        try:
            sql = """
                INSERT INTO candidates
                    (resume_id, filename, skills, experience_years, education, tfidf_score, raw_text, uploaded_at)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    filename         = VALUES(filename),
                    skills           = VALUES(skills),
                    experience_years = VALUES(experience_years),
                    education        = VALUES(education),
                    tfidf_score      = VALUES(tfidf_score),
                    raw_text         = VALUES(raw_text),
                    uploaded_at      = VALUES(uploaded_at)
            """
            cursor.execute(sql, (
                resume_id,
                filename,
                json.dumps(skills),
                experience_years,
                education,
                tfidf_score,
                raw_text[:65535],  # LONGTEXT cap for safety
                datetime.utcnow(),
            ))
            cursor.close()
            return True
        except Exception as exc:
            logger.warning(f"[CandidateDB] insert_candidate failed: {exc}")
            return False

    def query_by_skill(self, skill: str) -> List[Dict[str, Any]]:
        """
        Returns all candidates whose skills JSON array contains the given skill (case-insensitive).
        """
        cursor = self._cursor()
        if cursor is None:
            return []
        try:
            sql = """
                SELECT resume_id, filename, skills, experience_years, education, tfidf_score, uploaded_at
                FROM candidates
                WHERE JSON_SEARCH(LOWER(skills), 'one', LOWER(%s)) IS NOT NULL
                ORDER BY tfidf_score DESC
            """
            cursor.execute(sql, (skill,))
            rows = cursor.fetchall()
            cursor.close()
            for row in rows:
                if isinstance(row.get("skills"), str):
                    row["skills"] = json.loads(row["skills"])
            return rows
        except Exception as exc:
            logger.warning(f"[CandidateDB] query_by_skill failed: {exc}")
            return []

    def query_by_min_experience(self, min_years: float) -> List[Dict[str, Any]]:
        """Returns candidates with experience_years >= min_years."""
        cursor = self._cursor()
        if cursor is None:
            return []
        try:
            sql = """
                SELECT resume_id, filename, skills, experience_years, education, tfidf_score, uploaded_at
                FROM candidates
                WHERE experience_years >= %s
                ORDER BY experience_years DESC
            """
            cursor.execute(sql, (min_years,))
            rows = cursor.fetchall()
            cursor.close()
            for row in rows:
                if isinstance(row.get("skills"), str):
                    row["skills"] = json.loads(row["skills"])
            return rows
        except Exception as exc:
            logger.warning(f"[CandidateDB] query_by_min_experience failed: {exc}")
            return []

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all stored candidates ordered by upload time desc."""
        cursor = self._cursor()
        if cursor is None:
            return []
        try:
            cursor.execute("""
                SELECT resume_id, filename, skills, experience_years, education, tfidf_score, uploaded_at
                FROM candidates
                ORDER BY uploaded_at DESC
            """)
            rows = cursor.fetchall()
            cursor.close()
            for row in rows:
                if isinstance(row.get("skills"), str):
                    row["skills"] = json.loads(row["skills"])
            return rows
        except Exception as exc:
            logger.warning(f"[CandidateDB] get_all failed: {exc}")
            return []

    def clear(self) -> bool:
        """Truncates the candidates table."""
        cursor = self._cursor()
        if cursor is None:
            return False
        try:
            cursor.execute("TRUNCATE TABLE candidates")
            cursor.close()
            return True
        except Exception as exc:
            logger.warning(f"[CandidateDB] clear failed: {exc}")
            return False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
candidate_db = CandidateDB()
