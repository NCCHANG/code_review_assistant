"""Database layer for the Intelligent Code Review Assistant.

All persistence goes through this module.  The SQLite database is stored
at ``user_data.db`` in the project root.

Tables
------
User               – registered accounts
Analysis_Session   – one row per analysis run
Analyzed_File      – file / snippet metadata for each session
Detected_Issue     – per-function verdict from the classifier
Suggested_Fix      – CodeT5 fix + Groq explanation for buggy functions
"""

import os
import sqlite3

import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")


# ── Connection helper ────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema creation ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist. Call once on app startup."""
    with _get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS User (
                UserID      INTEGER PRIMARY KEY AUTOINCREMENT,
                Username    VARCHAR(50)  UNIQUE NOT NULL,
                Password    VARCHAR(255) NOT NULL,
                CreatedDate DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS Analysis_Session (
                SessionID    INTEGER PRIMARY KEY AUTOINCREMENT,
                UserID       INTEGER,
                OverallScore FLOAT,
                DateAnalyzed DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (UserID) REFERENCES User(UserID)
            );

            CREATE TABLE IF NOT EXISTS Analyzed_File (
                FileID    INTEGER PRIMARY KEY AUTOINCREMENT,
                SessionID INTEGER,
                FileName  VARCHAR(100),
                FilePath  VARCHAR(255),
                LineCount INTEGER,
                FOREIGN KEY (SessionID) REFERENCES Analysis_Session(SessionID)
            );

            CREATE TABLE IF NOT EXISTS Detected_Issue (
                IssueID    INTEGER PRIMARY KEY AUTOINCREMENT,
                FileID     INTEGER,
                LineNumber INTEGER,
                IssueType  VARCHAR(50),
                Confidence FLOAT,
                FOREIGN KEY (FileID) REFERENCES Analyzed_File(FileID)
            );

            CREATE TABLE IF NOT EXISTS Suggested_Fix (
                FixID        INTEGER PRIMARY KEY AUTOINCREMENT,
                IssueID      INTEGER,
                OriginalCode TEXT,
                FixedCode    TEXT,
                Explanation  TEXT,
                FOREIGN KEY (IssueID) REFERENCES Detected_Issue(IssueID)
            );
        """)


# ── User management ──────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO User (Username, Password) VALUES (?, ?)",
                (username, hashed.decode("utf-8")),
            )
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as exc:
        return False, f"Registration failed: {exc}"


def authenticate_user(username: str, password: str) -> tuple[bool, int | None]:
    """Verify credentials. Returns (success, user_id | None)."""
    try:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT UserID, Password FROM User WHERE Username = ?",
                (username,),
            ).fetchone()

        if row is None:
            return False, None

        if bcrypt.checkpw(password.encode("utf-8"), row["Password"].encode("utf-8")):
            return True, row["UserID"]
        return False, None
    except Exception as exc:
        print(f"[db] Authentication error: {exc}")
        return False, None


# ── Analysis persistence ─────────────────────────────────────────────────────

def save_analysis_session(
    user_id: int,
    overall_score: float,
    file_name: str | None,
    file_path: str | None,
    line_count: int,
    issues: list[dict],
) -> int:
    """Persist one complete analysis session and return its session_id.

    Parameters
    ----------
    user_id       : logged-in user (required)
    overall_score : mean classifier confidence across all functions
    file_name     : basename of the uploaded file, or None for manual input
    file_path     : full path of the uploaded file, or None for manual input
    line_count    : total lines in the submitted code
    issues        : list of dicts, one per function, with keys:
                        line_number   – int, starting line in the source file
                        issue_type    – str, e.g. "Clean" / "Buggy"
                        confidence    – float classifier score
                        original_code – str, the function source
                        fixed_code    – str | None
                        explanation   – str | None
    """
    with _get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO Analysis_Session (UserID, OverallScore) VALUES (?, ?)",
            (user_id, overall_score),
        )
        session_id = cur.lastrowid

        cur2 = conn.execute(
            "INSERT INTO Analyzed_File (SessionID, FileName, FilePath, LineCount) "
            "VALUES (?, ?, ?, ?)",
            (session_id, file_name, file_path, line_count),
        )
        file_id = cur2.lastrowid

        for issue in issues:
            cur3 = conn.execute(
                "INSERT INTO Detected_Issue (FileID, LineNumber, IssueType, Confidence) "
                "VALUES (?, ?, ?, ?)",
                (
                    file_id,
                    issue.get("line_number"),
                    issue.get("issue_type"),
                    issue.get("confidence"),
                ),
            )
            issue_id = cur3.lastrowid

            if issue.get("original_code") or issue.get("fixed_code") or issue.get("explanation"):
                conn.execute(
                    "INSERT INTO Suggested_Fix (IssueID, OriginalCode, FixedCode, Explanation) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        issue_id,
                        issue.get("original_code"),
                        issue.get("fixed_code"),
                        issue.get("explanation"),
                    ),
                )

    return session_id


# ── History queries ──────────────────────────────────────────────────────────

def get_user_sessions(user_id: int) -> list[dict]:
    """Return a summary list of all sessions for *user_id*, newest first.

    Each dict contains: session_id, date, overall_score, file_name,
    function_count (all functions), issue_count (buggy only).
    """
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.SessionID   AS session_id,
                s.DateAnalyzed AS date,
                s.OverallScore AS overall_score,
                f.FileName     AS file_name,
                COUNT(i.IssueID) AS function_count,
                COUNT(CASE WHEN i.IssueType != 'Clean' THEN 1 END) AS issue_count
            FROM Analysis_Session s
            LEFT JOIN Analyzed_File   f ON f.SessionID = s.SessionID
            LEFT JOIN Detected_Issue  i ON i.FileID    = f.FileID
            WHERE s.UserID = ?
            GROUP BY s.SessionID
            ORDER BY s.DateAnalyzed DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_detail(session_id: int) -> dict | None:
    """Return full detail for one session including all issues and fixes."""
    with _get_connection() as conn:
        session = conn.execute(
            "SELECT * FROM Analysis_Session WHERE SessionID = ?", (session_id,)
        ).fetchone()

        if session is None:
            return None

        file_row = conn.execute(
            "SELECT * FROM Analyzed_File WHERE SessionID = ?", (session_id,)
        ).fetchone()

        file_id = file_row["FileID"] if file_row else -1
        issues = conn.execute(
            """
            SELECT i.*, sf.OriginalCode, sf.FixedCode, sf.Explanation
            FROM Detected_Issue i
            LEFT JOIN Suggested_Fix sf ON sf.IssueID = i.IssueID
            WHERE i.FileID = ?
            ORDER BY i.LineNumber
            """,
            (file_id,),
        ).fetchall()

    return {
        "session": dict(session),
        "file": dict(file_row) if file_row else None,
        "issues": [dict(i) for i in issues],
    }
