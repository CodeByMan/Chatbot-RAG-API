import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "./rag_app.db"


def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users(email)
        )
    """)

    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)

    # Documents metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            filename TEXT NOT NULL,
            chunks_count INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users(email)
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user_email "
        "ON conversations(user_email)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id "
        "ON messages(conversation_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_user_email "
        "ON documents(user_email)"
    )
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit_events (
            client_key TEXT NOT NULL,
            requested_at REAL NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rate_limit_events_key_time "
        "ON rate_limit_events(client_key, requested_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rate_limit_events_requested_at "
        "ON rate_limit_events(requested_at)"
    )

    conn.commit()
    conn.close()


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def consume_rate_limit(client_key: str, limit: int, window_seconds: int) -> bool:
    """Atomically record a request using the shared SQLite database.

    SQLite is shared by worker processes on the same host, so this avoids the
    per-process-only behavior of an in-memory request counter. Deployments
    spanning multiple hosts should still enforce an edge rate limit.
    """
    if limit < 1 or window_seconds < 1:
        raise ValueError("Rate-limit values must be positive")

    now = time.time()
    cutoff = now - window_seconds
    cleanup_cutoff = now - max(window_seconds, 60)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM rate_limit_events WHERE requested_at < ?",
            (cleanup_cutoff,),
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM rate_limit_events "
            "WHERE client_key = ? AND requested_at >= ?",
            (client_key, cutoff),
        ).fetchone()[0]
        if count >= limit:
            conn.rollback()
            return False

        conn.execute(
            "INSERT INTO rate_limit_events (client_key, requested_at) "
            "VALUES (?, ?)",
            (client_key, now),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Initialize on import
init_db()
