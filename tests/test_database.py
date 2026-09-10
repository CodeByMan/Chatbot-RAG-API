import app.database as database


def test_database_enables_foreign_keys(tmp_path, monkeypatch):
    test_database_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", str(test_database_path))

    database.init_db()

    with database.get_db() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_database_creates_ownership_indexes(tmp_path, monkeypatch):
    test_database_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", str(test_database_path))

    database.init_db()

    with database.get_db() as conn:
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    assert "idx_conversations_user_email" in indexes
    assert "idx_documents_user_email" in indexes


def test_rate_limiter_rejects_requests_after_limit(tmp_path, monkeypatch):
    test_database_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", str(test_database_path))

    database.init_db()

    assert database.consume_rate_limit("test-client:/rag/query", 2, 60)
    assert database.consume_rate_limit("test-client:/rag/query", 2, 60)
    assert not database.consume_rate_limit("test-client:/rag/query", 2, 60)
