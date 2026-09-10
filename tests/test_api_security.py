import app.database as database
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()
    return TestClient(app)


def _register(client, email):
    response = client.post(
        "/auth/register",
        json={"email": email, "password": "safe-password-123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_health_endpoint_is_available(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_document_routes_require_authentication(client):
    response = client.get("/rag/documents")

    assert response.status_code in {401, 403}


def test_conversation_messages_are_user_scoped(client):
    owner_token = _register(client, "owner@test.example")
    other_token = _register(client, "other@test.example")

    with database.get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_email) VALUES (?, ?)",
            ("private-conversation", "owner@test.example"),
        )
        conn.commit()

    response = client.get(
        "/rag/conversations/private-conversation/messages",
        headers={"Authorization": f"Bearer {other_token}"},
    )

    assert response.status_code == 404
    assert owner_token
