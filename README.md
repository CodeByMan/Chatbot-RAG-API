<div align="center">

# 🤖 Chatbot RAG API

An authenticated FastAPI document-question-answering API with PDF ingestion, Cohere embeddings, FAISS vector retrieval, Groq response generation, JWT authentication, user-scoped conversations, and defensive AI-provider controls.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Web-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![SQLite](https://img.shields.io/badge/Database-SQLite-003b57?style=for-the-badge&logo=sqlite&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-ff6f00?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-16%20Passed-22c55e?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-f97316?style=for-the-badge)

</div>

---

## 📌 Project Overview

**RAG Chatbot API** is a local document-question-answering backend built to demonstrate practical Retrieval-Augmented Generation engineering. It accepts authenticated PDF uploads, extracts text, creates Cohere embeddings, stores vectors in FAISS, retrieves context belonging only to the authenticated user, and generates answers with Groq.

The application includes JWT bearer authentication, bcrypt password hashing, SQLite persistence, conversation history, request rate limiting, bounded AI-provider retries, similarity filtering, safe provider-error handling, and prompt-grounding safeguards.

The project is designed to be understandable and runnable by a junior Python or AI developer. It does not include a frontend and does not ship a country-specific demo dataset.

---

## ✨ Features

### 🔐 Authentication and Access Control

- JWT bearer authentication with expiring access tokens
- bcrypt password hashing through Passlib
- Email validation for registration and login
- Minimum password-length validation
- Explicit protection for all document and conversation routes
- User-scoped document retrieval
- User-scoped conversation and message access
- Generic authentication failure messages
- SQLite-backed request rate limiting for authentication and AI-heavy routes

### 📄 PDF and Document Workflow

- Authenticated PDF upload endpoint
- File extension, upload-size, and page-count validation
- Text extraction with `pypdf`
- Configurable maximum upload size and PDF page count
- Cohere document embeddings using `embed-english-v3.0`
- FAISS local vector index persistence
- JSON metadata persistence for document chunks
- Document listing and user-authorized deletion

### 🧠 Retrieval-Augmented Generation

- Cohere query embeddings using `search_query` input type
- FAISS L2-distance retrieval
- Configurable similarity-distance threshold
- Top-k retrieval bounds
- Context filtering by authenticated user
- Source filename, chunk identifier, and preview returned with answers
- Groq answer generation using `llama-3.1-8b-instant`
- Conversation-aware question answering
- Safe fallback when no relevant context is found

### 🛡️ Reliability and AI Safety

- Explicit provider request timeouts
- Bounded provider retries with backoff
- Safe provider-failure boundaries
- Provider embedding-shape validation
- Empty or malformed answer rejection
- Temporary FAISS/metadata files during persistence
- Startup reconciliation between SQLite document metadata and FAISS metadata
- Document text and conversation history treated as untrusted reference material
- No tools, agents, shell execution, or external action-taking capabilities

### 🧪 API and Validation

- Public health endpoint
- Public authentication endpoints
- Protected RAG endpoints
- Pydantic request and response validation
- Generic server-side error responses
- Interactive FastAPI documentation at `/docs`

---

## 🖼️ Screenshots

This backend-only project provides interactive API documentation at:

<http://127.0.0.1:8000/docs>

After starting the API, the Swagger UI can be used to inspect and exercise the available endpoints.

---

## 🧰 Tech Stack

### Core Python

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic
- Pydantic Settings
- SQLite
- `pypdf` PDF text extraction

### AI and Retrieval

- Cohere embeddings
- FAISS vector search
- Groq chat completion
- Retrieval-Augmented Generation
- Similarity threshold filtering
- Prompt grounding and untrusted-context handling

### Security and Operations

- JWT bearer authentication
- `python-jose` with the cryptography backend
- Passlib and bcrypt password hashing
- Explicit secret validation
- CORS origin configuration
- SQLite-backed request rate limiting
- Generic API error responses

### Quality Assurance

- pytest
- 16 automated tests
- Python AST/syntax validation
- Authentication and authorization tests
- RAG user-isolation tests
- Provider retry and response-validation tests
- GitHub Actions CI
- Dockerfile support

---

## 📁 Folder Structure

```text
rag-chatbot-api/
├── app/
│   ├── api/             # Authentication and RAG API routes
│   ├── core/            # Environment-backed application settings
│   ├── models/          # Pydantic request and response models
│   ├── services/        # Authentication and RAG services
│   ├── utils/           # PDF extraction utilities
│   ├── database.py      # SQLite schema and rate limiting
│   └── main.py          # FastAPI application and middleware
├── tests/               # API, database, model, and RAG isolation tests
├── .env.example         # Environment configuration template
├── .github/
│   └── workflows/
│       └── ci.yml       # Python 3.11 and 3.13 CI test matrix
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

Runtime databases, FAISS files, secrets, virtual environments, caches, and generated files are excluded from Git.

---

## 🔐 Environment Variables

Copy `.env.example` to `.env`. Keep the local `.env` file private:

```env
GROQ_API_KEY=<your-groq-api-key>
COHERE_API_KEY=<your-cohere-api-key>
SECRET_KEY=<unique-random-secret-at-least-32-characters>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=http://localhost:3000
MAX_UPLOAD_MB=10
MAX_PDF_PAGES=50
AI_REQUEST_TIMEOUT_SECONDS=30
AI_MAX_RETRIES=2
AI_RETRY_BACKOFF_SECONDS=0.5
RAG_MAX_L2_DISTANCE=1.4
```

The application rejects missing or predictable secret-key values. Generate a random secret and never commit `.env` to Git.

---

## 🚀 Getting Started on Windows

### Prerequisites

- Windows 10 or Windows 11
- Python 3.11 or newer
- Python 3.13.15 recommended
- Internet access for the initial dependency installation
- Cohere and Groq API keys for live AI requests

### 1. Create the Environment

Use **Windows Command Prompt (CMD)** from the project root:

```bat
cd /d "D:\\path\\to\\rag-chatbot-api"
py -3.13 -m venv .venv
.venv\\Scripts\\activate.bat
python -c "import sys; print(sys.executable); print(sys.version)"
```

The executable path should point to `.venv\\Scripts\\python.exe`.

### 2. Install Dependencies

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 3. Configure the Environment

```bat
copy /Y .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
notepad .env
```

Place the generated value in `SECRET_KEY`. Add real Cohere and Groq API keys before using upload or query endpoints.

### 4. Run the Tests

```bat
python -m compileall -q app tests
python -m pytest -q
```

Expected result:

```text
16 passed
```

Four deprecation warnings may appear depending on the installed dependency versions.

### 5. Start the API

```bat
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.

Check the health endpoint from another CMD window:

```bat
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"healthy"}
```

Press `Ctrl+C` to stop Uvicorn, then run `deactivate` to leave the virtual environment.

---

## 🧪 Tests and Verification

```bat
python -m pip check
python -m compileall -q app tests
python -m pytest -q
```

The test suite covers:

- Health endpoint availability
- Authentication requirements for protected routes
- User-scoped conversation access
- SQLite foreign-key enforcement
- Ownership indexes
- SQLite-backed rate limiting
- Request validation and top-k bounds
- Placeholder secret rejection
- User-isolated FAISS retrieval
- FAISS metadata JSON persistence
- Similarity-distance filtering
- Safe no-context responses
- Empty provider-answer rejection
- Transient-provider retry behavior

Tests use temporary SQLite paths where applicable and do not require live Cohere or Groq requests.

---

## 🔄 RAG Request Flow

1. A user registers or signs in.
2. The API returns a short-lived JWT bearer token.
3. The authenticated user uploads a PDF.
4. `pypdf` extracts text from the document.
5. Cohere creates document embeddings.
6. FAISS stores the vectors locally with JSON metadata.
7. The authenticated user submits a question.
8. Cohere creates a query embedding.
9. FAISS retrieves relevant user-owned chunks above the configured threshold.
10. Groq generates an answer using the retrieved context.
11. The API returns the answer and source previews.

---

## ⚠️ Scope and Limitations

- This is an educational local-first RAG API, not a production banking or enterprise platform.
- The project has no frontend; clients can use the API or Swagger UI.
- SQLite and the local FAISS index are intended for bounded local use.
- The in-process AI calls are synchronous and may require a worker or queue for larger deployments.
- FAISS and SQLite persistence includes reconciliation but is not a fully transactional single-store commit.
- The project does not include a Pakistan-specific training or evaluation dataset.
- Live Cohere and Groq credentials are required for actual embedding and answer-generation requests.
- The API should be deployed behind HTTPS with an edge or reverse-proxy rate limit in production.

---

## 🧑‍💻 Author

**Muhammad Ali Nawaz**  
AI Developer

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <b>⭐ If this project helps you, consider starring the repository!</b>
</p>
