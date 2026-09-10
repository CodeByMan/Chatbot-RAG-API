import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import auth, rag
from app.core.config import settings
from app.database import consume_rate_limit

app = FastAPI(
    title="RAG Chatbot API",
    description="Authenticated RAG system with JWT authentication",
    version="1.0.0"
)
logger = logging.getLogger(__name__)

allowed_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS.split(",")
    if origin.strip()
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


RATE_LIMITS = {
    "/auth/register": (5, 60),
    "/auth/login": (10, 60),
    "/rag/upload": (10, 60),
    "/rag/query": (30, 60),
    "/rag/chat": (30, 60),
}


@app.middleware("http")
async def rate_limit_requests(request, call_next):
    """Apply a shared SQLite-backed limit to auth and AI-heavy routes."""
    limit_config = RATE_LIMITS.get(request.url.path)
    if limit_config:
        limit, window_seconds = limit_config
        client_ip = request.client.host if request.client else "unknown"
        key = f"{client_ip}:{request.url.path}"
        try:
            allowed = consume_rate_limit(key, limit, window_seconds)
        except Exception:
            logger.exception("Rate-limit storage is unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "Request protection is temporarily unavailable."},
            )

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": str(window_seconds)},
            )

    return await call_next(request)

# Include routers
app.include_router(auth.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth",
            "rag": "/rag",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
