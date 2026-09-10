from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str
    COHERE_API_KEY: str
    SECRET_KEY: str = Field(min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: str = "http://localhost:3000"
    MAX_UPLOAD_MB: int = 10
    MAX_PDF_PAGES: int = 50
    AI_REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=120)
    AI_MAX_RETRIES: int = Field(default=2, ge=0, le=3)
    AI_RETRY_BACKOFF_SECONDS: float = Field(default=0.5, ge=0, le=5)
    RAG_MAX_L2_DISTANCE: float = Field(default=1.4, gt=0)

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        secret = value.strip()
        normalized = secret.lower()
        insecure_markers = (
            "replace_with",
            "your_",
            "change_me",
            "changeme",
            "default_secret",
        )
        if any(marker in normalized for marker in insecure_markers):
            raise ValueError(
                "SECRET_KEY must be replaced with a unique random secret"
            )
        return secret

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()