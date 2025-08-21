"""Application configuration settings."""
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application
    APP_NAME: str = "Smart Attendance Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL database URL")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_RETRY_COUNT: int = 3
    DATABASE_RETRY_DELAY: int = 5
    DATABASE_RETRY_ATTEMPTS: int = 3
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_POOL_SIZE: int = 10
    
    # Security
    SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Face Recognition
    FACE_DETECTION_CONFIDENCE: float = 0.6
    FACE_RECOGNITION_THRESHOLD: float = 0.5  # More lenient threshold (50%) for face recognition
    MAX_FACES_PER_IMAGE: int = 10
    
    # Face Tracking (Duplicate Prevention)
    FACE_DUPLICATE_PREVENTION_DISTANCE: int = 50  # Pixels distance to consider faces as the same person
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must start with postgresql://")
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables not defined in the model


# Global settings instance
settings = Settings()
