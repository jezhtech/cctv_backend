"""Database connection and session management."""
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from loguru import logger

from .config import settings

# Database URLs
DATABASE_URL = settings.DATABASE_URL
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Engine configuration
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Async engine configuration
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Metadata
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def create_tables():
    """Create all database tables."""
    # Import models only when needed to avoid circular imports
    from app.models.database.base import Base
    from app.models.database.camera import Camera, CameraStream
    from app.models.database.user import User, FaceEmbedding, Attendance, FaceDetection
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables."""
    # Import models only when needed to avoid circular imports
    from app.models.database.base import Base
    from app.models.database.camera import Camera, CameraStream
    from app.models.database.user import User, FaceEmbedding, Attendance, FaceDetection
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)


# Database health check
def check_db_health() -> bool:
    """Check database connection health."""
    try:
        with engine.connect() as connection:
            try:
                connection.execute(text("SELECT 1"))
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                return False
        return True
    except Exception:
        return False
