"""Database connection and session management."""
from typing import AsyncGenerator, Generator, Optional
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from loguru import logger
import time

from .config import settings

_engine = None
_SessionFactory: Optional[sessionmaker] = None
_ScopedSession = None

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
    from app.models.database import Base
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables."""
    # Import models only when needed to avoid circular imports
    from app.models.database import Base
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)


# Database health check
def check_db_health() -> bool:
    """Check database connection health with retry logic."""
    max_retries = settings.DATABASE_RETRY_ATTEMPTS
    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                try:
                    connection.execute(text("SELECT 1"))
                    return True
                except Exception as e:
                    logger.error(f"Database connection failed on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(settings.DATABASE_RETRY_DELAY)
                        continue
                    return False
        except Exception as e:
            logger.error(f"Database engine connection failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(settings.DATABASE_RETRY_DELAY)
                continue
            return False
    return False


def get_db_stats() -> dict:
    """Get database connection pool statistics."""
    try:
        if engine and hasattr(engine, 'pool'):
            pool = engine.pool
            return {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
        return {"error": "Engine not available"}
    except Exception as e:
        logger.error(f"Failed to get database stats: {str(e)}")
        return {"error": str(e)}


def reset_db_connections():
    """Reset all database connections in the pool."""
    try:
        if engine and hasattr(engine, 'pool'):
            engine.pool.dispose()
            logger.info("Database connection pool reset successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to reset database connections: {str(e)}")
        return False


def init_engine():
    global _engine, _SessionFactory, _ScopedSession
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,  # recycle stale conns
            future=True,
        )
        _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False, future=True)
        # scoped_session is important because you use threads (RTSP worker threads)
        _ScopedSession = scoped_session(_SessionFactory)

def dispose_engine():
    global _engine, _SessionFactory, _ScopedSession
    if _ScopedSession is not None:
        _ScopedSession.remove()
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
    _ScopedSession = None

def SessionLocal():
    # Always ensure engine exists in the **current process**
    if _ScopedSession is None:
        init_engine()
    return _ScopedSession