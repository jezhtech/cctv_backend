#!/usr/bin/env python3
"""Test script to verify database and Redis connections."""
import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

async def test_redis():
    """Test Redis connection."""
    try:
        from app.services.redis_service import redis_service
        await redis_service.initialize()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_database():
    """Test database connection."""
    try:
        from app.core.database import check_db_health
        if check_db_health():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def main():
    """Main test function."""
    print("Testing connections...")
    print(f"REDIS_URL: {os.getenv('REDIS_URL', 'Not set')}")
    print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
    print()
    
    # Test database
    db_ok = test_database()
    
    # Test Redis
    redis_ok = await test_redis()
    
    print()
    if db_ok and redis_ok:
        print("🎉 All connections successful!")
        return 0
    else:
        print("💥 Some connections failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
