"""API v1 router."""
from fastapi import APIRouter

from . import users, cameras, streams

api_router = APIRouter()

api_router.include_router(users.users_router, prefix="/users", tags=["users"])
api_router.include_router(cameras.cameras_router, prefix="/cameras", tags=["cameras"])
api_router.include_router(streams.streams_router, prefix="/streams", tags=["streams"])