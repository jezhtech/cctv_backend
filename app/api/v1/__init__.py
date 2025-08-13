"""API v1 package."""
from fastapi import APIRouter
from app.api.v1 import cameras_router, users_router

api_router = APIRouter()

api_router.include_router(cameras_router, prefix="/cameras", tags=["cameras"])
api_router.include_router(users_router, prefix="/users", tags=["users"])