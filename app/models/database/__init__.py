"""Database models package."""

# Import models in the correct order to avoid circular dependencies
from .base import Base, TimestampMixin

# Import Camera model first (it has no dependencies on other models)
from .camera import Camera, CameraStream

# Import User model (it has no dependencies on Camera)
from .user import User, FaceEmbedding

# Import models that depend on both Camera and User
from .user import Attendance, FaceDetection

# Export all models
__all__ = [
    "Base",
    "TimestampMixin", 
    "Camera",
    "CameraStream",
    "User",
    "FaceEmbedding",
    "Attendance",
    "FaceDetection"
]
