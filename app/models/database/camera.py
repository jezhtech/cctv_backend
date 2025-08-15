"""Camera database model."""
from sqlalchemy import Column, Integer, String, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .base import Base, TimestampMixin, SoftDeleteMixin


class Camera(Base, TimestampMixin, SoftDeleteMixin):
    """Camera model for storing RTSP camera information."""
    
    __tablename__ = "cameras"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Network Configuration
    ip_address = Column(String(45), nullable=False)  # IPv4/IPv6 address
    port = Column(Integer, default=554, nullable=False)  # RTSP port
    path = Column(String(255), nullable=True)  # Optional path (e.g., /h264.sdp)
    username = Column(String(100), nullable=True)
    password = Column(String(255), nullable=True)
    
    # Camera Settings
    is_active = Column(Boolean, default=True, nullable=False)
    frame_rate = Column(Integer, default=5, nullable=False)
    resolution_width = Column(Integer, default=1920, nullable=False)
    resolution_height = Column(Integer, default=1080, nullable=False)
    
    # Location
    location = Column(String(255), nullable=True)
    
    # Additional Configuration
    settings = Column(JSON, nullable=True)  # Store additional camera-specific settings
    
    # Relationships
    streams = relationship("CameraStream", back_populates="camera", cascade="all, delete-orphan", lazy="dynamic")
    detections = relationship("FaceDetection", back_populates="camera", cascade="all, delete-orphan", lazy="dynamic")
    attendances = relationship("Attendance", back_populates="camera", cascade="all, delete-orphan", lazy="dynamic")
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name='{self.name}', ip='{self.ip_address}:{self.port}')>"
    
    @property
    def is_online(self) -> bool:
        """Check if camera is currently online."""
        return any(stream.is_active for stream in self.streams)
    
    @property
    def current_stream(self):
        """Get current active stream."""
        return next((stream for stream in self.streams if stream.is_active), None)
    
    @property
    def rtsp_url(self) -> str:
        """Construct RTSP URL from components."""
        # Handle path properly - avoid double slashes
        path_part = self.path if self.path else ""
        if path_part and not path_part.startswith("/"):
            path_part = "/" + path_part
        
        if self.username and self.password:
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.port}{path_part}"
        else:
            return f"rtsp://{self.ip_address}:{self.port}{path_part}"


class CameraStream(Base, TimestampMixin):
    """Camera stream model for managing active streams."""
    
    __tablename__ = "camera_streams"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=False)
    
    # Stream Status
    is_active = Column(Boolean, default=True, nullable=False)
    started_at = Column(String(255), nullable=True)  # ISO timestamp
    last_frame_at = Column(String(255), nullable=True)  # ISO timestamp
    
    # Stream Metrics
    total_frames_processed = Column(Integer, default=0, nullable=False)
    faces_detected = Column(Integer, default=0, nullable=False)
    errors_count = Column(Integer, default=0, nullable=False)
    
    # Performance
    fps = Column(Integer, default=0, nullable=False)
    memory_usage_mb = Column(Integer, default=0, nullable=False)
    
    # Relationships
    camera = relationship("Camera", back_populates="streams", lazy="joined")
    
    def __repr__(self):
        return f"<CameraStream(id={self.id}, camera_id={self.camera_id}, is_active={self.is_active})>"
