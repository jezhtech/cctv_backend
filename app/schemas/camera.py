"""Camera-related Pydantic schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class CameraBase(BaseModel):
    """Base camera schema."""
    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    description: Optional[str] = Field(None, max_length=1000, description="Camera description")
    ip_address: str = Field(..., description="Camera IP address")
    port: int = Field(default=554, ge=1, le=65535, description="RTSP port number")
    path: Optional[str] = Field(None, max_length=255, description="Optional RTSP path (e.g., /h264.sdp)")
    username: Optional[str] = Field(None, max_length=100, description="RTSP username")
    password: Optional[str] = Field(None, max_length=255, description="RTSP password")
    frame_rate: int = Field(default=5, ge=1, le=60, description="Frame rate for processing")
    resolution_width: int = Field(default=1920, ge=640, le=3840, description="Camera resolution width")
    resolution_height: int = Field(default=1080, ge=480, le=2160, description="Camera resolution height")
    location: Optional[str] = Field(None, max_length=255, description="Camera location")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional camera settings")


class CameraCreate(CameraBase):
    """Schema for creating a new camera."""
    pass


class CameraUpdate(BaseModel):
    """Schema for updating a camera."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Camera name")
    description: Optional[str] = Field(None, max_length=1000, description="Camera description")
    ip_address: Optional[str] = Field(None, description="Camera IP address")
    port: Optional[int] = Field(None, ge=1, le=65535, description="RTSP port number")
    path: Optional[str] = Field(None, max_length=255, description="Optional RTSP path")
    username: Optional[str] = Field(None, max_length=100, description="RTSP username")
    password: Optional[str] = Field(None, max_length=255, description="RTSP password")
    frame_rate: Optional[int] = Field(None, ge=1, le=60, description="Frame rate for processing")
    resolution_width: Optional[int] = Field(None, ge=640, le=3840, description="Camera resolution width")
    resolution_height: Optional[int] = Field(None, ge=480, le=2160, description="Camera resolution height")
    location: Optional[str] = Field(None, max_length=255, description="Camera location")
    is_active: Optional[bool] = Field(None, description="Camera active status")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional camera settings")


class CameraResponse(CameraBase):
    """Schema for camera response."""
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CameraListResponse(BaseModel):
    """Schema for paginated camera list response."""
    data: List[CameraResponse]
    total: int
    page: int
    size: int
    pages: int


class CameraTestConnection(BaseModel):
    """Schema for testing camera connection."""
    ip_address: str
    port: int = Field(default=554, ge=1, le=65535)
    path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = Field(default=30, ge=5, le=120)
    
    @property
    def rtsp_url(self) -> str:
        """Construct RTSP URL from components."""
        if self.username and self.password:
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.port}{self.path or ''}"
        else:
            return f"rtsp://{self.ip_address}:{self.port}{self.path or ''}"


class CameraTestResponse(BaseModel):
    """Schema for camera connection test response."""
    success: bool
    message: str
    error_details: Optional[str] = None
    processing_time_ms: Optional[float] = None
    frame_rate: Optional[float] = None
    resolution: Optional[Dict[str, int]] = None


class CameraHealthResponse(BaseModel):
    """Schema for camera health check response."""
    camera_id: uuid.UUID
    is_online: bool
    rtsp_accessible: bool
    frame_rate: float
    last_frame_time: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: float


class CameraStatistics(BaseModel):
    """Schema for camera statistics."""
    total_cameras: int
    active_cameras: int
    offline_cameras: int
    total_streams: int
    active_streams: int
    total_detections: int
    total_attendance_records: int
    by_location: Dict[str, int]
    by_frame_rate: Dict[str, int]


class CameraStreamStatus(BaseModel):
    """Schema for camera stream status."""
    is_running: bool
    frame_count: int
    fps: float
    faces_detected: int
    errors_count: int
    start_time: Optional[str] = None
    last_frame_time: Optional[str] = None
    memory_usage_mb: float
