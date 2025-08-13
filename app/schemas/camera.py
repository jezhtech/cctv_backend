"""Camera-related Pydantic schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class CameraBase(BaseModel):
    """Base camera schema."""
    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    description: Optional[str] = Field(None, max_length=1000, description="Camera description")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    username: Optional[str] = Field(None, max_length=100, description="RTSP username")
    password: Optional[str] = Field(None, max_length=255, description="RTSP password")
    frame_rate: int = Field(default=5, ge=1, le=30, description="Frame rate for processing")
    resolution_width: int = Field(default=1920, ge=640, le=3840, description="Camera resolution width")
    resolution_height: int = Field(default=1080, ge=480, le=2160, description="Camera resolution height")
    location: Optional[str] = Field(None, max_length=255, description="Camera location")
    latitude: Optional[str] = Field(None, max_length=20, description="Camera latitude")
    longitude: Optional[str] = Field(None, max_length=20, description="Camera longitude")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional camera settings")


class CameraCreate(CameraBase):
    """Schema for creating a new camera."""
    pass


class CameraUpdate(BaseModel):
    """Schema for updating camera information."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    rtsp_url: Optional[str] = None
    username: Optional[str] = Field(None, max_length=100)
    password: Optional[str] = Field(None, max_length=255)
    frame_rate: Optional[int] = Field(None, ge=1, le=30)
    resolution_width: Optional[int] = Field(None, ge=640, le=3840)
    resolution_height: Optional[int] = Field(None, ge=480, le=2160)
    location: Optional[str] = Field(None, max_length=255)
    latitude: Optional[str] = Field(None, max_length=20)
    longitude: Optional[str] = Field(None, max_length=20)
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class CameraResponse(CameraBase):
    """Schema for camera response."""
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    is_online: bool
    
    class Config:
        from_attributes = True


class CameraStreamResponse(BaseModel):
    """Schema for camera stream response."""
    id: uuid.UUID
    camera_id: uuid.UUID
    is_active: bool
    started_at: Optional[str]
    last_frame_at: Optional[str]
    total_frames_processed: int
    faces_detected: int
    errors_count: int
    fps: int
    memory_usage_mb: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CameraWithStreams(CameraResponse):
    """Schema for camera with stream information."""
    streams: List[CameraStreamResponse] = []
    
    class Config:
        from_attributes = True


class CameraHealthCheck(BaseModel):
    """Schema for camera health check response."""
    camera_id: uuid.UUID
    is_online: bool
    rtsp_accessible: bool
    frame_rate: float
    last_frame_time: Optional[datetime]
    error_message: Optional[str]
    response_time_ms: float


class CameraStatistics(BaseModel):
    """Schema for camera statistics."""
    camera_id: uuid.UUID
    total_detections: int
    total_recognitions: int
    average_confidence: float
    uptime_seconds: int
    last_activity: Optional[datetime]
    error_rate: float


class CameraListResponse(BaseModel):
    """Schema for paginated camera list response."""
    cameras: List[CameraResponse]
    total: int
    page: int
    size: int
    pages: int


class CameraTestConnection(BaseModel):
    """Schema for testing camera connection."""
    rtsp_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = Field(default=30, ge=5, le=120)


class CameraTestResponse(BaseModel):
    """Schema for camera connection test response."""
    success: bool
    message: str
    frame_count: Optional[int] = None
    resolution: Optional[Dict[str, int]] = None
    frame_rate: Optional[float] = None
    error_details: Optional[str] = None
