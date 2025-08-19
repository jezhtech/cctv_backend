"""User and face recognition related Pydantic schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, EmailStr
from datetime import datetime
import uuid


class UserBase(BaseModel):
    """Base user schema."""
    first_name: str = Field(..., min_length=1, max_length=100, description="User first name")
    last_name: str = Field(..., min_length=1, max_length=100, description="User last name")
    email: EmailStr = Field(..., description="User email address")
    phone: Optional[str] = Field(None, max_length=20, description="User phone number")
    department: Optional[str] = Field(None, max_length=100, description="User department")
    employee_id: Optional[str] = Field(None, max_length=50, description="Employee ID")
    position: Optional[str] = Field(None, max_length=100, description="User position")
    user_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user metadata")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, description="User password")
    profile_images: Optional[List[Dict[str, Any]]] = Field(None, description="Pre-assigned profile image names")


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=100)
    employee_id: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=100)
    user_metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    profile_images: Optional[List[Dict[str, Any]]] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: uuid.UUID
    is_active: bool
    is_verified: bool
    avatar_url: Optional[str]
    profile_images: Optional[List[Dict[str, Any]]] = []
    created_at: datetime
    updated_at: datetime
    has_face_data: bool
    
    class Config:
        from_attributes = True


class ProfileImageUpload(BaseModel):
    """Schema for profile image upload."""
    user_id: uuid.UUID
    image_url: str = Field(..., description="URL of the uploaded image")
    is_primary: bool = Field(default=False, description="Whether this is the primary profile image")


class ProfileImageResponse(BaseModel):
    """Schema for profile image response."""
    id: str
    url: str
    is_primary: bool
    uploaded_at: datetime
    size: Optional[int] = None
    content_type: Optional[str] = None


class ProfileImageListResponse(BaseModel):
    """Schema for profile image list response."""
    images: List[ProfileImageResponse]
    total: int
    primary_image: Optional[ProfileImageResponse] = None


class FaceEmbeddingBase(BaseModel):
    """Base face embedding schema."""
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Face detection confidence")
    face_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Face quality score")
    face_bbox: Optional[Dict[str, Any]] = Field(None, description="Face bounding box coordinates")
    landmarks: Optional[Dict[str, Any]] = Field(None, description="Facial landmarks")
    source_image: Optional[str] = Field(None, description="Source image URL")
    is_primary: bool = Field(default=False, description="Whether this is the primary face")


class FaceEmbeddingCreate(FaceEmbeddingBase):
    """Schema for creating a new face embedding."""
    user_id: uuid.UUID
    embedding: str = Field(..., description="Base64 encoded face embedding")
    face_image_url: Optional[str] = Field(None, description="Face image URL")


class FaceEmbeddingUpdate(BaseModel):
    """Schema for updating face embedding."""
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    face_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    face_bbox: Optional[Dict[str, Any]] = None
    landmarks: Optional[Dict[str, Any]] = None
    is_primary: Optional[bool] = None


class FaceEmbeddingResponse(FaceEmbeddingBase):
    """Schema for face embedding response."""
    id: uuid.UUID
    user_id: uuid.UUID
    embedding: str
    face_image_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserWithFaces(UserResponse):
    """Schema for user with face embeddings."""
    face_embeddings: List[FaceEmbeddingResponse] = []
    
    class Config:
        from_attributes = True


class FaceRecognitionRequest(BaseModel):
    """Schema for face recognition request."""
    image_data: str = Field(..., description="Base64 encoded image data")
    camera_id: Optional[uuid.UUID] = Field(None, description="Camera ID where face was detected")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Recognition confidence threshold")


class FaceRecognitionResponse(BaseModel):
    """Schema for face recognition response."""
    success: bool
    recognized_user: Optional[UserResponse] = None
    confidence_score: Optional[float] = None
    face_bbox: Optional[Dict[str, Any]] = None
    landmarks: Optional[Dict[str, Any]] = None
    message: str
    processing_time_ms: float


class FaceDetectionResponse(BaseModel):
    """Schema for face detection response."""
    id: uuid.UUID
    camera_id: uuid.UUID
    timestamp: str
    confidence_score: float
    face_bbox: Dict[str, Any]
    landmarks: Optional[Dict[str, Any]]
    recognized_user_id: Optional[uuid.UUID]
    recognition_confidence: Optional[float]
    face_image_url: Optional[str]
    full_frame_url: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AttendanceBase(BaseModel):
    """Base attendance schema."""
    user_id: uuid.UUID
    camera_id: uuid.UUID
    check_in_time: str = Field(..., description="Check-in timestamp (ISO format)")
    check_out_time: Optional[str] = Field(None, description="Check-out timestamp (ISO format)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence")
    location: Optional[str] = Field(None, max_length=255, description="Attendance location")


class AttendanceCreate(AttendanceBase):
    """Schema for creating a new attendance record."""
    pass


class AttendanceUpdate(BaseModel):
    """Schema for updating attendance record."""
    check_out_time: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    location: Optional[str] = Field(None, max_length=255)


class AttendanceResponse(AttendanceBase):
    """Schema for attendance response."""
    id: uuid.UUID
    face_embedding_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AttendanceWithUser(AttendanceResponse):
    """Schema for attendance with user information."""
    user: UserResponse
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Schema for paginated user list response."""
    data: List[UserResponse]
    total: int
    page: int
    size: int
    pages: int


class AttendanceListResponse(BaseModel):
    """Schema for paginated attendance list response."""
    attendances: List[AttendanceWithUser]
    total: int
    page: int
    size: int
    pages: int


class FaceUploadRequest(BaseModel):
    """Schema for face image upload request."""
    user_id: uuid.UUID
    image_data: str = Field(..., description="Base64 encoded image data")
    is_primary: bool = Field(default=False, description="Whether this is the primary face")
    source_description: Optional[str] = Field(None, max_length=500, description="Source description")


class FaceUploadResponse(BaseModel):
    """Schema for face image upload response."""
    success: bool
    face_embedding_id: uuid.UUID
    confidence_score: float
    face_quality_score: float
    message: str
    processing_time_ms: float
