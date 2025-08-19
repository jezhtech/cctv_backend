"""User and face recognition database models."""
from sqlalchemy import Column, Integer, String, Text, Boolean, JSON, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .base import Base, TimestampMixin, SoftDeleteMixin


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model for storing user information."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    phone = Column(String(20), nullable=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Profile
    avatar_url = Column(String(500), nullable=True)
    profile_images = Column(JSON, nullable=True)  # Array of profile image URLs
    department = Column(String(100), nullable=True)
    employee_id = Column(String(50), nullable=True, unique=True)
    position = Column(String(100), nullable=True)
    
    # Additional Information
    user_metadata = Column(JSON, nullable=True)  # Store additional user data
    
    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan", lazy="dynamic")
    attendances = relationship("Attendance", back_populates="user", cascade="all, delete-orphan", lazy="dynamic")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', name='{self.first_name} {self.last_name}')>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def has_face_data(self) -> bool:
        """Check if user has face embeddings."""
        try:
            return self.face_embeddings.count() > 0
        except:
            return False
    
    @property
    def profile_images_list(self) -> list:
        """Get profile images as a list."""
        try:
            if self.profile_images:
                if isinstance(self.profile_images, list):
                    return self.profile_images
                elif isinstance(self.profile_images, str):
                    # Handle case where it might be stored as a JSON string
                    import json
                    try:
                        return json.loads(self.profile_images)
                    except:
                        return []
                else:
                    return []
            return []
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting profile_images_list: {e}")
            return []


class FaceEmbedding(Base, TimestampMixin):
    """Face embedding model for storing facial recognition data."""
    
    __tablename__ = "face_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Face Data
    embedding = Column(Text, nullable=False)  # Store as base64 encoded string
    face_image_url = Column(String(500), nullable=True)
    
    # Quality Metrics
    confidence_score = Column(Float, nullable=False)
    face_quality_score = Column(Float, nullable=True)
    
    # Detection Details
    face_bbox = Column(JSON, nullable=True)  # Bounding box coordinates
    landmarks = Column(JSON, nullable=True)  # Facial landmarks
    
    # Metadata
    source_image = Column(String(500), nullable=True)
    is_primary = Column(Boolean, default=False, nullable=False)  # Primary face for user
    
    # Relationships
    user = relationship("User", back_populates="face_embeddings")
    
    def __repr__(self):
        return f"<FaceEmbedding(id={self.id}, user_id={self.user_id}, confidence={self.confidence_score})>"


class Attendance(Base, TimestampMixin):
    """Attendance model for tracking user attendance."""
    
    __tablename__ = "attendance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=False)
    
    # Attendance Details
    check_in_time = Column(String(255), nullable=False)  # ISO timestamp
    check_out_time = Column(String(255), nullable=True)  # ISO timestamp
    
    # Recognition Details
    confidence_score = Column(Float, nullable=False)
    face_embedding_id = Column(UUID(as_uuid=True), ForeignKey("face_embeddings.id"), nullable=True)
    
    # Location
    location = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="attendances", lazy="joined")
    camera = relationship("Camera", back_populates="attendances", lazy="joined")
    face_embedding = relationship("FaceEmbedding", lazy="joined")
    
    def __repr__(self):
        return f"<Attendance(id={self.id}, user_id={self.user_id}, check_in='{self.check_in_time}')>"


class FaceDetection(Base, TimestampMixin):
    """Face detection model for tracking detected faces."""
    
    __tablename__ = "face_detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=False)
    
    # Detection Details
    timestamp = Column(String(255), nullable=False)  # ISO timestamp
    confidence_score = Column(Float, nullable=False)
    
    # Face Information
    face_bbox = Column(JSON, nullable=False)  # Bounding box coordinates
    landmarks = Column(JSON, nullable=True)  # Facial landmarks
    
    # Recognition Results
    recognized_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    recognition_confidence = Column(Float, nullable=True)
    
    # Image Data
    face_image_url = Column(String(500), nullable=True)
    full_frame_url = Column(String(500), nullable=True)
    
    # Relationships
    camera = relationship("Camera", back_populates="detections", lazy="joined")
    recognized_user = relationship("User", lazy="joined")
    
    def __repr__(self):
        return f"<FaceDetection(id={self.id}, camera_id={self.camera_id}, timestamp='{self.timestamp}')>"
