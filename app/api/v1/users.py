"""User and face recognition API endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.orm import Session
import uuid
import base64
import os
from pathlib import Path
from datetime import datetime

from app.core.database import get_db
from app.models.database.user import User
from app.schemas.user import (
    UserCreate, UserUpdate, UserResponse, UserListResponse,
    FaceEmbeddingCreate, FaceEmbeddingResponse, FaceRecognitionRequest,
    FaceRecognitionResponse, FaceUploadRequest, FaceUploadResponse,
    AttendanceCreate, AttendanceUpdate, AttendanceResponse, AttendanceListResponse,
    ProfileImageUpload, ProfileImageResponse, ProfileImageListResponse
)
from app.services.face_recognition.face_service import face_recognition_service
from app.services.user.user_service import user_service

users_router = APIRouter()

# File upload configuration
UPLOAD_DIR = Path("uploads/users")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}


# User management endpoints
@users_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Create a new user."""
    try:
        user = await user_service.create_user(db, user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


@users_router.get("/", response_model=UserListResponse)
async def get_users(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    active_only: bool = Query(False, description="Return only active users"),
    db: Session = Depends(get_db)
):
    """Get paginated list of users."""
    try:
        users, total = await user_service.get_users(db, skip, limit, active_only)
        
        # Calculate pagination
        pages = (total + limit - 1) // limit
        page = (skip // limit) + 1
        
        return UserListResponse(
            data=users,
            total=total,
            page=page,
            size=len(users),
            pages=pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users: {str(e)}"
        )


@users_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get user by ID."""
    try:
        user = await user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


@users_router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    user_data: UserUpdate,
    db: Session = Depends(get_db)
):
    """Update user information."""
    try:
        user = await user_service.update_user(db, user_id, user_data)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )


@users_router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete user."""
    try:
        success = await user_service.delete_user(db, user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )


# Face recognition endpoints
@users_router.post("/face-recognition", response_model=FaceRecognitionResponse)
async def recognize_face(
    request: FaceRecognitionRequest,
    db: Session = Depends(get_db)
):
    """Recognize face from image data."""
    try:
        result = await face_recognition_service.recognize_face(db, request)
        return FaceRecognitionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face recognition failed: {str(e)}"
        )


@users_router.post("/upload-face", response_model=FaceUploadResponse)
async def upload_face_image(
    request: FaceUploadRequest,
    db: Session = Depends(get_db)
):
    """Upload and process face image for user."""
    try:
        result = await face_recognition_service.upload_face_image(db, request)
        return FaceUploadResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face upload failed: {str(e)}"
        )


@users_router.post("/upload-face-file", response_model=FaceUploadResponse)
async def upload_face_file(
    user_id: uuid.UUID,
    file: UploadFile = File(...),
    is_primary: bool = Query(False, description="Whether this is the primary face"),
    source_description: Optional[str] = Query(None, description="Source description"),
    db: Session = Depends(get_db)
):
    """Upload face image file for user."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read file content
        file_content = await file.read()
        image_data = base64.b64encode(file_content).decode('utf-8')
        
        # Create upload request
        request = FaceUploadRequest(
            user_id=user_id,
            image_data=image_data,
            is_primary=is_primary,
            source_description=source_description
        )
        
        result = await face_recognition_service.upload_face_image(db, request)
        return FaceUploadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face upload failed: {str(e)}"
        )


@users_router.get("/{user_id}/face-embeddings", response_model=List[FaceEmbeddingResponse])
async def get_user_face_embeddings(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get face embeddings for a user."""
    try:
        # Check if user exists
        user = await user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get face embeddings
        embeddings = await user_service.get_user_face_embeddings(db, user_id)
        return embeddings
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get face embeddings: {str(e)}"
        )


@users_router.delete("/face-embeddings/{embedding_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face_embedding(
    embedding_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete a face embedding."""
    try:
        success = await user_service.delete_face_embedding(db, embedding_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face embedding not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete face embedding: {str(e)}"
        )


# Attendance endpoints
@users_router.post("/attendance", response_model=AttendanceResponse, status_code=status.HTTP_201_CREATED)
async def create_attendance(
    attendance_data: AttendanceCreate,
    db: Session = Depends(get_db)
):
    """Create a new attendance record."""
    try:
        attendance = await user_service.create_attendance(db, attendance_data)
        return attendance
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create attendance: {str(e)}"
        )


@users_router.get("/attendance", response_model=AttendanceListResponse)
async def get_attendance(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    camera_id: Optional[uuid.UUID] = Query(None, description="Filter by camera ID"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    db: Session = Depends(get_db)
):
    """Get paginated list of attendance records."""
    try:
        attendances, total = await user_service.get_attendance(
            db, skip, limit, user_id, camera_id, date_from, date_to
        )
        
        # Calculate pagination
        pages = (total + limit - 1) // limit
        page = (skip // limit) + 1
        
        return AttendanceListResponse(
            attendances=attendances,
            total=total,
            page=page,
            size=len(attendances),
            pages=pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get attendance: {str(e)}"
        )


@users_router.get("/attendance/{attendance_id}", response_model=AttendanceResponse)
async def get_attendance_record(
    attendance_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get attendance record by ID."""
    try:
        attendance = await user_service.get_attendance_record(db, attendance_id)
        if not attendance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Attendance record not found"
            )
        return attendance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get attendance record: {str(e)}"
        )


@users_router.put("/attendance/{attendance_id}", response_model=AttendanceResponse)
async def update_attendance(
    attendance_id: uuid.UUID,
    attendance_data: AttendanceUpdate,
    db: Session = Depends(get_db)
):
    """Update attendance record."""
    try:
        attendance = await user_service.update_attendance(db, attendance_id, attendance_data)
        if not attendance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Attendance record not found"
            )
        return attendance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update attendance: {str(e)}"
        )


@users_router.delete("/attendance/{attendance_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_attendance(
    attendance_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete attendance record."""
    try:
        success = await user_service.delete_attendance(db, attendance_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Attendance record not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete attendance: {str(e)}"
        )


# User statistics endpoints
@users_router.get("/{user_id}/statistics")
async def get_user_statistics(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get user statistics."""
    try:
        stats = await user_service.get_user_statistics(db, user_id)
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user statistics: {str(e)}"
        )


@users_router.get("/attendance/statistics")
async def get_attendance_statistics(
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    db: Session = Depends(get_db)
):
    """Get attendance statistics."""
    try:
        stats = await user_service.get_attendance_statistics(db, date_from, date_to)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get attendance statistics: {str(e)}"
        )


# Profile image management endpoints
@users_router.post("/{user_id}/profile-images", response_model=ProfileImageResponse)
async def upload_profile_image(
    user_id: uuid.UUID,
    file: UploadFile = File(...),
    image_name: str = Form(..., description="Name of the image to upload"),
    db: Session = Depends(get_db)
):
    """Upload a profile image for a user with a specific name."""
    try:
        # Check if user exists
        user = await user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size must be less than {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File extension must be one of: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Use the provided image name with extension
        filename = f"{image_name}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Return success response - no user update needed
        return ProfileImageResponse(
            id=str(uuid.uuid4()),
            url=f"/uploads/users/{filename}",
            is_primary=False,
            uploaded_at=datetime.now(),
            size=len(file_content),
            content_type=file.content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload profile image: {str(e)}"
        )


@users_router.get("/{user_id}/profile-images", response_model=ProfileImageListResponse)
async def get_user_profile_images(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get all profile images for a user."""
    try:
        # Check if user exists
        user = await user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        profile_images = user.profile_images_list
        primary_image = next((img for img in profile_images if img.get('is_primary', False)), None)
        
        return ProfileImageListResponse(
            images=[
                ProfileImageResponse(
                    id=img["id"],
                    url=f"/uploads/users/{img['filename']}",
                    is_primary=img.get("is_primary", False),
                    uploaded_at=datetime.fromisoformat(img["uploaded_at"]),
                    size=img.get("size"),
                    content_type=img.get("content_type")
                )
                for img in profile_images
            ],
            total=len(profile_images),
            primary_image=ProfileImageResponse(
                id=primary_image["id"],
                url=f"/uploads/users/{primary_image['filename']}",
                is_primary=primary_image["is_primary"],
                uploaded_at=datetime.fromisoformat(primary_image["uploaded_at"]),
                size=primary_image.get("size"),
                content_type=primary_image.get("content_type")
            ) if primary_image else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile images: {str(e)}"
        )


@users_router.delete("/{user_id}/profile-images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile_image(
    user_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db)
):
    """Delete a profile image for a user."""
    try:
        # Check if user exists
        user = await user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        profile_images = user.profile_images_list
        image_to_delete = next((img for img in profile_images if img["id"] == image_id), None)
        
        if not image_to_delete:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile image not found"
            )
        
        # Remove image from list
        profile_images = [img for img in profile_images if img["id"] != image_id]
        
        # Update user
        update_data = UserUpdate(profile_images=profile_images)
        await user_service.update_user(db, user_id, update_data)
        
        # Delete file from disk
        try:
            file_path = UPLOAD_DIR / image_to_delete["filename"]
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            # Log error but don't fail the request
            print(f"Failed to delete file {file_path}: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete profile image: {str(e)}"
        )


@users_router.put("/{user_id}/profile-images/{image_id}/primary", response_model=ProfileImageResponse)
async def set_primary_profile_image(
    user_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db)
):
    """Set a profile image as primary."""
    try:
        # Check if user exists
        user = await user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        profile_images = user.profile_images_list
        target_image = next((img for img in profile_images if img["id"] == image_id), None)
        
        if not target_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile image not found"
            )
        
        # Update all images to remove primary flag
        for img in profile_images:
            img["is_primary"] = False
        
        # Set target image as primary
        target_image["is_primary"] = True
        
        # Update user
        update_data = UserUpdate(profile_images=profile_images)
        await user_service.update_user(db, user_id, update_data)
        
        return ProfileImageResponse(
            id=target_image["id"],
            url=f"/uploads/users/{target_image['filename']}",
            is_primary=target_image["is_primary"],
            uploaded_at=datetime.fromisoformat(target_image["uploaded_at"]),
            size=target_image.get("size"),
            content_type=target_image.get("content_type")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set primary profile image: {str(e)}"
        )
