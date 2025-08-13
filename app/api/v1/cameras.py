"""Camera API endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
import uuid

from app.core.database import get_db
from app.schemas.camera import (
    CameraCreate, CameraUpdate, CameraResponse, CameraListResponse,
    CameraHealthCheck, CameraStatistics, CameraTestConnection, CameraTestResponse
)
from app.services.camera.camera_service import camera_service
from app.services.streaming.stream_service import stream_manager

cameras_router = APIRouter()


@cameras_router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: Session = Depends(get_db)
):
    """Create a new camera."""
    try:
        camera = await camera_service.create_camera(db, camera_data)
        return camera
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create camera: {str(e)}"
        )


@cameras_router.get("/", response_model=CameraListResponse)
async def get_cameras(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    active_only: bool = Query(False, description="Return only active cameras"),
    db: Session = Depends(get_db)
):
    """Get paginated list of cameras."""
    try:
        cameras, total = await camera_service.get_cameras(db, skip, limit, active_only)
        
        # Calculate pagination
        pages = (total + limit - 1) // limit
        page = (skip // limit) + 1
        
        return CameraListResponse(
            cameras=cameras,
            total=total,
            page=page,
            size=len(cameras),
            pages=pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cameras: {str(e)}"
        )


@cameras_router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get camera by ID."""
    try:
        camera = await camera_service.get_camera(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found"
            )
        return camera
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get camera: {str(e)}"
        )


@cameras_router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: uuid.UUID,
    camera_data: CameraUpdate,
    db: Session = Depends(get_db)
):
    """Update camera information."""
    try:
        camera = await camera_service.update_camera(db, camera_id, camera_data)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found"
            )
        return camera
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update camera: {str(e)}"
        )


@cameras_router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete camera."""
    try:
        success = await camera_service.delete_camera(db, camera_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete camera: {str(e)}"
        )


@cameras_router.post("/test-connection", response_model=CameraTestResponse)
async def test_camera_connection(
    test_data: CameraTestConnection
):
    """Test RTSP camera connection."""
    try:
        result = await camera_service.test_camera_connection(test_data)
        return CameraTestResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test camera connection: {str(e)}"
        )


@cameras_router.get("/{camera_id}/health", response_model=CameraHealthCheck)
async def get_camera_health(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get camera health status."""
    try:
        health = await camera_service.get_camera_health(db, camera_id)
        if "error" in health:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=health["error"]
            )
        return CameraHealthCheck(**health)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get camera health: {str(e)}"
        )


@cameras_router.get("/{camera_id}/statistics", response_model=CameraStatistics)
async def get_camera_statistics(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get camera statistics."""
    try:
        stats = await camera_service.get_camera_statistics(db, camera_id)
        if "error" in stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=stats["error"]
            )
        return CameraStatistics(**stats)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get camera statistics: {str(e)}"
        )


@cameras_router.post("/{camera_id}/start-stream", status_code=status.HTTP_200_OK)
async def start_camera_stream(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Start streaming from camera."""
    try:
        # Check if camera exists
        camera = await camera_service.get_camera(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found"
            )
        
        # Start stream
        success = await camera_service.start_camera_stream(camera_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start camera stream"
            )
        
        return {"message": f"Camera stream started successfully for camera {camera_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start camera stream: {str(e)}"
        )


@cameras_router.post("/{camera_id}/stop-stream", status_code=status.HTTP_200_OK)
async def stop_camera_stream(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Stop streaming from camera."""
    try:
        # Check if camera exists
        camera = await camera_service.get_camera(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found"
            )
        
        # Stop stream
        success = await camera_service.stop_camera_stream(camera_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop camera stream"
            )
        
        return {"message": f"Camera stream stopped successfully for camera {camera_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop camera stream: {str(e)}"
        )


@cameras_router.get("/{camera_id}/stream-status")
async def get_camera_stream_status(
    camera_id: uuid.UUID
):
    """Get camera stream status."""
    try:
        status_info = stream_manager.get_stream_status(camera_id)
        if not status_info:
            return {"message": "No active stream for this camera"}
        return status_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stream status: {str(e)}"
        )


@cameras_router.get("/streams/status")
async def get_all_streams_status():
    """Get status of all active streams."""
    try:
        return stream_manager.get_all_streams_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get streams status: {str(e)}"
        )


@cameras_router.post("/{camera_id}/restart-stream", status_code=status.HTTP_200_OK)
async def restart_camera_stream(
    camera_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Restart camera stream."""
    try:
        # Check if camera exists
        camera = await camera_service.get_camera(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found"
            )
        
        # Stop stream if running
        await camera_service.stop_camera_stream(camera_id)
        
        # Wait a bit
        import asyncio
        await asyncio.sleep(2)
        
        # Start stream again
        success = await camera_service.start_camera_stream(camera_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to restart camera stream"
            )
        
        return {"message": f"Camera stream restarted successfully for camera {camera_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart camera stream: {str(e)}"
        )
