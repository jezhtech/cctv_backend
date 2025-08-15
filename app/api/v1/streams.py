"""Streaming API endpoints for handling camera streams directly in FastAPI."""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from sqlalchemy.orm import Session
import uuid
from loguru import logger

from app.core.database import SessionLocal
from app.services.camera.camera_service import camera_service
from app.services.streaming.stream_service import stream_manager

streams_router = APIRouter()


@streams_router.post("/{camera_id}/start")
async def start_camera_stream(camera_id: str):
    """Start a camera stream directly in FastAPI."""
    try:
        camera_uuid = uuid.UUID(camera_id)
        
        # Get camera from database
        db = SessionLocal()
        try:
            camera = await camera_service.get_camera(db, camera_uuid)
            if not camera:
                raise HTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail="Camera not found"
                )
        finally:
            db.close()
        
        # Start stream
        success = await stream_manager.start_stream(camera_uuid, camera)
        
        if success:
            return {
                "success": True,
                "message": f"Started stream for camera {camera.name}",
                "camera_id": str(camera_uuid)
            }
        else:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start stream"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting stream for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@streams_router.post("/{camera_id}/stop")
async def stop_camera_stream(camera_id: str):
    """Stop a camera stream."""
    try:
        camera_uuid = uuid.UUID(camera_id)
        
        # Stop stream
        success = await stream_manager.stop_stream(camera_uuid)
        
        if success:
            return {
                "success": True,
                "message": f"Stopped stream for camera {camera_id}",
                "camera_id": str(camera_uuid)
            }
        else:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop stream"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping stream for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@streams_router.get("/{camera_id}/status")
async def get_stream_status(camera_id: str):
    """Get streaming status for a camera."""
    try:
        camera_uuid = uuid.UUID(camera_id)
        
        # Get stream status
        stream_status = stream_manager.get_stream_status(camera_uuid)
        
        if stream_status:
            return stream_status
        else:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Stream not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stream status for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@streams_router.get("/status")
async def get_all_streams_status():
    """Get status of all active streams."""
    try:
        # Get all stream statuses
        statuses = stream_manager.get_all_streams_status()
        
        return {
            "total_streams": len(statuses),
            "active_streams": len([s for s in statuses.values() if s.get("is_running", False)]),
            "streams": statuses
        }
        
    except Exception as e:
        logger.error(f"Error getting all stream statuses: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@streams_router.post("/stop-all")
async def stop_all_streams():
    """Stop all active camera streams."""
    try:
        # Stop all streams
        await stream_manager.stop_all_streams()
        
        return {
            "success": True,
            "message": "All streams stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping all streams: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@streams_router.get("/{camera_id}/mjpeg")
async def get_mjpeg_stream(camera_id: str):
    """Get MJPEG stream for a camera from in-memory storage."""
    try:
        from fastapi.responses import StreamingResponse
        import asyncio
        
        camera_uuid = uuid.UUID(camera_id)
        
        # Check if stream is active
        if not stream_manager.is_stream_active(camera_uuid):
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Stream not found or not active"
            )
        
        # Return MJPEG stream from in-memory storage
        async def generate_mjpeg_from_memory():
            while True:
                try:
                    # Get latest frame from in-memory storage
                    processor = stream_manager.get_processor(camera_uuid)
                    if processor and processor.is_running:
                        # Get the latest frame from the processor
                        if hasattr(processor, 'latest_frame') and processor.latest_frame is not None:
                            # Encode frame as JPEG
                            import cv2
                            _, buffer = cv2.imencode('.jpg', processor.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            frame_bytes = buffer.tobytes()
                            
                            # MJPEG format
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            
                            # Minimal delay for real-time streaming
                            await asyncio.sleep(0.033)  # ~30 FPS for real-time
                        else:
                            # No frame available, minimal wait
                            await asyncio.sleep(0.01)
                    else:
                        # Stream not active, minimal wait
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    logger.error(f"Error generating MJPEG from memory for camera {camera_id}: {str(e)}")
                    await asyncio.sleep(0.1)
        
        return StreamingResponse(
            generate_mjpeg_from_memory(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving MJPEG stream from memory for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@streams_router.get("/{camera_id}/latest-frame")
async def get_latest_frame(camera_id: str):
    """Get the latest frame as a single image for debugging."""
    try:
        from fastapi.responses import Response
        
        camera_uuid = uuid.UUID(camera_id)
        
        # Get latest frame from in-memory storage
        processor = stream_manager.get_processor(camera_uuid)
        if processor and processor.is_running and hasattr(processor, 'latest_frame') and processor.latest_frame is not None:
            # Encode frame as JPEG
            import cv2
            _, buffer = cv2.imencode('.jpg', processor.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_bytes = buffer.tobytes()
            
            return Response(
                content=frame_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="No frame available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving latest frame for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
