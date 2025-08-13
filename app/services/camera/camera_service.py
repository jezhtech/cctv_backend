"""Camera service for managing RTSP cameras."""
import asyncio
import cv2
import time
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from loguru import logger
import uuid
from datetime import datetime, timedelta

from app.models.database.camera import Camera, CameraStream
from app.schemas.camera import CameraCreate, CameraUpdate, CameraTestConnection
from app.core.celery_app import celery_app
from app.services.streaming.stream_service import StreamManager


class CameraService:
    """Service for managing RTSP cameras."""
    
    def __init__(self):
        self.stream_manager = StreamManager()
        self.active_streams: Dict[uuid.UUID, Any] = {}
    
    async def create_camera(self, db: Session, camera_data: CameraCreate) -> Camera:
        """Create a new camera."""
        try:
            # Test connection before creating
            test_result = await self.test_camera_connection(camera_data)
            if not test_result["success"]:
                raise ValueError(f"Camera connection test failed: {test_result['message']}")
            
            # Create camera instance
            camera = Camera(**camera_data.dict())
            db.add(camera)
            db.commit()
            db.refresh(camera)
            
            logger.info(f"Camera created successfully: {camera.id}")
            return camera
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create camera: {str(e)}")
            raise
    
    async def get_camera(self, db: Session, camera_id: uuid.UUID) -> Optional[Camera]:
        """Get camera by ID."""
        try:
            camera = db.query(Camera).filter(
                and_(Camera.id == camera_id, Camera.is_deleted == False)
            ).first()
            return camera
        except Exception as e:
            logger.error(f"Failed to get camera {camera_id}: {str(e)}")
            return None
    
    async def get_cameras(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = False
    ) -> Tuple[List[Camera], int]:
        """Get paginated list of cameras."""
        try:
            query = db.query(Camera).filter(Camera.is_deleted == False)
            
            if active_only:
                query = query.filter(Camera.is_active == True)
            
            total = query.count()
            cameras = query.offset(skip).limit(limit).all()
            
            return cameras, total
        except Exception as e:
            logger.error(f"Failed to get cameras: {str(e)}")
            return [], 0
    
    async def update_camera(
        self, 
        db: Session, 
        camera_id: uuid.UUID, 
        camera_data: CameraUpdate
    ) -> Optional[Camera]:
        """Update camera information."""
        try:
            camera = await self.get_camera(db, camera_id)
            if not camera:
                return None
            
            # Update fields
            update_data = camera_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(camera, field, value)
            
            camera.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(camera)
            
            logger.info(f"Camera updated successfully: {camera_id}")
            return camera
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update camera {camera_id}: {str(e)}")
            return None
    
    async def delete_camera(self, db: Session, camera_id: uuid.UUID) -> bool:
        """Soft delete camera."""
        try:
            camera = await self.get_camera(db, camera_id)
            if not camera:
                return False
            
            # Stop any active streams
            await self.stop_camera_stream(camera_id)
            
            # Soft delete
            camera.is_deleted = True
            camera.deleted_at = datetime.utcnow()
            camera.is_active = False
            
            db.commit()
            logger.info(f"Camera deleted successfully: {camera_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete camera {camera_id}: {str(e)}")
            return False
    
    async def test_camera_connection(self, camera_data: CameraTestConnection) -> Dict[str, Any]:
        """Test RTSP camera connection."""
        start_time = time.time()
        cap = None
        
        try:
            # Build RTSP URL with credentials if provided
            rtsp_url = camera_data.rtsp_url
            if camera_data.username and camera_data.password:
                # Insert credentials into RTSP URL
                if "rtsp://" in rtsp_url:
                    protocol, rest = rtsp_url.split("://", 1)
                    host_port, path = rest.split("/", 1)
                    rtsp_url = f"{protocol}://{camera_data.username}:{camera_data.password}@{host_port}/{path}"
            
            # Open RTSP stream
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                return {
                    "success": False,
                    "message": "Failed to open RTSP stream",
                    "error_details": "Stream could not be opened"
                }
            
            # Read a few frames to test
            frame_count = 0
            max_frames = 10
            start_frame_time = time.time()
            
            for _ in range(max_frames):
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    if frame_count == 1:
                        height, width = frame.shape[:2]
                        resolution = {"width": width, "height": height}
                
                # Limit test time
                if time.time() - start_frame_time > camera_data.timeout:
                    break
            
            processing_time = (time.time() - start_time) * 1000
            
            if frame_count > 0:
                frame_rate = frame_count / (time.time() - start_frame_time)
                return {
                    "success": True,
                    "message": f"Successfully connected to camera. Processed {frame_count} frames.",
                    "frame_count": frame_count,
                    "resolution": resolution,
                    "frame_rate": round(frame_rate, 2),
                    "processing_time_ms": round(processing_time, 2)
                }
            else:
                return {
                    "success": False,
                    "message": "No frames received from camera",
                    "error_details": "Stream opened but no frames received"
                }
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Camera connection test failed: {str(e)}")
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}",
                "error_details": str(e),
                "processing_time_ms": round(processing_time, 2)
            }
        finally:
            if cap:
                cap.release()
    
    async def start_camera_stream(self, camera_id: uuid.UUID) -> bool:
        """Start streaming from a camera."""
        try:
            # Check if stream is already active
            if camera_id in self.active_streams:
                logger.info(f"Stream already active for camera {camera_id}")
                return True
            
            # Start stream using Celery task
            task = celery_app.send_task(
                "app.services.streaming.stream_service.start_camera_stream",
                args=[str(camera_id)]
            )
            
            logger.info(f"Started camera stream task for camera {camera_id}: {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera stream {camera_id}: {str(e)}")
            return False
    
    async def stop_camera_stream(self, camera_id: uuid.UUID) -> bool:
        """Stop streaming from a camera."""
        try:
            if camera_id in self.active_streams:
                # Stop stream using Celery task
                task = celery_app.send_task(
                    "app.services.streaming.stream_service.stop_camera_stream",
                    args=[str(camera_id)]
                )
                
                logger.info(f"Stopped camera stream for camera {camera_id}: {task.id}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop camera stream {camera_id}: {str(e)}")
            return False
    
    async def get_camera_health(self, db: Session, camera_id: uuid.UUID) -> Dict[str, Any]:
        """Get camera health status."""
        try:
            camera = await self.get_camera(db, camera_id)
            if not camera:
                return {"error": "Camera not found"}
            
            start_time = time.time()
            
            # Test RTSP connection
            test_data = CameraTestConnection(
                rtsp_url=camera.rtsp_url,
                username=camera.username,
                password=camera.password
            )
            test_result = await self.test_camera_connection(test_data)
            
            response_time = (time.time() - start_time) * 1000
            
            # Get stream status
            active_stream = db.query(CameraStream).filter(
                and_(
                    CameraStream.camera_id == camera_id,
                    CameraStream.is_active == True
                )
            ).first()
            
            return {
                "camera_id": camera_id,
                "is_online": test_result["success"],
                "rtsp_accessible": test_result["success"],
                "frame_rate": test_result.get("frame_rate", 0.0),
                "last_frame_time": active_stream.last_frame_at if active_stream else None,
                "error_message": None if test_result["success"] else test_result.get("message"),
                "response_time_ms": round(response_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get camera health {camera_id}: {str(e)}")
            return {
                "camera_id": camera_id,
                "is_online": False,
                "rtsp_accessible": False,
                "frame_rate": 0.0,
                "last_frame_time": None,
                "error_message": str(e),
                "response_time_ms": 0.0
            }
    
    async def get_camera_statistics(self, db: Session, camera_id: uuid.UUID) -> Dict[str, Any]:
        """Get camera statistics."""
        try:
            camera = await self.get_camera(db, camera_id)
            if not camera:
                return {"error": "Camera not found"}
            
            # Get detection counts
            from app.models.database.user import FaceDetection, Attendance
            
            total_detections = db.query(FaceDetection).filter(
                FaceDetection.camera_id == camera_id
            ).count()
            
            total_recognitions = db.query(FaceDetection).filter(
                and_(
                    FaceDetection.camera_id == camera_id,
                    FaceDetection.recognized_user_id.isnot(None)
                )
            ).count()
            
            # Get average confidence
            avg_confidence = db.query(FaceDetection.confidence_score).filter(
                FaceDetection.camera_id == camera_id
            ).scalar()
            
            # Get uptime
            active_stream = db.query(CameraStream).filter(
                and_(
                    CameraStream.camera_id == camera_id,
                    CameraStream.is_active == True
                )
            ).first()
            
            uptime = 0
            if active_stream and active_stream.started_at:
                start_time = datetime.fromisoformat(active_stream.started_at)
                uptime = int((datetime.utcnow() - start_time).total_seconds())
            
            # Get last activity
            last_detection = db.query(FaceDetection).filter(
                FaceDetection.camera_id == camera_id
            ).order_by(desc(FaceDetection.created_at)).first()
            
            last_activity = last_detection.created_at if last_detection else None
            
            # Calculate error rate
            total_errors = active_stream.errors_count if active_stream else 0
            total_frames = active_stream.total_frames_processed if active_stream else 1
            error_rate = (total_errors / total_frames) * 100 if total_frames > 0 else 0
            
            return {
                "camera_id": camera_id,
                "total_detections": total_detections,
                "total_recognitions": total_recognitions,
                "average_confidence": round(avg_confidence or 0.0, 3),
                "uptime_seconds": uptime,
                "last_activity": last_activity,
                "error_rate": round(error_rate, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get camera statistics {camera_id}: {str(e)}")
            return {"error": str(e)}


# Global camera service instance
camera_service = CameraService()


# Celery tasks
@celery_app.task(name="app.services.camera.camera_service.check_camera_health")
def check_camera_health():
    """Periodic task to check camera health."""
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Get all active cameras
        cameras, _ = camera_service.get_cameras(db, active_only=True)
        
        for camera in cameras:
            try:
                health = camera_service.get_camera_health(db, camera.id)
                logger.info(f"Camera {camera.id} health: {health}")
            except Exception as e:
                logger.error(f"Failed to check health for camera {camera.id}: {str(e)}")
    finally:
        db.close()
