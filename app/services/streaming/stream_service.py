"""Streaming service for handling RTSP camera streams."""
import cv2
import asyncio
import threading
import time
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from loguru import logger
import uuid
from datetime import datetime
import numpy as np
import base64
from sqlalchemy import and_, func

from app.models.database.camera import Camera, CameraStream
from app.services.face_recognition.face_service import face_recognition_service
from app.core.config import settings
from app.core.celery_app import celery_app


class StreamManager:
    """Manages multiple camera streams."""
    
    def __init__(self):
        self.active_streams: Dict[uuid.UUID, 'CameraStreamProcessor'] = {}
        self.stream_lock = threading.Lock()
    
    def start_stream(self, camera_id: uuid.UUID, camera: Camera) -> bool:
        """Start a new camera stream."""
        try:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    logger.info(f"Stream already active for camera {camera_id}")
                    return True
                
                # Create new stream processor
                stream_processor = CameraStreamProcessor(camera_id, camera)
                if stream_processor.start():
                    self.active_streams[camera_id] = stream_processor
                    logger.info(f"Started stream for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to start stream for camera {camera_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error starting stream for camera {camera_id}: {str(e)}")
            return False
    
    def stop_stream(self, camera_id: uuid.UUID) -> bool:
        """Stop a camera stream."""
        try:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    stream_processor = self.active_streams[camera_id]
                    stream_processor.stop()
                    del self.active_streams[camera_id]
                    logger.info(f"Stopped stream for camera {camera_id}")
                    return True
                return True
                
        except Exception as e:
            logger.error(f"Error stopping stream for camera {camera_id}: {str(e)}")
            return False
    
    def get_stream_status(self, camera_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get stream status for a camera."""
        try:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    return self.active_streams[camera_id].get_status()
                return None
        except Exception as e:
            logger.error(f"Error getting stream status for camera {camera_id}: {str(e)}")
            return None
    
    def get_all_streams_status(self) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Get status of all active streams."""
        try:
            with self.stream_lock:
                return {
                    camera_id: stream.get_status()
                    for camera_id, stream in self.active_streams.items()
                }
        except Exception as e:
            logger.error(f"Error getting all stream statuses: {str(e)}")
            return {}


class CameraStreamProcessor:
    """Processes individual camera streams."""
    
    def __init__(self, camera_id: uuid.UUID, camera: Camera):
        self.camera_id = camera_id
        self.camera = camera
        self.is_running = False
        self.thread = None
        self.cap = None
        
        # Stream metrics
        self.frame_count = 0
        self.faces_detected = 0
        self.errors_count = 0
        self.start_time = None
        self.last_frame_time = None
        self.fps = 0
        self.memory_usage = 0
        
        # Face detection settings
        self.detection_interval = max(1, int(30 / camera.frame_rate))  # Detect every N frames
        self.last_detection_time = 0
        
        # Database session
        self.db_session = None
    
    def start(self) -> bool:
        """Start the stream processor."""
        try:
            # Build RTSP URL with credentials
            rtsp_url = self.camera.rtsp_url
            if self.camera.username and self.camera.password:
                if "rtsp://" in rtsp_url:
                    protocol, rest = rtsp_url.split("://", 1)
                    host_port, path = rest.split("/", 1)
                    rtsp_url = f"{protocol}://{self.camera.username}:{self.camera.password}@{host_port}/{path}"
            
            # Open RTSP stream
            self.cap = cv2.VideoCapture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera.frame_rate)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream for camera {self.camera_id}")
                return False
            
            # Start processing thread
            self.is_running = True
            self.thread = threading.Thread(target=self._process_stream, daemon=True)
            self.thread.start()
            
            self.start_time = datetime.utcnow()
            logger.info(f"Started stream processor for camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream processor for camera {self.camera_id}: {str(e)}")
            return False
    
    def stop(self):
        """Stop the stream processor."""
        try:
            self.is_running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
            
            if self.cap:
                self.cap.release()
            
            logger.info(f"Stopped stream processor for camera {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error stopping stream processor for camera {self.camera_id}: {str(e)}")
    
    def _process_stream(self):
        """Main stream processing loop."""
        frame_times = []
        
        try:
            while self.is_running:
                start_frame_time = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.errors_count += 1
                    time.sleep(0.1)
                    continue
                
                # Update metrics
                self.frame_count += 1
                self.last_frame_time = datetime.utcnow()
                
                # Calculate FPS
                frame_times.append(start_frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                if len(frame_times) > 1:
                    self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                
                # Face detection (every N frames)
                if self.frame_count % self.detection_interval == 0:
                    self._process_frame_for_faces(frame)
                
                # Control frame rate
                elapsed = time.time() - start_frame_time
                target_delay = 1.0 / self.camera.frame_rate
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                
        except Exception as e:
            logger.error(f"Error in stream processing for camera {self.camera_id}: {str(e)}")
            self.errors_count += 1
        finally:
            self.is_running = False
    
    def _process_frame_for_faces(self, frame: np.ndarray):
        """Process frame for face detection and recognition."""
        try:
            # Skip if too soon since last detection
            current_time = time.time()
            if current_time - self.last_detection_time < 1.0:  # Max 1 detection per second
                return
            
            # Encode frame for face recognition
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Create face recognition request
            from app.schemas.user import FaceRecognitionRequest
            
            request = FaceRecognitionRequest(
                image_data=image_data,
                camera_id=self.camera_id,
                confidence_threshold=settings.FACE_RECOGNITION_THRESHOLD
            )
            
            # Process face recognition asynchronously
            asyncio.create_task(self._async_face_recognition(request))
            
            self.last_detection_time = current_time
            
        except Exception as e:
            logger.error(f"Error processing frame for faces: {str(e)}")
            self.errors_count += 1
    
    async def _async_face_recognition(self, request):
        """Process face recognition asynchronously."""
        try:
            # Get database session
            from app.core.database import SessionLocal
            db = SessionLocal()
            
            try:
                # Process face recognition
                result = await face_recognition_service.recognize_face(db, request)
                
                if result["success"]:
                    self.faces_detected += 1
                    logger.info(f"Face recognized: {result['recognized_user'].full_name} "
                              f"with confidence {result['confidence_score']:.3f}")
                    
                    # Create attendance record
                    await self._create_attendance_record(db, result)
                else:
                    logger.debug(f"No face recognized in frame: {result['message']}")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in async face recognition: {str(e)}")
            self.errors_count += 1
    
    async def _create_attendance_record(self, db: Session, recognition_result: Dict[str, Any]):
        """Create attendance record for recognized user."""
        try:
            from app.models.database.user import Attendance
            from app.schemas.user import AttendanceCreate
            
            # Check if user already has attendance today
            today = datetime.utcnow().date()
            existing_attendance = db.query(Attendance).filter(
                and_(
                    Attendance.user_id == recognition_result["recognized_user"].id,
                    Attendance.camera_id == self.camera_id,
                    func.date(Attendance.created_at) == today
                )
            ).first()
            
            if existing_attendance:
                # Update check-out time
                existing_attendance.check_out_time = datetime.utcnow().isoformat()
                existing_attendance.confidence_score = max(
                    existing_attendance.confidence_score,
                    recognition_result["confidence_score"]
                )
            else:
                # Create new attendance record
                attendance_data = AttendanceCreate(
                    user_id=recognition_result["recognized_user"].id,
                    camera_id=self.camera_id,
                    check_in_time=datetime.utcnow().isoformat(),
                    confidence_score=recognition_result["confidence_score"],
                    location=self.camera.location
                )
                
                attendance = Attendance(**attendance_data.dict())
                db.add(attendance)
            
            db.commit()
            logger.info(f"Attendance record created/updated for user "
                       f"{recognition_result['recognized_user'].full_name}")
            
        except Exception as e:
            logger.error(f"Failed to create attendance record: {str(e)}")
            db.rollback()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stream status."""
        try:
            uptime = 0
            if self.start_time:
                uptime = int((datetime.utcnow() - self.start_time).total_seconds())
            
            return {
                "camera_id": str(self.camera_id),
                "is_running": self.is_running,
                "frame_count": self.frame_count,
                "faces_detected": self.faces_detected,
                "errors_count": self.errors_count,
                "fps": round(self.fps, 2),
                "uptime_seconds": uptime,
                "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
                "memory_usage_mb": self.memory_usage
            }
        except Exception as e:
            logger.error(f"Error getting stream status: {str(e)}")
            return {"error": str(e)}


# Global stream manager instance
stream_manager = StreamManager()


# Celery tasks
@celery_app.task(name="app.services.streaming.stream_service.start_camera_stream")
def start_camera_stream(camera_id_str: str):
    """Celery task to start camera stream."""
    try:
        from app.core.database import SessionLocal
        
        camera_id = uuid.UUID(camera_id_str)
        db = SessionLocal()
        
        try:
            # Get camera
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                logger.error(f"Camera not found: {camera_id}")
                return False
            
            # Start stream
            success = stream_manager.start_stream(camera_id, camera)
            
            if success:
                # Create stream record
                stream = CameraStream(
                    camera_id=camera_id,
                    is_active=True,
                    started_at=datetime.utcnow().isoformat()
                )
                db.add(stream)
                db.commit()
                
                logger.info(f"Started camera stream for camera {camera_id}")
                return True
            else:
                logger.error(f"Failed to start camera stream for camera {camera_id}")
                return False
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error starting camera stream {camera_id_str}: {str(e)}")
        return False


@celery_app.task(name="app.services.streaming.stream_service.stop_camera_stream")
def stop_camera_stream(camera_id_str: str):
    """Celery task to stop camera stream."""
    try:
        camera_id = uuid.UUID(camera_id_str)
        
        # Stop stream
        success = stream_manager.stop_stream(camera_id)
        
        if success:
            # Update stream record
            from app.core.database import SessionLocal
            db = SessionLocal()
            
            try:
                stream = db.query(CameraStream).filter(
                    and_(
                        CameraStream.camera_id == camera_id,
                        CameraStream.is_active == True
                    )
                ).first()
                
                if stream:
                    stream.is_active = False
                    db.commit()
                
            finally:
                db.close()
            
            logger.info(f"Stopped camera stream for camera {camera_id}")
            return True
        else:
            logger.error(f"Failed to stop camera stream for camera {camera_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error stopping camera stream {camera_id_str}: {str(e)}")
        return False
