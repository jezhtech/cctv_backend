"""Streaming service for handling RTSP camera streams directly in FastAPI."""
import cv2
import time
import asyncio
import uuid
import numpy as np
import math
from loguru import logger
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.database.camera import Camera
from app.models.database.user import FaceDetection
from app.services.face_recognition.face_service import face_recognition_service
from app.services.redis_service import redis_service


def safe_float(value: float, default: float = 0.0) -> float:
    """Convert value to safe float for JSON serialization."""
    if value is None:
        return default
    if math.isinf(value) or math.isnan(value):
        return default
    return float(value)


class StreamProcessor:
    """Processes individual camera streams directly in FastAPI."""
    
    def __init__(self, camera_id: uuid.UUID, camera: Camera):
        self.camera_id = camera_id
        self.camera = camera
        self.is_running = False
        self.task = None
        self.start_time = None
        
        # Metrics
        self.frame_count = 0
        self.fps = 0.0
        self.last_frame_time = None
        self.errors_count = 0
        self.faces_detected = 0
        self.last_detection_time = 0
        
        # OpenCV streaming
        self.cap = None
        self.frame_buffer = []
        self.max_buffer_size = 10  # Keep last 10 frames for slower processing
        
        # In-memory frame storage (replaces Redis)
        self.latest_frame = None
        self.frame_timestamp = None
        
        # Stream info
        self.stream_info = {}
        self.detection_interval = 60  # Process every 60 frames for high FPS target
        
        # Face detection settings - reduced frequency for slower processing
        self.detection_interval = max(10, int(120 / camera.frame_rate))  # Detect every N frames for slower processing
        self.last_detection_time = 0
        
        # Performance monitoring for high FPS
        self.performance_metrics = {
            'avg_frame_time': 0.0,
            'max_frame_time': 0.0,
            'min_frame_time': 0.0,  # Initialize to 0 instead of inf
            'frame_time_history': []
        }
        
        # Update frequency for in-memory storage
        self.update_counter = 0
        self.update_frequency = 1  # Update every frame for high FPS streaming
        
        # Database session
        self.db_session = None
        
        # Async event loop for FastAPI integration
        self.loop = None
        self._stop_event = asyncio.Event()
        
        # Face detection metrics
        self.face_detection_metrics = {
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'last_recognition_time': None,
            'recognition_accuracy': 0.0,
            'detection_history': [],
            'active_faces': {},  # Track active faces to prevent duplicates
            'face_tracking_timeout': 5.0  # Seconds to consider a face as "new"
        }
    
    async def start_async(self):
        """Start the stream processor asynchronously in FastAPI."""
        try:
            logger.info(f"Starting FastAPI stream processor for camera {self.camera_id}")
            
            # Get the current event loop
            self.loop = asyncio.get_event_loop()
            
            # Test RTSP connection first
            if not await self.test_rtsp_connection_async():
                logger.error(f"RTSP connection test failed for camera {self.camera_id}")
                return False
            
            # Start OpenCV streaming
            await self._start_opencv_stream_async()
            
            # Initialize face recognition embeddings
            await self._initialize_face_recognition()
            
            # Start processing task
            self.is_running = True
            self._stop_event.clear()
            self.task = asyncio.create_task(self._process_stream_async())
            
            self.start_time = datetime.utcnow()
            logger.info(f"Started FastAPI stream processor for camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start FastAPI stream processor for camera {self.camera_id}: {str(e)}")
            # Clean up on error
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            return False
    
    async def stop_async(self):
        """Stop the stream processor asynchronously."""
        try:
            self.is_running = False
            self._stop_event.set()
            
            if self.task and not self.task.done():
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            
            if self.cap:
                self.cap.release()
            
            # Clear in-memory frame storage
            self.latest_frame = None
            self.frame_timestamp = None
            
            logger.info(f"Stopped FastAPI stream processor for camera {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error stopping FastAPI stream processor for camera {self.camera_id}: {str(e)}")
    
    async def test_rtsp_connection_async(self) -> bool:
        """Test RTSP connection asynchronously."""
        try:
            logger.info(f"Testing RTSP connection for camera {self.camera_id}")
            
            # Run the blocking RTSP test in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._test_rtsp_connection_sync)
            
            return result
            
        except Exception as e:
            logger.error(f"RTSP connection test error for camera {self.camera_id}: {str(e)}")
            return False
    
    def _test_rtsp_connection_sync(self) -> bool:
        """Synchronous RTSP connection test."""
        try:
            # Try a quick connection test with OpenCV
            test_cap = cv2.VideoCapture(self.camera.rtsp_url)
            if test_cap.isOpened():
                # Try to read one frame
                ret, frame = test_cap.read()
                test_cap.release()
                
                if ret and frame is not None:
                    logger.info(f"RTSP connection test successful for camera {self.camera_id} (frame shape: {frame.shape})")
                    return True
                else:
                    logger.warning(f"RTSP connection test: camera opened but no frames received for camera {self.camera_id}")
                    return False
            else:
                logger.error(f"RTSP connection test failed: could not open stream for camera {self.camera_id}")
                return False
                
        except Exception as e:
            logger.error(f"RTSP connection test error for camera {self.camera_id}: {str(e)}")
            return False
    
    async def _start_opencv_stream_async(self):
        """Start OpenCV RTSP streaming asynchronously."""
        try:
            logger.info(f"Opening RTSP stream with OpenCV for camera {self.camera_id}")
            
            # Run OpenCV operations in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._start_opencv_stream_sync)
            
        except Exception as e:
            logger.error(f"Failed to start OpenCV stream for camera {self.camera_id}: {str(e)}")
            raise
    
    async def _initialize_face_recognition(self):
        """Initialize face recognition embeddings for this stream."""
        try:
            logger.info(f"Initializing face recognition for camera {self.camera_id}")
            
            # Create database session
            db = SessionLocal()
            face_recognition_service.is_initialized = True
            
            try:
                # Load face embeddings into memory
                logger.info(f"Camera {self.camera_id}: Loading face embeddings...")
                face_recognition_service._load_face_embeddings(db)
                
                if face_recognition_service.face_index is not None:
                    total_embeddings = len(face_recognition_service.face_embeddings) if face_recognition_service.face_embeddings else 0
                    logger.info(f"Camera {self.camera_id}: ✅ Loaded {total_embeddings} face embeddings")
                    
                    # Debug: Check what embeddings are loaded
                    if total_embeddings > 0:
                        logger.info(f"Camera {self.camera_id}: Face embeddings loaded successfully")
                        # Log first few embedding IDs for debugging
                        first_embeddings = list(face_recognition_service.face_embeddings.keys())[:3]
                        logger.info(f"Camera {self.camera_id}: First few embedding IDs: {first_embeddings}")
                    else:
                        logger.warning(f"Camera {self.camera_id}: ❌ No face embeddings loaded - check database")
                else:
                    logger.warning(f"Camera {self.camera_id}: ❌ Face index not created - no embeddings loaded")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to initialize face recognition for camera {self.camera_id}: {str(e)}")
            # Don't fail the stream startup for face recognition issues
    
    def _start_opencv_stream_sync(self):
        """Synchronous OpenCV stream initialization."""
        # Open RTSP stream with OpenCV
        self.cap = cv2.VideoCapture(self.camera.rtsp_url)
        
        # Set OpenCV properties for high FPS streaming
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
        self.cap.set(cv2.CAP_PROP_FPS, self.camera.frame_rate)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # Optimize for H264
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera.resolution_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera.resolution_height)
        
        # Check if stream opened successfully
        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream for camera {self.camera_id}")
            raise Exception("Failed to open RTSP stream")
        
        # Try to read one frame to verify connection
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning(f"RTSP stream opened but no frames received for camera {self.camera_id}")
            # Don't fail here, some cameras take time to start streaming
        else:
            logger.info(f"Successfully read first frame from camera {self.camera_id} (shape: {frame.shape})")
        
        logger.info(f"OpenCV RTSP streaming started successfully for camera {self.camera_id}")
    
    async def _process_stream_async(self):
        """Main async stream processing loop."""
        frame_times = []
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.is_running and not self._stop_event.is_set():
                try:
                    # Check if OpenCV stream is still running
                    if not self.cap or not self.cap.isOpened():
                        logger.error(f"OpenCV stream died for camera {self.camera_id}")
                        break
                    
                    start_frame_time = time.time()
                    
                    # Read frame from OpenCV in thread pool
                    loop = asyncio.get_event_loop()
                    frame_result = await loop.run_in_executor(None, self._read_frame_sync)
                    
                    if frame_result is None:
                        logger.warning(f"No frame received from camera {self.camera_id}")
                        consecutive_errors += 1
                        self.errors_count += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(f"No frames received for camera {self.camera_id}, stopping stream")
                            break
                    else:
                        # Store frame in memory (replaces Redis)
                        self._store_frame_in_memory(frame_result)
                        
                        # Process frame for face detection (non-blocking)
                        await self._process_frame_for_faces_async(frame_result)
                        
                        # Update frame metrics
                        self.frame_count += 1
                        consecutive_errors = 0  # Reset error counter on successful frame
                        
                        # Calculate FPS
                        frame_time = time.time() - start_frame_time
                        frame_times.append(frame_time)
                        
                        # Keep only last 30 frame times for FPS calculation
                        if len(frame_times) > 30:
                            frame_times.pop(0)
                        
                        # Calculate average FPS
                        if frame_times:
                            avg_frame_time = sum(frame_times) / len(frame_times)
                            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    
                    # CRITICAL: Yield control back to event loop to prevent blocking
                    # Optimized sleep for 30+ FPS - balance between responsiveness and performance
                    await asyncio.sleep(0.02)  # 20ms sleep for smooth 30+ FPS (50 FPS max theoretical)
                    
                except Exception as e:
                    logger.error(f"Error in stream processing for camera {self.camera_id}: {str(e)}")
                    consecutive_errors += 1
                    self.errors_count += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors for camera {self.camera_id}, stopping stream")
                        break
                    
                    
        except Exception as e:
            logger.error(f"Critical error in FastAPI stream processing for camera {self.camera_id}: {str(e)}")
            self.errors_count += 1
        finally:
            self.is_running = False
            logger.info(f"FastAPI stream processing stopped for camera {self.camera_id}")
    
    def _read_frame_sync(self):
        """Synchronous frame reading."""
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            return None
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None
    
    def _store_frame_in_memory(self, frame: np.ndarray):
        """Store frame in memory (replaces Redis storage)."""
        try:
            # Store frame in memory
            self.latest_frame = frame.copy()
            self.frame_timestamp = datetime.utcnow()
            
            # Add to frame buffer
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Only log occasionally to reduce overhead
            if self.frame_count % 30 == 0:
                logger.debug(f"Stored frame {self.frame_count} in memory for camera {self.camera_id}")
                
        except Exception as e:
            logger.debug(f"Failed to store frame in memory: {str(e)}")
    
    async def _process_frame_for_faces_async(self, frame: np.ndarray):
        """Process frame for face detection and recognition asynchronously."""
        try:
            # Process every 3rd frame for optimal 30+ FPS performance
            # This balances responsiveness with performance
            current_time = time.time()
            if self.frame_count % 3 != 0:  # Process every 3rd frame
                return
            
            # Additional frame rate limiting to prevent GPU overload
            if hasattr(self, '_last_processing_time'):
                time_since_last = current_time - self._last_processing_time
                if time_since_last < 0.033:  # Max 30 FPS processing (33ms between frames)
                    return
            
            self._last_processing_time = current_time
            
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame received for camera {self.camera_id}")
                return
            
            # Check if frame actually contains faces before processing
            if not self._frame_has_potential_faces(frame):
                return
            
            # Process frame for face detection and recognition
            await self._detect_and_recognize_faces(frame)
            
            self.last_detection_time = current_time
            
            # Update metrics every 30 frames for optimal performance
            if self.frame_count % 30 == 0:
                self._update_metrics()
                self._cleanup_expired_faces()
            
        except Exception as e:
            logger.warning(f"Face detection processing error for camera {self.camera_id}: {str(e)}")
            # Don't crash the stream for face detection errors
    
    async def _detect_and_recognize_faces(self, frame: np.ndarray):
        """Detect and recognize faces in the frame."""
        try:
            # Check if face recognition service is initialized
            if not face_recognition_service.is_initialized:
                logger.warning(f"Face recognition service not initialized for camera {self.camera_id}")
                return
            
            # Extract face embeddings from the frame
            faces = face_recognition_service._extract_face_embeddings(frame)
            
            # Optimized logging for performance - only log every 30 frames
            if faces:
                if self.frame_count % 30 == 0:  # Reduce logging overhead
                    logger.info(f"Camera {self.camera_id}: Raw MTCNN detection - {len(faces)} faces found in frame {self.frame_count}")
                for i, face in enumerate(faces):
                    bbox = face.get("bbox", {})
                    confidence = face.get("confidence", 0)
                    if self.frame_count % 30 == 0:  # Reduce logging overhead
                        logger.debug(f"  Face {i+1}: bbox={bbox}, confidence={confidence:.3f}")
                        # Log confidence scores to help tune threshold
                        if confidence < settings.FACE_DETECTION_CONFIDENCE:
                            logger.debug(f"  Face {i+1}: REJECTED - confidence {confidence:.3f} < threshold {settings.FACE_DETECTION_CONFIDENCE}")
                        else:
                            logger.debug(f"  Face {i+1}: ACCEPTED - confidence {confidence:.3f} >= threshold {settings.FACE_DETECTION_CONFIDENCE}")
            else:
                if self.frame_count % 20 == 0:  # Reduce logging overhead
                    logger.debug(f"Camera {self.camera_id}: No faces detected by MTCNN in frame {self.frame_count}")
                return
            
            # Validate that faces have proper bounding boxes using strict validation
            valid_faces = []
            for face in faces:
                if self._validate_face_data(face, frame.shape):
                    valid_faces.append(face)
            
            if not valid_faces:
                if self.frame_count % 30 == 0:  # Reduce logging overhead
                    logger.warning(f"Camera {self.camera_id}: No valid faces after validation in frame {self.frame_count}")
                return
            
            if self.frame_count % 30 == 0:  # Reduce logging overhead
                logger.info(f"Camera {self.camera_id}: Valid faces after validation: {len(valid_faces)}")
            
            # Apply face deduplication to remove overlapping detections
            deduplicated_faces = self._deduplicate_faces(valid_faces)
            if self.frame_count % 30 == 0:  # Reduce logging overhead
                logger.info(f"Camera {self.camera_id}: After deduplication: {len(deduplicated_faces)} faces")
            
            # Process each deduplicated face SIMULTANEOUSLY
            new_faces_count = 0
            face_tasks = []  # Collect all face processing tasks
            
            for face in deduplicated_faces:
                # Check if this face matches an existing tracked face
                existing_face_id = self._find_existing_face_id(face)
                
                if existing_face_id:
                    # Update existing face tracking
                    self._update_existing_face_tracking(existing_face_id, face)
                    logger.debug(f"Camera {self.camera_id}: Updated existing face {existing_face_id}")
                else:
                    # New face detected
                    new_faces_count += 1
                    face_id = self._add_new_face_to_tracking(face)
                    logger.debug(f"Camera {self.camera_id}: Added new face {face_id}")
                
                # Collect face processing task (don't await yet)
                face_tasks.append(self._process_detected_face(frame, face))
            
            # Process ALL faces simultaneously using asyncio.gather
            if face_tasks:
                if self.frame_count % 30 == 0:  # Reduce logging overhead
                    logger.info(f"Camera {self.camera_id}: Processing {len(face_tasks)} faces simultaneously")
                await asyncio.gather(*face_tasks, return_exceptions=True)
            
            # Update face detection metrics only for new faces
            if new_faces_count > 0:
                self.face_detection_metrics['total_faces_detected'] += new_faces_count
                logger.info(f"Camera {self.camera_id}: Detected {new_faces_count} new faces in frame {self.frame_count}")
            
            # Log detection results every 30 frames for performance
            if self.frame_count % 30 == 0:
                logger.info(f"Camera {self.camera_id}: Frame {self.frame_count} - Raw: {len(faces)}, Valid: {len(valid_faces)}, Deduped: {len(deduplicated_faces)}, New: {new_faces_count}")
            
        except Exception as e:
            logger.error(f"Error in face detection and recognition for camera {self.camera_id}: {str(e)}")
    
    async def _process_detected_face(self, frame: np.ndarray, face: Dict[str, Any]):
        """Process a single detected face for recognition."""
        try:
            # Use the validation method
            if not self._validate_face_data(face, frame.shape):
                logger.warning(f"Camera {self.camera_id}: Face data validation failed")
                return
            
            # Find similar faces using FAISS index
            similar_faces = face_recognition_service._find_similar_faces(
                face["embedding"], 
                settings.FACE_RECOGNITION_THRESHOLD
            )
            
            user_id = None
            confidence_score = 0.0
            user_name = "Unknown"
            
            # Add detailed logging for face recognition debugging
            logger.info(f"Camera {self.camera_id}: Face recognition attempt - Threshold: {settings.FACE_RECOGNITION_THRESHOLD}")
            logger.info(f"Camera {self.camera_id}: Similar faces found: {len(similar_faces) if similar_faces else 0}")
            
            # Debug: Check if embeddings are loaded
            total_embeddings = len(face_recognition_service.face_embeddings) if face_recognition_service.face_embeddings else 0
            logger.info(f"Camera {self.camera_id}: Total embeddings loaded: {total_embeddings}")
            if total_embeddings == 0:
                logger.error(f"Camera {self.camera_id}: ❌ NO FACE EMBEDDINGS LOADED - This is why recognition fails!")
                # Skip recognition if no embeddings are loaded
                logger.info(f"Camera {self.camera_id}: Skipping face recognition - no embeddings available")
                return
            
            if similar_faces:
                # Get best match
                best_match_id, confidence_score = similar_faces[0]
                
                # Validate confidence score - should never be above 1.0 (100%)
                if confidence_score > 1.0:
                    logger.warning(f"Camera {self.camera_id}: ⚠️ Invalid confidence score {confidence_score:.3f} > 1.0, capping at 1.0")
                    confidence_score = 1.0
                elif confidence_score < 0.0:
                    logger.warning(f"Camera {self.camera_id}: ⚠️ Invalid confidence score {confidence_score:.3f} < 0.0, setting to 0.0")
                    confidence_score = 0.0
                
                user_id = face_recognition_service.user_embeddings.get(best_match_id)
                
                # Log face recognition details for debugging
                logger.info(f"Camera {self.camera_id}: Face recognition - {len(similar_faces)} matches found")
                logger.info(f"Camera {self.camera_id}: Best match - ID: {best_match_id}, User: {user_id}, Confidence: {confidence_score:.3f}")
                logger.info(f"Camera {self.camera_id}: Recognition threshold: {settings.FACE_RECOGNITION_THRESHOLD}")
                
                # Check if confidence meets threshold
                if confidence_score >= settings.FACE_RECOGNITION_THRESHOLD:
                    logger.info(f"Camera {self.camera_id}: ✅ Confidence {confidence_score:.3f} >= threshold {settings.FACE_RECOGNITION_THRESHOLD} - RECOGNIZED")
                else:
                    logger.warning(f"Camera {self.camera_id}: ❌ Confidence {confidence_score:.3f} < threshold {settings.FACE_RECOGNITION_THRESHOLD} - NOT RECOGNIZED")
                
                if user_id:
                    # Get user name from database
                    user_name = await self._get_user_name(user_id)
                    
                    # Update recognition metrics
                    self.face_detection_metrics['total_faces_recognized'] += 1
                    self.face_detection_metrics['last_recognition_time'] = datetime.utcnow()
                    
                    # Calculate recognition accuracy - ensure we don't divide by zero
                    total_detected = self.face_detection_metrics['total_faces_detected']
                    total_recognized = self.face_detection_metrics['total_faces_recognized']
                    
                    if total_detected > 0:
                        accuracy = total_recognized / total_detected
                        # Ensure accuracy is between 0 and 1, and not infinite
                        self.face_detection_metrics['recognition_accuracy'] = safe_float(accuracy, 0.0)
                        # Cap accuracy at 100%
                        if self.face_detection_metrics['recognition_accuracy'] > 1.0:
                            self.face_detection_metrics['recognition_accuracy'] = 1.0
                        logger.debug(f"Camera {self.camera_id}: Accuracy updated: {total_recognized}/{total_detected} = {self.face_detection_metrics['recognition_accuracy']:.3f}")
                    else:
                        self.face_detection_metrics['recognition_accuracy'] = 0.0
                    
                    # Store detection in database
                    await self._store_face_detection(frame, face, user_id, confidence_score)
                    
                    # Log recognition result
                    logger.info(f"Camera {self.camera_id}: Recognized user {user_name} ({user_id}) with confidence {confidence_score:.3f}")
                    
                    # Add to detection history
                    self.face_detection_metrics['detection_history'].append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'user_id': str(user_id),
                        'user_name': user_name,
                        'confidence': confidence_score,
                        'bbox': face["bbox"]
                    })
                    
                    # Keep only last 100 detections in history
                    if len(self.face_detection_metrics['detection_history']) > 100:
                        self.face_detection_metrics['detection_history'].pop(0)
                    
                    # Update the active face tracking with user information
                    # Find which active face this detection corresponds to
                    for face_id, face_info in self.face_detection_metrics['active_faces'].items():
                        if (face_info['bbox']['x'] == face["bbox"]["x"] and 
                            face_info['bbox']['y'] == face["bbox"]["y"] and
                            face_info['bbox']['width'] == face["bbox"]["width"] and
                            face_info['bbox']['height'] == face["bbox"]["height"]):
                            face_info['user_id'] = user_id
                            face_info['user_name'] = user_name
                            face_info['confidence'] = confidence_score
                            logger.debug(f"Camera {self.camera_id}: Updated face {face_id} with user {user_name}")
                            break
                else:
                    logger.warning(f"Camera {self.camera_id}: User not found for matched face embedding")
            else:
                # No similar faces found
                logger.warning(f"Camera {self.camera_id}: ❌ No similar faces found - threshold too high or no embeddings loaded")
                
                # Face detected but not recognized
                logger.debug(f"Camera {self.camera_id}: Face detected but not recognized (confidence too low)")
                
                # Store unknown face detection
                await self._store_face_detection(frame, face, None, 0.0)
            
            # Note: Bounding boxes are now drawn in get_frame_with_bounding_boxes() 
            # based on tracked faces to avoid duplicates
                
        except Exception as e:
            logger.error(f"Error processing detected face for camera {self.camera_id}: {str(e)}")
    
    async def _store_face_detection(self, frame: np.ndarray, face: Dict[str, Any], user_id: Optional[uuid.UUID], confidence: float):
        """Store face detection in database."""
        try:
            # Create database session
            db = SessionLocal()
            
            try:
                # Save face image
                face_image_url = f"detections/{self.camera_id}/{uuid.uuid4()}.jpg"
                
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj
                
                # Convert face data to JSON-serializable format
                converted_bbox = convert_numpy_types(face["bbox"])
                converted_landmarks = convert_numpy_types(face.get("landmarks"))
                converted_confidence = convert_numpy_types(face["confidence"])
                
                # Create detection record
                detection = FaceDetection(
                    camera_id=self.camera_id,
                    timestamp=datetime.utcnow().isoformat(),
                    confidence_score=converted_confidence,
                    face_bbox=converted_bbox,
                    landmarks=converted_landmarks,
                    recognized_user_id=user_id,
                    recognition_confidence=confidence if user_id else None,
                    face_image_url=face_image_url
                )
                
                db.add(detection)
                db.commit()
                
                logger.debug(f"Camera {self.camera_id}: Stored face detection for user {user_id}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error storing face detection for camera {self.camera_id}: {str(e)}")
            if 'db' in locals():
                db.rollback()
    
    async def _get_user_name(self, user_id: uuid.UUID) -> str:
        """Get user name from database."""
        try:
            db = SessionLocal()
            try:
                from app.models.database.user import User
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    return user.full_name or f"User {user_id}"
                return f"User {user_id}"
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error getting user name for {user_id}: {str(e)}")
            return f"User {user_id}"
    
    def _draw_face_box_and_name(self, frame: np.ndarray, face: Dict[str, Any], user_name: str, confidence: float):
        """Draw bounding box and name label on the frame."""
        try:
            bbox = face.get("bbox", {})
            if not bbox or 'x' not in bbox or 'y' not in bbox or 'width' not in bbox or 'height' not in bbox:
                return
            
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Validate coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return
            if x + w > frame.shape[1] or y + h > frame.shape[0]:
                return
            
            # Draw red bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Prepare label text
            if user_name == "Unknown":
                label_text = "Unknown"
                label_color = (0, 0, 255)  # Red for unknown
            else:
                label_text = f"{user_name} ({confidence:.2f})"
                label_color = (0, 255, 0)  # Green for recognized
            
            # Get text size for label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Calculate label position (below the bounding box)
            label_x = x
            label_y = y + h + text_height + 10
            
            # Ensure label is within frame bounds
            if label_y + text_height > frame.shape[0]:
                label_y = y - 10  # Place above the box if below is out of bounds
            
            # Draw label background (solid red rectangle)
            cv2.rectangle(frame, 
                         (label_x, label_y - text_height - 5), 
                         (label_x + text_width + 10, label_y + 5), 
                         (0, 0, 255), -1)  # -1 for filled rectangle
            
            # Draw white text on red background
            cv2.putText(frame, label_text, (label_x + 5, label_y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            logger.debug(f"Camera {self.camera_id}: Drew face box for {user_name} at ({x}, {y})")
            
        except Exception as e:
            logger.error(f"Error drawing face box for camera {self.camera_id}: {str(e)}")
    
    def _update_metrics(self):
        """Update internal metrics."""
        try:
            # Update stream info
            self.stream_info = {
                "camera_id": str(self.camera_id),
                "frame_count": self.frame_count,
                "fps": round(self.fps, 2),
                "errors_count": self.errors_count,
                "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
                "uptime_seconds": round((datetime.utcnow() - self.start_time).total_seconds()) if self.start_time else 0,
                "latest_frame_timestamp": self.frame_timestamp.isoformat() if self.frame_timestamp else None,
                "face_detection_metrics": self.face_detection_metrics
            }
            
        except Exception as e:
            logger.debug(f"Failed to update metrics: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stream status."""
        try:
            uptime = 0
            if self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Check if OpenCV stream is active
            is_opencv_active = (
                self.cap and 
                self.cap.isOpened()
            )
            
            # Check if processing task is alive
            is_task_alive = (
                self.task and 
                not self.task.done()
            )
            
            # Get performance metrics for high FPS monitoring
            performance_info = {
                'avg_frame_time_ms': round(safe_float(self.performance_metrics['avg_frame_time']) * 1000, 2),
                'max_frame_time_ms': round(safe_float(self.performance_metrics['max_frame_time']) * 1000, 2),
                'min_frame_time_ms': round(safe_float(self.performance_metrics['min_frame_time']) * 1000, 2),
                'frame_time_history_count': len(self.performance_metrics['frame_time_history'])
            }
            
            return {
                "is_running": self.is_running,
                "camera_id": str(self.camera_id),
                "frame_count": self.frame_count,
                "fps": round(self.fps, 2),
                "errors_count": self.errors_count,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
                "uptime_seconds": round(uptime, 2),
                "opencv_active": is_opencv_active,
                "task_alive": is_task_alive,
                "camera_connected": is_opencv_active,
                "stream_health": "healthy" if self.frame_count > 0 and self.errors_count < 10 else "degraded",
                "performance_metrics": performance_info,
                "processor_type": "FastAPI",
                "latest_frame_available": self.latest_frame is not None,
                "latest_frame_timestamp": self.frame_timestamp.isoformat() if self.frame_timestamp else None,
                "frame_buffer_size": len(self.frame_buffer),
                "face_detection_metrics": self.face_detection_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting stream status for camera {self.camera_id}: {str(e)}")
            return {
                "is_running": False,
                "camera_id": str(self.camera_id),
                "error": str(e)
            }
    
    def get_latest_face_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest face detection results."""
        try:
            # Return recent detections from history
            recent_detections = self.face_detection_metrics['detection_history'][-limit:]
            
            # Calculate current active faces count
            current_time = time.time()
            active_faces_count = len([
                face_id for face_id, face_info in self.face_detection_metrics['active_faces'].items()
                if current_time - face_info['last_seen'] <= self.face_detection_metrics['face_tracking_timeout']
            ])
            
            # Add current metrics
            result = {
                "recent_detections": recent_detections,
                "total_faces_detected": self.face_detection_metrics['total_faces_detected'],
                "total_faces_recognized": self.face_detection_metrics['total_faces_recognized'],
                "recognition_accuracy": round(self.face_detection_metrics['recognition_accuracy'], 3),
                "active_faces_count": active_faces_count,
                "last_recognition_time": self.face_detection_metrics['last_recognition_time'].isoformat() if self.face_detection_metrics['last_recognition_time'] else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting latest face detections for camera {self.camera_id}: {str(e)}")
            return {
                "recent_detections": [],
                "total_faces_detected": 0,
                "total_faces_recognized": 0,
                "recognition_accuracy": 0.0,
                "active_faces_count": 0,
                "last_recognition_time": None
            }
    
    def get_frame_with_bounding_boxes(self) -> np.ndarray:
        """Get the latest frame with bounding boxes drawn around detected faces."""
        try:
            if self.latest_frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create a copy of the frame to draw on
            frame_with_boxes = self.latest_frame.copy()
            
            # Draw bounding boxes for currently tracked faces (no duplicates)
            current_time = time.time()
            active_faces = [
                (face_id, face_info) for face_id, face_info in self.face_detection_metrics['active_faces'].items()
                if current_time - face_info['last_seen'] <= self.face_detection_metrics['face_tracking_timeout']
            ]
            
            for face_id, face_info in active_faces:
                bbox = face_info.get('bbox', {})
                
                if bbox and 'x' in bbox and 'y' in bbox and 'width' in bbox and 'height' in bbox:
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    
                    # Get user information directly from the tracked face
                    user_name = face_info.get('user_name', "Unknown")
                    confidence = face_info.get('confidence', 0.0)
                    
                    # Draw bounding box around the face
                    color = (0, 255, 0) if user_name != "Unknown" else (0, 0, 255)  # Green for recognized, Red for unknown
                    thickness = 3
                    cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), color, thickness)
                    
                    # Draw name label below the bounding box
                    if user_name != "Unknown":
                        label_text = f"{user_name} ({confidence:.2f})"
                        label_color = (0, 255, 0)  # Green for recognized
                    else:
                        label_text = "Unknown"
                        label_color = (0, 0, 255)  # Red for unknown
                    
                    # Get text size for label background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    
                    # Calculate label position (below the bounding box)
                    label_x = x
                    label_y = y + h + text_height + 10
                    
                    # Ensure label is within frame bounds
                    if label_y + text_height > frame_with_boxes.shape[0]:
                        label_y = y - 10  # Place above the box if below is out of bounds
                    
                    # Draw label background (solid colored rectangle)
                    cv2.rectangle(frame_with_boxes, 
                                 (label_x, label_y - text_height - 5), 
                                 (label_x + text_width + 10, label_y + 5), 
                                 label_color, -1)  # -1 for filled rectangle
                    
                    # Draw white text on colored background
                    cv2.putText(frame_with_boxes, label_text, (label_x + 5, label_y - 5), 
                               font, font_scale, (255, 255, 255), thickness)
            
            # Add information text in the left corner (no labels above faces)
            # Create a semi-transparent background for text
            overlay = frame_with_boxes.copy()
            cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame_with_boxes, 0.3, 0, frame_with_boxes)
            
            # Add text information in the left corner
            y_offset = 35
            line_height = 25
            
            # Camera name
            cv2.putText(frame_with_boxes, f"Camera: {self.camera.name}", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # FPS information
            cv2.putText(frame_with_boxes, f"FPS: {self.fps:.1f}", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # Total faces detected (unique)
            cv2.putText(frame_with_boxes, f"Total Detected: {self.face_detection_metrics['total_faces_detected']}", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # Active faces currently visible
            current_time = time.time()
            active_faces_count = len([
                face_id for face_id, face_info in self.face_detection_metrics['active_faces'].items()
                if current_time - face_info['last_seen'] <= self.face_detection_metrics['face_tracking_timeout']
            ])
            cv2.putText(frame_with_boxes, f"Active Faces: {active_faces_count}", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # Recognition accuracy (capped at 100%)
            accuracy = min(self.face_detection_metrics.get('recognition_accuracy', 0.0), 1.0)
            cv2.putText(frame_with_boxes, f"Accuracy: {accuracy:.1%}", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame_with_boxes
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes for camera {self.camera_id}: {str(e)}")
            return self.latest_frame if self.latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _frame_has_potential_faces(self, frame: np.ndarray) -> bool:
        """Check if frame has potential faces using basic image analysis."""
        try:
            if frame is None or frame.size == 0:
                return False
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic checks for potential faces:
            # 1. Check if image has reasonable brightness (not too dark or too bright)
            mean_brightness = np.mean(gray)
            if mean_brightness < 30 or mean_brightness > 225:  # Too dark or too bright
                return False
            
            # 2. Check if image has reasonable contrast
            contrast = np.std(gray)
            if contrast < 20:  # Too low contrast
                return False
            
            # 3. Check if image has reasonable size (not too small)
            height, width = frame.shape[:2]
            if height < 100 or width < 100:  # Too small
                return False
            
            # 4. Check if image has some texture (not just flat color)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:  # Too flat
                return False
            
            # Frame passes basic checks, might contain faces
            return True
            
        except Exception as e:
            logger.debug(f"Error checking frame for potential faces: {str(e)}")
            return True  # Default to True if check fails
    
    def _validate_face_data(self, face: Dict[str, Any], frame_shape: tuple) -> bool:
        """Validate face data before processing with stricter criteria."""
        try:
            # Check if face data exists
            if not face or "embedding" not in face or "bbox" not in face:
                return False
            
            # Validate bounding box
            bbox = face.get("bbox", {})
            if not (bbox and 'x' in bbox and 'y' in bbox and 'width' in bbox and 'height' in bbox):
                return False
            
            # Check if face is within frame boundaries
            frame_height, frame_width = frame_shape[:2]
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            if (x < 0 or y < 0 or x + w > frame_width or y + h > frame_height or
                w <= 0 or h <= 0):
                return False
            
            # Stricter confidence check - must be above threshold
            confidence = face.get("confidence", 0)
            if confidence < settings.FACE_DETECTION_CONFIDENCE:
                return False
            
            # Check face size - must be reasonable (not too small, not too large)
            face_area = w * h
            frame_area = frame_width * frame_height
            face_area_ratio = face_area / frame_area
            
            # Face should be between 0.1% and 80% of frame area (more permissive)
            if face_area_ratio < 0.001 or face_area_ratio > 0.8:
                return False
            
            # Check aspect ratio - face should be roughly human-like (more permissive)
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                return False
            
            # Check minimum face size - must be at least 20x20 pixels (more permissive)
            if w < 20 or h < 20:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating face data: {str(e)}")
            return False
    
    def _is_duplicate_face(self, face: Dict[str, Any]) -> bool:
        """Check if this face is a duplicate of an already detected face."""
        try:
            if not face or "bbox" not in face:
                return False
            
            bbox = face.get("bbox", {})
            if not (bbox and 'x' in bbox and 'y' in bbox and 'width' in bbox and 'height' in bbox):
                return False
            
            current_time = time.time()
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            face_center = (x + w//2, y + h//2)
            
            # Check against active faces
            for face_id, face_info in list(self.face_detection_metrics['active_faces'].items()):
                # Remove expired faces
                if current_time - face_info['last_seen'] > self.face_detection_metrics['face_tracking_timeout']:
                    del self.face_detection_metrics['active_faces'][face_id]
                    continue
                
                # Check if centers are close (within 50 pixels)
                existing_center = face_info['center']
                distance = ((face_center[0] - existing_center[0])**2 + (face_center[1] - existing_center[1])**2)**0.5
                
                if distance < 50:  # Same face if centers are within 50 pixels
                    # Update last seen time
                    self.face_detection_metrics['active_faces'][face_id]['last_seen'] = current_time
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking for duplicate face: {str(e)}")
            return False
    
    def _deduplicate_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping face detections to keep only the best one per face."""
        try:
            if len(faces) <= 1:
                return faces
            
            # Sort faces by confidence score (highest first)
            sorted_faces = sorted(faces, key=lambda x: x.get("confidence", 0), reverse=True)
            
            deduplicated = []
            processed_areas = []
            
            for face in sorted_faces:
                bbox = face.get("bbox", {})
                if not bbox:
                    continue
                
                # Calculate face area and center
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                face_area = w * h
                face_center = (x + w // 2, y + h // 2)
                
                # Check if this face overlaps significantly with already processed faces
                is_duplicate = False
                for processed_face in processed_areas:
                    px, py, pw, ph = processed_face['bbox']['x'], processed_face['bbox']['y'], processed_face['bbox']['width'], processed_face['bbox']['height']
                    processed_center = (px + pw // 2, py + ph // 2)
                    
                    # Calculate distance between centers
                    center_distance = ((face_center[0] - processed_center[0]) ** 2 + (face_center[1] - processed_center[1]) ** 2) ** 0.5
                    
                    # Calculate overlap threshold - be more strict about duplicates
                    overlap_threshold = min(w, h) * 0.2  # Reduced from 30% to 20% for stricter deduplication
                    
                    if center_distance < overlap_threshold:
                        # This is likely a duplicate detection
                        is_duplicate = True
                        logger.debug(f"Camera {self.camera_id}: Removed duplicate face detection (center distance: {center_distance:.1f}, threshold: {overlap_threshold:.1f})")
                        break
                
                if not is_duplicate:
                    deduplicated.append(face)
                    processed_areas.append({
                        'bbox': bbox,
                        'area': face_area,
                        'center': face_center
                    })
            
            logger.debug(f"Camera {self.camera_id}: Deduplicated {len(faces)} faces to {len(deduplicated)} unique faces")
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error deduplicating faces: {str(e)}")
            return faces
    
    def _add_face_to_tracking(self, face: Dict[str, Any]) -> str:
        """Add a new face to tracking system."""
        try:
            bbox = face.get("bbox", {})
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            face_center = (x + w//2, y + h//2)
            
            # Check if this face is close to an existing tracked face
            current_time = time.time()
            for face_id, face_info in list(self.face_detection_metrics['active_faces'].items()):
                # Remove expired faces
                if current_time - face_info['last_seen'] > self.face_detection_metrics['face_tracking_timeout']:
                    del self.face_detection_metrics['active_faces'][face_id]
                    continue
                
                # Check if centers are close (within 100 pixels for same person)
                existing_center = face_info['center']
                distance = ((face_center[0] - existing_center[0])**2 + (face_center[1] - existing_center[1])**2)**0.5
                
                if distance < 100:  # Same person if centers are within 100 pixels
                    # Update existing face tracking
                    face_info['last_seen'] = current_time
                    face_info['center'] = face_center  # Update position
                    face_info['bbox'] = bbox  # Update bounding box
                    face_info['detection_count'] += 1
                    logger.debug(f"Camera {self.camera_id}: Updated existing face tracking (ID: {face_id})")
                    return face_id
            
            # Generate new face ID for truly new face
            face_id = str(uuid.uuid4())
            
            self.face_detection_metrics['active_faces'][face_id] = {
                'center': face_center,
                'bbox': bbox,
                'first_seen': current_time,
                'last_seen': current_time,
                'detection_count': 1,
                'user_id': None,
                'user_name': "Unknown",
                'confidence': 0.0
            }
            
            logger.debug(f"Camera {self.camera_id}: Added new face to tracking (ID: {face_id})")
            return face_id
            
        except Exception as e:
            logger.debug(f"Error adding face to tracking: {str(e)}")
            return str(uuid.uuid4())
    
    def _find_existing_face_id(self, face: Dict[str, Any]) -> Optional[str]:
        """Find if a detected face matches an existing tracked face."""
        try:
            bbox = face.get("bbox", {})
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            face_center = (x + w//2, y + h//2)
            
            current_time = time.time()
            for face_id, face_info in list(self.face_detection_metrics['active_faces'].items()):
                # Remove expired faces
                if current_time - face_info['last_seen'] > self.face_detection_metrics['face_tracking_timeout']:
                    del self.face_detection_metrics['active_faces'][face_id]
                    continue
                
                # Check if centers are close (within 50 pixels for same person)
                # Reduced from 100 to 50 pixels to be more strict about face matching
                existing_center = face_info['center']
                distance = ((face_center[0] - existing_center[0])**2 + (face_center[1] - existing_center[1])**2)**0.5
                
                if distance < 50:  # Same person if centers are within 50 pixels (more strict)
                    return face_id
            
            return None
            
        except Exception as e:
            logger.debug(f"Error finding existing face ID: {str(e)}")
            return None
    
    def _update_existing_face_tracking(self, face_id: str, face: Dict[str, Any]):
        """Update existing face tracking with new position and data."""
        try:
            bbox = face.get("bbox", {})
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            face_center = (x + w//2, y + h//2)
            current_time = time.time()
            
            if face_id in self.face_detection_metrics['active_faces']:
                face_info = self.face_detection_metrics['active_faces'][face_id]
                face_info['last_seen'] = current_time
                face_info['center'] = face_center
                face_info['bbox'] = bbox
                face_info['detection_count'] += 1
                
        except Exception as e:
            logger.debug(f"Error updating existing face tracking: {str(e)}")
    
    def _add_new_face_to_tracking(self, face: Dict[str, Any]) -> str:
        """Add a completely new face to tracking system."""
        try:
            bbox = face.get("bbox", {})
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            face_center = (x + w//2, y + h//2)
            current_time = time.time()
            
            # Generate new face ID
            face_id = str(uuid.uuid4())
            
            self.face_detection_metrics['active_faces'][face_id] = {
                'center': face_center,
                'bbox': bbox,
                'first_seen': current_time,
                'last_seen': current_time,
                'detection_count': 1,
                'user_id': None,
                'user_name': "Unknown",
                'confidence': 0.0
            }
            
            return face_id
            
        except Exception as e:
            logger.debug(f"Error adding new face to tracking: {str(e)}")
            return str(uuid.uuid4())
    
    def _cleanup_expired_faces(self):
        """Clean up expired faces from tracking."""
        try:
            current_time = time.time()
            expired_faces = []
            
            for face_id, face_info in self.face_detection_metrics['active_faces'].items():
                if current_time - face_info['last_seen'] > self.face_detection_metrics['face_tracking_timeout']:
                    expired_faces.append(face_id)
            
            # Remove expired faces
            for face_id in expired_faces:
                del self.face_detection_metrics['active_faces'][face_id]
            
            if expired_faces:
                logger.debug(f"Camera {self.camera_id}: Cleaned up {len(expired_faces)} expired faces")
                
        except Exception as e:
            logger.debug(f"Error cleaning up expired faces: {str(e)}")


# Global stream manager instance for FastAPI
class StreamManager:
    """Manages multiple camera streams using Redis for cross-process/thread communication."""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StreamManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Initialize Redis service for stream metadata
            self.redis_service = redis_service
            # Keep StreamProcessor instances in memory for frame storage
            self.active_streams = {}  # camera_id -> StreamProcessor instance
            self._lock = asyncio.Lock()
            self._initialized = True
    
    async def start_stream(self, camera_id: uuid.UUID, camera: Camera) -> bool:
        """Start a new camera stream asynchronously."""
        async with self._lock:
            try:
                # Check if stream is already running using Redis
                camera_id_str = str(camera_id)
                try:
                    is_active = await self.redis_service.is_stream_active(camera_id_str)
                    if is_active:
                        logger.warning(f"Stream already running for camera {camera_id}")
                        return True
                except Exception as redis_error:
                    logger.warning(f"Redis check failed, proceeding with stream start: {str(redis_error)}")
                
                # Log camera details for debugging
                logger.info(f"Starting FastAPI stream for camera {camera_id}: {camera.name}")
                logger.info(f"Camera IP: {camera.ip_address}:{camera.port}")
                logger.info(f"Camera path: {camera.path}")
                logger.info(f"Camera credentials: username={camera.username}, password={'*' * len(camera.password) if camera.password else 'None'}")
                
                # Create new stream processor
                stream_processor = StreamProcessor(camera_id, camera)
                if await stream_processor.start_async():
                    # Store StreamProcessor instance in memory for frame access
                    self.active_streams[camera_id] = stream_processor
                    
                    # Store stream metadata in Redis
                    try:
                        stream_data = {
                            "camera_id": str(camera_id),
                            "camera_name": camera.name,
                            "started_at": datetime.utcnow().isoformat(),
                            "is_running": True,
                            "status": "active"
                        }
                        await self.redis_service.add_active_stream(camera_id_str, stream_data)
                        logger.info(f"Started FastAPI stream for camera {camera_id} - processor in memory, metadata in Redis")
                    except Exception as redis_error:
                        logger.warning(f"Failed to store stream data in Redis, but stream is running: {str(redis_error)}")
                        logger.info(f"Started FastAPI stream for camera {camera_id} (Redis storage failed)")
                    
                    return True
                else:
                    logger.error(f"Failed to start FastAPI stream for camera {camera_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error starting FastAPI stream for camera {camera_id}: {str(e)}")
                return False
    
    async def stop_stream(self, camera_id: uuid.UUID) -> bool:
        """Stop a camera stream asynchronously."""
        async with self._lock:
            try:
                camera_id_str = str(camera_id)
                
                # Check if stream is active in Redis
                try:
                    is_active = await self.redis_service.is_stream_active(camera_id_str)
                    if not is_active:
                        logger.warning(f"Stream not found for camera {camera_id}")
                        return False
                    
                    # Stop the StreamProcessor instance
                    if camera_id in self.active_streams:
                        processor = self.active_streams[camera_id]
                        await processor.stop_async()
                        del self.active_streams[camera_id]
                        logger.info(f"Stopped StreamProcessor for camera {camera_id}")
                    
                    # Remove from Redis active streams
                    await self.redis_service.remove_active_stream(camera_id_str)
                    logger.info(f"Stopped FastAPI stream for camera {camera_id} - processor stopped, metadata removed from Redis")
                except Exception as redis_error:
                    logger.warning(f"Redis operations failed, but stream stopping completed: {str(redis_error)}")
                    logger.info(f"Stopped FastAPI stream for camera {camera_id} (Redis cleanup failed)")
                
                return True
                    
            except Exception as e:
                logger.error(f"Error stopping FastAPI stream for camera {camera_id}: {str(e)}")
                return False
    
    async def get_stream_status(self, camera_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get streaming status for a camera."""
        try:
            camera_id_str = str(camera_id)
            
            # Check Redis for stream status
            status = await self.redis_service.get_stream_status(camera_id_str)
            if status:
                return status
            
            logger.debug(f"No stream status found in Redis for camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting stream status for camera {camera_id}: {str(e)}")
            return None
    
    async def get_processor(self, camera_id: uuid.UUID) -> Optional[StreamProcessor]:
        """Get StreamProcessor instance for a camera (for frame access)."""
        try:
            # Return the actual StreamProcessor instance from memory
            return self.active_streams.get(camera_id)
        except Exception as e:
            logger.error(f"Error getting processor instance for camera {camera_id}: {str(e)}")
            return None
    
    async def get_face_detection_results(self, camera_id: uuid.UUID, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Get face detection results for a specific camera."""
        try:
            processor = self.active_streams.get(camera_id)
            if processor:
                return processor.get_latest_face_detections(limit)
            return None
        except Exception as e:
            logger.error(f"Error getting face detection results for camera {camera_id}: {str(e)}")
            return None
    
    async def is_stream_active(self, camera_id: uuid.UUID) -> bool:
        """Check if a stream is actually active for a camera."""
        try:
            camera_id_str = str(camera_id)
            return await self.redis_service.is_stream_active(camera_id_str)
        except Exception as e:
            logger.error(f"Error checking stream active status for camera {camera_id}: {str(e)}")
            return False
    
    async def force_refresh_status(self, camera_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Force refresh the stream status for a camera."""
        try:
            logger.info(f"Force refreshing status for camera {camera_id}")
            return self.get_stream_status(camera_id)
        except Exception as e:
            logger.error(f"Error force refreshing status for camera {camera_id}: {str(e)}")
            return None
    
    async def get_all_streams_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active streams."""
        try:
            return await self.redis_service.get_all_streams_status()
        except Exception as e:
            logger.error(f"Error getting all stream statuses: {str(e)}")
            return {}
    
    async def stop_all_streams(self):
        """Stop all active streams."""
        async with self._lock:
            try:
                # Stop all StreamProcessor instances in memory
                for camera_id, processor in list(self.active_streams.items()):
                    try:
                        await processor.stop_async()
                        logger.info(f"Stopped StreamProcessor for camera {camera_id}")
                    except Exception as e:
                        logger.error(f"Error stopping StreamProcessor for camera {camera_id}: {str(e)}")
                
                # Clear the active streams dictionary
                self.active_streams.clear()
                
                # Get all active streams from Redis and remove them
                active_streams = await self.redis_service.get_active_streams()
                for camera_id in active_streams:
                    await self.redis_service.remove_active_stream(camera_id)
                logger.info(f"Stopped all {len(active_streams)} active streams - processors stopped, Redis cleaned")
            except Exception as e:
                logger.error(f"Error stopping all streams: {str(e)}")
    
    async def get_all_face_detection_results(self, limit_per_camera: int = 10) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Get face detection results for all active cameras."""
        try:
            all_results = {}
            
            for camera_id, processor in self.active_streams.items():
                all_results[camera_id] = processor.get_latest_face_detections(limit_per_camera)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error getting all face detection results: {str(e)}")
            return {}
    
    async def refresh_face_recognition_embeddings(self):
        """Refresh face recognition embeddings for all active streams."""
        try:
            logger.info("Refreshing face recognition embeddings for all active streams")
            
            # Create database session
            db = SessionLocal()
            
            try:
                # Reload face embeddings
                face_recognition_service._load_face_embeddings(db)
                
                # Update all active processors
                for camera_id, processor in self.active_streams.items():
                    logger.info(f"Refreshed face recognition embeddings for camera {camera_id}")
                
                logger.info(f"Successfully refreshed face recognition embeddings for {len(self.active_streams)} active streams")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error refreshing face recognition embeddings: {str(e)}")
    
    async def get_face_detection_statistics(self) -> Dict[str, Any]:
        """Get overall face detection statistics across all cameras."""
        try:
            total_faces_detected = 0
            total_faces_recognized = 0
            total_cameras_with_detections = 0
            overall_recognition_accuracy = 0.0
            total_invalid_detections = 0
            
            for processor in self.active_streams.values():
                metrics = processor.face_detection_metrics
                total_faces_detected += metrics['total_faces_detected']
                total_faces_recognized += metrics['total_faces_recognized']
                
                if metrics['total_faces_detected'] > 0:
                    total_cameras_with_detections += 1
                    overall_recognition_accuracy += metrics['recognition_accuracy']
            
            # Calculate average recognition accuracy
            if total_cameras_with_detections > 0:
                overall_recognition_accuracy /= total_cameras_with_detections
                # Ensure accuracy is safe for JSON
                overall_recognition_accuracy = safe_float(overall_recognition_accuracy, 0.0)
                # Cap accuracy at 100%
                if overall_recognition_accuracy > 1.0:
                    overall_recognition_accuracy = 1.0
            
            return {
                "total_cameras_active": len(self.active_streams),
                "total_cameras_with_detections": total_cameras_with_detections,
                "total_faces_detected": total_faces_detected,
                "total_faces_recognized": total_faces_recognized,
                "overall_recognition_accuracy": round(overall_recognition_accuracy, 3),
                "total_invalid_detections": total_invalid_detections,
                "detection_quality": "high" if total_faces_detected > 0 and safe_float(total_faces_recognized / max(total_faces_detected, 1), 0.0) > 0.5 else "low",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting face detection statistics: {str(e)}")
            return {
                "total_cameras_active": 0,
                "total_cameras_with_detections": 0,
                "total_faces_detected": 0,
                "total_faces_recognized": 0,
                "overall_recognition_accuracy": 0.0,
                "total_invalid_detections": 0,
                "detection_quality": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }


# Global FastAPI stream manager instance
stream_manager = StreamManager()
