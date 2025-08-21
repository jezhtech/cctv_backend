"""Lightweight face recognition service using face_recognition library."""
import cv2
import face_recognition
import numpy as np
import base64
import time
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_
from loguru import logger
import uuid
from datetime import datetime, timedelta
import os
from collections import defaultdict

from app.models.database.user import User, FaceEmbedding, FaceDetection, Attendance
from app.schemas.user import FaceRecognitionRequest, FaceUploadRequest
from app.core.config import settings


class FaceRecognitionService:
    """Lightweight service for face detection and recognition using face_recognition library."""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_user_ids = []
        self.known_face_names = []
        self.cooldown_track = defaultdict(float)
        self.COOLDOWN_SECONDS = 120  # 2 minutes cooldown between attendance marks
        self.is_initialized = False
        self.face_recognition_tolerance = 0.5  # Lower = stricter matching
        
        # Initialize the service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the face recognition service."""
        try:
            logger.info("Initializing lightweight face recognition service...")
            
            # Set tolerance based on settings
            if hasattr(settings, 'FACE_RECOGNITION_THRESHOLD'):
                # Map our threshold (0-1, higher = stricter) to face_recognition tolerance (lower = stricter)
                # Our threshold 0.6 (60%) should map to tolerance 0.6 (more lenient)
                # Our threshold 0.8 (80%) should map to tolerance 0.4 (more strict)
                self.face_recognition_tolerance = max(0.3, min(0.7, 1.0 - settings.FACE_RECOGNITION_THRESHOLD))
            
            logger.info(f"Face recognition tolerance set to: {self.face_recognition_tolerance}")
            self.is_initialized = True
            logger.info("Lightweight face recognition service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face recognition service: {str(e)}")
            self.is_initialized = False
    
    def _load_face_embeddings(self, db: Session):
        """Load face embeddings from database into memory."""
        try:
            if not self.is_initialized:
                return
            
            logger.info("Loading face embeddings from database...")
            
            # Clear existing data
            self.known_face_encodings.clear()
            self.known_face_user_ids.clear()
            self.known_face_names.clear()
            
            # Get all face embeddings
            embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.confidence_score >= 0.5  # Basic quality filter
            ).all()
            
            if not embeddings:
                logger.info("No face embeddings found in database")
                return
            
            # Load embeddings into memory
            for embedding in embeddings:
                try:
                    # Decode base64 embedding
                    embedding_bytes = base64.b64decode(embedding.embedding)
                    face_encoding = np.frombuffer(embedding_bytes, dtype=np.float64)
                    
                    # Get user information
                    user = db.query(User).filter(User.id == embedding.user_id).first()
                    if user:
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_user_ids.append(embedding.user_id)
                        self.known_face_names.append(user.full_name or f"User {embedding.user_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load embedding {embedding.id}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(self.known_face_encodings)} face embeddings into memory")
            
        except Exception as e:
            logger.error(f"Failed to load face embeddings: {str(e)}")
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode OpenCV image to base64 string."""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            return ""
    
    def _decode_base64_to_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 string to OpenCV image."""
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {str(e)}")
            return None
    
    def _extract_face_embeddings(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract face embeddings from image using face_recognition library."""
        try:
            if not self.is_initialized:
                return []
            
            # Convert BGR to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            results = []
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                try:
                    # Convert face_recognition format to our format
                    top, right, bottom, left = face_location
                    
                    bbox = {
                        "x": int(left),
                        "y": int(top),
                        "width": int(right - left),
                        "height": int(bottom - top)
                    }
                    
                    # Calculate quality score
                    quality_score = self._calculate_face_quality(image, bbox)
                    
                    results.append({
                        "embedding": face_encoding,
                        "quality_score": quality_score,
                        "bbox": bbox,
                        "landmarks": [],  # face_recognition doesn't provide landmarks
                        "confidence": 0.9,  # High confidence for face_recognition
                        "size": {"width": bbox["width"], "height": bbox["height"]},
                        "angle": 0.0,  # Not calculated
                        "lighting_score": self._calculate_lighting_score(image, bbox),
                        "blur_score": self._calculate_blur_score(image, bbox)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process face {i}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract face embeddings: {str(e)}")
            return []
    
    def _calculate_face_quality(self, image: np.ndarray, bbox: Dict[str, int]) -> float:
        """Calculate face quality score."""
        try:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return 0.0
            
            # Calculate quality metrics
            # 1. Size quality (larger faces are better)
            size_quality = min(1.0, (w * h) / (100 * 100))
            
            # 2. Brightness quality
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_quality = 1.0 - abs(brightness - 128) / 128
            
            # 3. Sharpness quality (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_quality = min(1.0, laplacian_var / 500)
            
            # Combine quality scores
            overall_quality = (size_quality + brightness_quality + sharpness_quality) / 3
            
            return max(0.0, min(1.0, overall_quality))
            
        except Exception as e:
            logger.error(f"Failed to calculate face quality: {str(e)}")
            return 0.0
    
    def _calculate_lighting_score(self, image: np.ndarray, bbox: Dict[str, int]) -> float:
        """Calculate lighting quality score for face region."""
        try:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate lighting metrics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Ideal lighting: mean around 128, good contrast (std > 20)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            contrast_score = min(1.0, std_brightness / 50)
            
            # Combine scores
            lighting_score = (brightness_score + contrast_score) / 2
            
            return max(0.0, min(1.0, lighting_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate lighting score: {str(e)}")
            return 0.0
    
    def _calculate_blur_score(self, image: np.ndarray, bbox: Dict[str, int]) -> float:
        """Calculate blur/quality score for face region."""
        try:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize blur score (higher variance = less blur)
            blur_score = min(1.0, laplacian_var / 500)
            
            return max(0.0, min(1.0, blur_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate blur score: {str(e)}")
            return 0.0
    
    def _convert_numpy_types(self, value):
        """Convert numpy types to native Python types for database storage."""
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._convert_numpy_types(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_numpy_types(item) for item in value]
        else:
            return value
    
    def _find_similar_faces(self, query_encoding: np.ndarray, threshold: float = None) -> List[Tuple[int, float]]:
        """Find similar faces using face_recognition library."""
        try:
            if not self.known_face_encodings:
                return []
            
            threshold = threshold or settings.FACE_RECOGNITION_THRESHOLD
            
            # Use face_recognition's compare_faces with tolerance
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                query_encoding, 
                tolerance=self.face_recognition_tolerance
            )
            
            results = []
            for i, is_match in enumerate(matches):
                if is_match:
                    # Calculate distance for confidence score
                    distance = face_recognition.face_distance([self.known_face_encodings[i]], query_encoding)[0]
                    
                    # Convert distance to similarity score (0-1, higher is better)
                    # face_recognition distance is typically 0.0-1.0, where 0.0 is perfect match
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    if similarity_score >= threshold:
                        results.append((i, float(similarity_score)))
            
            # Sort by similarity (best first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar faces: {str(e)}")
            return []
    
    async def recognize_face(self, db: Session, request: FaceRecognitionRequest) -> Dict[str, Any]:
        """Recognize face from image data."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                return {
                    "success": False,
                    "message": "Face recognition service not initialized",
                    "processing_time_ms": 0
                }
            
            # Decode image
            image = self._decode_base64_to_image(request.image_data)
            if image is None:
                return {
                    "success": False,
                    "message": "Failed to decode image data",
                    "processing_time_ms": 0
                }
            
            # Extract face embeddings
            faces = self._extract_face_embeddings(image)
            if not faces:
                return {
                    "success": False,
                    "message": "No faces detected in image",
                    "processing_time_ms": 0
                }
            
            # Use the first detected face
            face = faces[0]
            
            # Find similar faces
            similar_faces = self._find_similar_faces(
                face["embedding"], 
                request.confidence_threshold
            )
            
            if not similar_faces:
                return {
                    "success": False,
                    "message": "No matching faces found",
                    "processing_time_ms": 0
                }
            
            # Get best match
            best_match_index, confidence_score = similar_faces[0]
            user_id = self.known_face_user_ids[best_match_index]
            user_name = self.known_face_names[best_match_index]
            
            if not user_id:
                return {
                    "success": False,
                    "message": "User not found for matched face",
                    "processing_time_ms": 0
                }
            
            # Get user information
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {
                    "success": False,
                    "message": "User not found",
                    "processing_time_ms": 0
                }
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create face detection record if camera_id is provided
            if request.camera_id:
                await self._create_face_detection(
                    db, request.camera_id, face, user_id, confidence_score
                )
            
            return {
                "success": True,
                "recognized_user": user,
                "confidence_score": confidence_score,
                "face_bbox": face["bbox"],
                "landmarks": face["landmarks"],
                "message": f"Face recognized as {user.full_name}",
                "processing_time_ms": round(processing_time, 2)
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Face recognition failed: {str(e)}")
            return {
                "success": False,
                "message": f"Face recognition failed: {str(e)}",
                "processing_time_ms": round(processing_time, 2)
            }
    
    async def upload_face_image(self, db: Session, request: FaceUploadRequest) -> Dict[str, Any]:
        """Upload and process face image for user."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                return {
                    "success": False,
                    "message": "Face recognition service not initialized",
                    "processing_time_ms": 0
                }
            
            # Check if user exists
            user = db.query(User).filter(User.id == request.user_id).first()
            if not user:
                return {
                    "success": False,
                    "message": "User not found",
                    "processing_time_ms": 0
                }
            
            # Decode image
            image = self._decode_base64_to_image(request.image_data)
            if image is None:
                return {
                    "success": False,
                    "message": "Failed to decode image data",
                    "processing_time_ms": 0
                }
            
            # Extract face embeddings
            faces = self._extract_face_embeddings(image)
            if not faces:
                return {
                    "success": False,
                    "message": "No faces detected in image",
                    "processing_time_ms": 0
                }
            
            # Process all detected faces
            created_embeddings = []
            primary_face_processed = False
            
            for i, face in enumerate(faces):
                try:
                    # Encode embedding to base64
                    embedding_bytes = face["embedding"].tobytes()
                    embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                    
                    # Use the actual uploaded image URL from our storage
                    if request.source_description and "Profile image:" in request.source_description:
                        filename = request.source_description.split("Profile image: ")[-1]
                        face_image_url = f"/uploads/users/{filename}"
                    else:
                        face_image_url = f"/uploads/faces/{request.user_id}/{uuid.uuid4()}.jpg"
                    
                    # Determine if this should be primary
                    is_primary = (i == 0 and request.is_primary) or (i == 0 and not primary_face_processed)
                    
                    # Create face embedding record
                    face_embedding = FaceEmbedding(
                        user_id=request.user_id,
                        embedding=embedding_b64,
                        face_image_url=face_image_url,
                        confidence_score=self._convert_numpy_types(face["confidence"]),
                        face_quality_score=self._convert_numpy_types(face["quality_score"]),
                        face_bbox=self._convert_numpy_types(face["bbox"]),
                        landmarks=self._convert_numpy_types(face["landmarks"]),
                        source_image=f"{request.source_description} (Face {i+1})",
                        is_primary=is_primary,
                        face_size=self._convert_numpy_types(face.get("size", {"width": 0, "height": 0})),
                        face_angle=self._convert_numpy_types(face.get("angle", 0.0)),
                        lighting_score=self._convert_numpy_types(face.get("lighting_score", 0.0)),
                        blur_score=self._convert_numpy_types(face.get("blur_score", 0.0))
                    )
                    
                    # If this is primary, unset other primary embeddings for this user
                    if is_primary:
                        db.query(FaceEmbedding).filter(
                            and_(
                                FaceEmbedding.user_id == request.user_id,
                                FaceEmbedding.is_primary == True
                            )
                        ).update({"is_primary": False})
                        primary_face_processed = True
                    
                    db.add(face_embedding)
                    db.flush()
                    
                    created_embeddings.append({
                        "id": str(face_embedding.id),
                        "confidence_score": face_embedding.confidence_score,
                        "face_quality_score": face_embedding.face_quality_score,
                        "is_primary": face_embedding.is_primary,
                        "face_number": i + 1,
                        "total_faces": len(faces)
                    })
                    
                except Exception as face_error:
                    logger.error(f"Failed to process face {i+1}: {str(face_error)}")
                    continue
            
            # Commit all face embeddings
            db.commit()
            
            # Reload face embeddings
            self._load_face_embeddings(db)
            
            processing_time = (time.time() - start_time) * 1000
            
            if created_embeddings:
                return {
                    "success": True,
                    "face_embedding_id": created_embeddings[0]["id"],
                    "confidence_score": created_embeddings[0]["confidence_score"],
                    "face_quality_score": created_embeddings[0]["face_quality_score"],
                    "message": f"Successfully processed {len(created_embeddings)} faces from image",
                    "processing_time_ms": round(processing_time, 2),
                    "total_faces_processed": len(created_embeddings),
                    "embeddings": created_embeddings
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to process any faces from image",
                    "processing_time_ms": round(processing_time, 2)
                }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Face upload failed: {str(e)}")
            return {
                "success": False,
                "message": f"Face upload failed: {str(e)}",
                "processing_time_ms": round(processing_time, 2)
            }
    
    async def _create_face_detection(
        self, 
        db: Session, 
        camera_id: uuid.UUID, 
        face: Dict[str, Any], 
        user_id: uuid.UUID, 
        confidence: float
    ):
        """Create face detection record."""
        try:
            # Save face image
            face_image_url = f"detections/{camera_id}/{uuid.uuid4()}.jpg"
            
            # Create detection record
            detection = FaceDetection(
                camera_id=camera_id,
                timestamp=datetime.utcnow().isoformat(),
                confidence_score=face["confidence"],
                face_bbox=face["bbox"],
                landmarks=face["landmarks"],
                recognized_user_id=user_id,
                recognition_confidence=confidence,
                face_image_url=face_image_url
            )
            
            db.add(detection)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to create face detection: {str(e)}")
    
    async def cleanup_old_detections(self, db: Session, days: int = 30):
        """Clean up old face detection records."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old detections
            deleted_count = db.query(FaceDetection).filter(
                FaceDetection.created_at < cutoff_date
            ).delete()
            
            db.commit()
            logger.info(f"Cleaned up {deleted_count} old face detection records")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old detections: {str(e)}")
            db.rollback()
    
    def check_attendance_cooldown(self, user_name: str, camera_id: str) -> bool:
        """Check if attendance can be marked (cooldown period)."""
        try:
            now = time.time()
            key = (user_name, camera_id)
            
            if now - self.cooldown_track[key] > self.COOLDOWN_SECONDS:
                self.cooldown_track[key] = now
                return True  # Can mark attendance
            else:
                remaining = self.COOLDOWN_SECONDS - (now - self.cooldown_track[key])
                logger.info(f"COOLDOWN: {user_name} recently marked on {camera_id}, {remaining:.1f}s remaining")
                return False  # Cannot mark attendance yet
                
        except Exception as e:
            logger.error(f"Error checking attendance cooldown: {str(e)}")
            return True  # Default to allowing attendance if check fails


# Global face recognition service instance
face_recognition_service = FaceRecognitionService()
