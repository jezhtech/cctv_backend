"""Face recognition service using InsightFace."""
import cv2
import numpy as np
import base64
import time
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from loguru import logger
import uuid
from datetime import datetime, timedelta
import insightface
from insightface.app import FaceAnalysis
import faiss
import pickle
import os

from app.models.database.user import User, FaceEmbedding, FaceDetection, Attendance
from app.schemas.user import FaceRecognitionRequest, FaceUploadRequest
from app.core.config import settings
from app.core.celery_app import celery_app


class FaceRecognitionService:
    """Service for face detection and recognition."""
    
    def __init__(self):
        self.face_analyzer = None
        self.face_index = None
        self.face_embeddings = {}
        self.user_embeddings = {}
        self.is_initialized = False
        self.initialization_lock = False
        
        # Initialize face analyzer
        self._initialize_face_analyzer()
    
    def _initialize_face_analyzer(self):
        """Initialize InsightFace analyzer."""
        try:
            if self.initialization_lock:
                return
            
            self.initialization_lock = True
            logger.info("Initializing InsightFace analyzer...")
            
            # Initialize InsightFace
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l",
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(
                ctx_id=0,
                det_size=(640, 640),
                det_thresh=settings.FACE_DETECTION_CONFIDENCE
            )
            
            logger.info("InsightFace analyzer initialized successfully")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {str(e)}")
            self.is_initialized = False
        finally:
            self.initialization_lock = False
    
    def _load_face_embeddings(self, db: Session):
        """Load face embeddings from database into memory."""
        try:
            if not self.is_initialized:
                return
            
            logger.info("Loading face embeddings from database...")
            
            # Get all face embeddings
            embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.confidence_score >= settings.FACE_RECOGNITION_THRESHOLD
            ).all()
            
            if not embeddings:
                logger.info("No face embeddings found in database")
                return
            
            # Build FAISS index
            embedding_dim = 512  # InsightFace embedding dimension
            self.face_index = faiss.IndexFlatIP(embedding_dim)
            
            # Store embeddings and build index
            for embedding in embeddings:
                embedding_vector = np.frombuffer(
                    base64.b64decode(embedding.embedding), 
                    dtype=np.float32
                )
                
                self.face_index.add(embedding_vector.reshape(1, -1))
                self.face_embeddings[len(self.face_embeddings)] = embedding.id
                self.user_embeddings[embedding.id] = embedding.user_id
            
            logger.info(f"Loaded {len(embeddings)} face embeddings into memory")
            
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
        """Extract face embeddings from image."""
        try:
            if not self.is_initialized:
                return []
            
            # Analyze faces in image
            faces = self.face_analyzer.get(image)
            
            results = []
            for face in faces:
                if face.det_score >= settings.FACE_DETECTION_CONFIDENCE:
                    # Get embedding
                    embedding = face.embedding
                    
                    # Get bounding box
                    bbox = face.bbox.astype(int)
                    face_bbox = {
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "width": int(bbox[2] - bbox[0]),
                        "height": int(bbox[3] - bbox[1])
                    }
                    
                    # Get landmarks
                    landmarks = face.kps.astype(int).tolist() if hasattr(face, 'kps') else None
                    
                    # Calculate quality score
                    quality_score = self._calculate_face_quality(image, face_bbox)
                    
                    # Calculate additional metrics
                    face_size = {"width": face_bbox["width"], "height": face_bbox["height"]}
                    face_angle = self._calculate_face_angle(face_bbox, landmarks)
                    lighting_score = self._calculate_lighting_score(image, face_bbox)
                    blur_score = self._calculate_blur_score(image, face_bbox)
                    
                    results.append({
                        "embedding": embedding,
                        "bbox": face_bbox,
                        "landmarks": landmarks,
                        "confidence": float(face.det_score),
                        "quality_score": quality_score,
                        "size": face_size,
                        "angle": face_angle,
                        "lighting_score": lighting_score,
                        "blur_score": blur_score
                    })
            
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
    
    def _calculate_face_angle(self, bbox: Dict[str, int], landmarks: List[List[int]]) -> float:
        """Calculate face angle/orientation based on landmarks."""
        try:
            if not landmarks or len(landmarks) < 5:
                return 0.0
            
            # Use eye landmarks to calculate face angle
            # Assuming landmarks[0] and landmarks[1] are left and right eye centers
            if len(landmarks) >= 2:
                left_eye = np.array(landmarks[0])
                right_eye = np.array(landmarks[1])
                
                # Calculate angle between eyes
                eye_vector = right_eye - left_eye
                angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
                
                # Normalize to -90 to 90 degrees
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                
                return float(angle)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate face angle: {str(e)}")
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
    
    def _find_similar_faces(self, query_embedding: np.ndarray, threshold: float = None) -> List[Tuple[int, float]]:
        """Find similar faces using FAISS index."""
        try:
            if self.face_index is None or len(self.face_embeddings) == 0:
                return []
            
            threshold = threshold or settings.FACE_RECOGNITION_THRESHOLD
            
            # Search for similar embeddings
            query_vector = query_embedding.reshape(1, -1)
            similarities, indices = self.face_index.search(query_vector, k=min(10, len(self.face_embeddings)))
            
            # Track best match per user to avoid duplicate user matches
            best_per_user = {}
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= threshold and idx < len(self.face_embeddings):
                    embedding_id = self.face_embeddings[idx]
                    user_id = self.user_embeddings.get(embedding_id)
                    
                    if user_id:
                        # Keep only the best match per user
                        if user_id not in best_per_user or similarity > best_per_user[user_id][1]:
                            best_per_user[user_id] = (embedding_id, float(similarity))
            
            # Return results sorted by similarity (best first)
            results = list(best_per_user.values())
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Add additional logging to debug face matching
            if results:
                logger.debug(f"Face recognition results: {len(results)} matches found")
                for i, (embedding_id, similarity) in enumerate(results):
                    user_id = self.user_embeddings.get(embedding_id)
                    logger.debug(f"  Match {i+1}: User {user_id}, Similarity: {similarity:.3f}")
            
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
            best_match_id, confidence_score = similar_faces[0]
            user_id = self.user_embeddings.get(best_match_id)
            
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
            
            # Process all detected faces (support multiple faces)
            created_embeddings = []
            primary_face_processed = False
            
            for i, face in enumerate(faces):
                try:
                    # Encode embedding to base64
                    embedding_bytes = face["embedding"].tobytes()
                    embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                    
                    # Use the actual uploaded image URL from our storage
                    # The image is already saved in uploads/users/ directory
                    # We'll construct the URL based on the source description
                    if request.source_description and "Profile image:" in request.source_description:
                        # Extract filename from source description (e.g., "Profile image: img_001_1703123456789_a1b2c3")
                        filename = request.source_description.split("Profile image: ")[-1]
                        face_image_url = f"/uploads/users/{filename}"
                    else:
                        # Fallback for other sources
                        face_image_url = f"/uploads/faces/{request.user_id}/{uuid.uuid4()}.jpg"
                    
                    # Determine if this should be primary (first face or explicitly marked)
                    is_primary = (i == 0 and request.is_primary) or (i == 0 and not primary_face_processed)
                    
                    # Create face embedding record with all numpy types converted to native Python types
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
                        # Add new quality metrics - convert numpy types to native Python types
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
                    db.flush()  # Flush to get the ID
                    
                    # Add to created embeddings list
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
                    "face_embedding_id": created_embeddings[0]["id"],  # Return first one for backward compatibility
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


# Global face recognition service instance
face_recognition_service = FaceRecognitionService()


# Celery tasks
@celery_app.task(name="app.services.face_recognition.face_service.cleanup_old_detections")
def cleanup_old_detections():
    """Periodic task to cleanup old face detection records."""
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        face_recognition_service.cleanup_old_detections(db)
    finally:
        db.close()
