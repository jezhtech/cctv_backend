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
                    
                    results.append({
                        "embedding": embedding,
                        "bbox": face_bbox,
                        "landmarks": landmarks,
                        "confidence": float(face.det_score),
                        "quality_score": quality_score
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
    
    def _find_similar_faces(self, query_embedding: np.ndarray, threshold: float = None) -> List[Tuple[int, float]]:
        """Find similar faces using FAISS index."""
        try:
            if self.face_index is None or len(self.face_embeddings) == 0:
                return []
            
            threshold = threshold or settings.FACE_RECOGNITION_THRESHOLD
            
            # Search for similar embeddings
            query_vector = query_embedding.reshape(1, -1)
            similarities, indices = self.face_index.search(query_vector, k=min(10, len(self.face_embeddings)))
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= threshold and idx < len(self.face_embeddings):
                    embedding_id = self.face_embeddings[idx]
                    results.append((embedding_id, float(similarity)))
            
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
            
            if len(faces) > 1:
                return {
                    "success": False,
                    "message": "Multiple faces detected. Please upload image with single face.",
                    "processing_time_ms": 0
                }
            
            face = faces[0]
            
            # Encode embedding to base64
            embedding_bytes = face["embedding"].tobytes()
            embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
            
            # Save face image (you might want to implement file storage service)
            face_image_url = f"faces/{user.id}/{uuid.uuid4()}.jpg"
            
            # Create face embedding record
            face_embedding = FaceEmbedding(
                user_id=request.user_id,
                embedding=embedding_b64,
                face_image_url=face_image_url,
                confidence_score=face["confidence"],
                face_quality_score=face["quality_score"],
                face_bbox=face["bbox"],
                landmarks=face["landmarks"],
                source_image=request.source_description,
                is_primary=request.is_primary
            )
            
            # If this is primary, unset other primary embeddings
            if request.is_primary:
                db.query(FaceEmbedding).filter(
                    and_(
                        FaceEmbedding.user_id == request.user_id,
                        FaceEmbedding.is_primary == True
                    )
                ).update({"is_primary": False})
            
            db.add(face_embedding)
            db.commit()
            db.refresh(face_embedding)
            
            # Reload face embeddings
            self._load_face_embeddings(db)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "face_embedding_id": face_embedding.id,
                "confidence_score": face["confidence"],
                "face_quality_score": face["quality_score"],
                "message": "Face image uploaded and processed successfully",
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
