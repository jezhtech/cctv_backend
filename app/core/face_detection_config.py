"""Face detection configuration settings."""
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class FaceDetectionSettings(BaseSettings):
    """Face detection specific configuration settings."""
    
    # Detection Settings
    DETECTION_INTERVAL_FRAMES: int = Field(
        default=60,
        description="Process every N frames for face detection"
    )
    MAX_DETECTIONS_PER_FRAME: int = Field(
        default=10,
        description="Maximum number of faces to detect per frame"
    )
    DETECTION_TIMEOUT_SECONDS: float = Field(
        default=0.1,
        description="Minimum time between detections to avoid overwhelming"
    )
    
    # Recognition Settings
    RECOGNITION_BATCH_SIZE: int = Field(
        default=5,
        description="Number of faces to process together for recognition"
    )
    RECOGNITION_TIMEOUT_SECONDS: float = Field(
        default=1.0,
        description="Timeout for face recognition processing"
    )
    
    # Performance Settings
    ENABLE_GPU_ACCELERATION: bool = Field(
        default=False,
        description="Enable GPU acceleration for face detection"
    )
    MAX_MEMORY_USAGE_MB: int = Field(
        default=1024,
        description="Maximum memory usage for face detection in MB"
    )
    ENABLE_BATCH_PROCESSING: bool = Field(
        default=True,
        description="Enable batch processing for multiple faces"
    )
    
    # Quality Settings
    MIN_FACE_SIZE_PIXELS: int = Field(
        default=80,
        description="Minimum face size in pixels for detection"
    )
    MAX_FACE_SIZE_PIXELS: int = Field(
        default=800,
        description="Maximum face size in pixels for detection"
    )
    QUALITY_THRESHOLD: float = Field(
        default=0.3,
        description="Minimum quality score for face processing"
    )
    
    # Storage Settings
    MAX_DETECTION_HISTORY: int = Field(
        default=100,
        description="Maximum number of detections to keep in memory per camera"
    )
    ENABLE_FACE_IMAGE_SAVING: bool = Field(
        default=True,
        description="Save detected face images to storage"
    )
    FACE_IMAGE_QUALITY: int = Field(
        default=80,
        description="JPEG quality for saved face images (1-100)"
    )
    
    # Monitoring Settings
    ENABLE_DETECTION_METRICS: bool = Field(
        default=True,
        description="Enable real-time detection metrics"
    )
    METRICS_UPDATE_INTERVAL: int = Field(
        default=60,
        description="Update metrics every N frames"
    )
    ENABLE_DETECTION_LOGGING: bool = Field(
        default=True,
        description="Enable detailed logging for face detection"
    )
    
    # Advanced Settings
    ENABLE_FACE_TRACKING: bool = Field(
        default=False,
        description="Enable face tracking across frames"
    )
    TRACKING_HISTORY_FRAMES: int = Field(
        default=30,
        description="Number of frames to keep for face tracking"
    )
    ENABLE_LANDMARK_DETECTION: bool = Field(
        default=True,
        description="Enable facial landmark detection"
    )
    ENABLE_ATTRIBUTE_ANALYSIS: bool = Field(
        default=False,
        description="Enable age, gender, and emotion analysis"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "FACE_DETECTION_"
        case_sensitive = True
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration as dictionary."""
        return {
            "detection_interval_frames": self.DETECTION_INTERVAL_FRAMES,
            "max_detections_per_frame": self.MAX_DETECTIONS_PER_FRAME,
            "detection_timeout_seconds": self.DETECTION_TIMEOUT_SECONDS,
            "min_face_size_pixels": self.MIN_FACE_SIZE_PIXELS,
            "max_face_size_pixels": self.MAX_FACE_SIZE_PIXELS,
            "quality_threshold": self.QUALITY_THRESHOLD
        }
    
    def get_recognition_config(self) -> Dict[str, Any]:
        """Get recognition configuration as dictionary."""
        return {
            "recognition_batch_size": self.RECOGNITION_BATCH_SIZE,
            "recognition_timeout_seconds": self.RECOGNITION_TIMEOUT_SECONDS,
            "enable_batch_processing": self.ENABLE_BATCH_PROCESSING
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration as dictionary."""
        return {
            "enable_gpu_acceleration": self.ENABLE_GPU_ACCELERATION,
            "max_memory_usage_mb": self.MAX_MEMORY_USAGE_MB,
            "max_detection_history": self.MAX_DETECTION_HISTORY
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration as dictionary."""
        return {
            "enable_face_image_saving": self.ENABLE_FACE_IMAGE_SAVING,
            "face_image_quality": self.FACE_IMAGE_QUALITY
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration as dictionary."""
        return {
            "enable_detection_metrics": self.ENABLE_DETECTION_METRICS,
            "metrics_update_interval": self.METRICS_UPDATE_INTERVAL,
            "enable_detection_logging": self.ENABLE_DETECTION_LOGGING
        }


# Global face detection settings instance
face_detection_settings = FaceDetectionSettings()


# Configuration presets for different use cases
class FaceDetectionPresets:
    """Predefined configuration presets for different scenarios."""
    
    @staticmethod
    def high_performance() -> Dict[str, Any]:
        """High performance preset for real-time applications."""
        return {
            "DETECTION_INTERVAL_FRAMES": 30,
            "MAX_DETECTIONS_PER_FRAME": 5,
            "DETECTION_TIMEOUT_SECONDS": 0.05,
            "ENABLE_GPU_ACCELERATION": True,
            "MAX_MEMORY_USAGE_MB": 2048,
            "ENABLE_BATCH_PROCESSING": True,
            "MIN_FACE_SIZE_PIXELS": 100,
            "MAX_DETECTION_HISTORY": 50
        }
    
    @staticmethod
    def high_accuracy() -> Dict[str, Any]:
        """High accuracy preset for security applications."""
        return {
            "DETECTION_INTERVAL_FRAMES": 10,
            "MAX_DETECTIONS_PER_FRAME": 20,
            "DETECTION_TIMEOUT_SECONDS": 0.2,
            "QUALITY_THRESHOLD": 0.5,
            "ENABLE_LANDMARK_DETECTION": True,
            "ENABLE_ATTRIBUTE_ANALYSIS": True,
            "MAX_DETECTION_HISTORY": 200,
            "FACE_IMAGE_QUALITY": 95
        }
    
    @staticmethod
    def low_resource() -> Dict[str, Any]:
        """Low resource preset for embedded systems."""
        return {
            "DETECTION_INTERVAL_FRAMES": 120,
            "MAX_DETECTIONS_PER_FRAME": 3,
            "DETECTION_TIMEOUT_SECONDS": 0.5,
            "ENABLE_GPU_ACCELERATION": False,
            "MAX_MEMORY_USAGE_MB": 256,
            "ENABLE_BATCH_PROCESSING": False,
            "MIN_FACE_SIZE_PIXELS": 120,
            "MAX_DETECTION_HISTORY": 25,
            "ENABLE_LANDMARK_DETECTION": False,
            "ENABLE_ATTRIBUTE_ANALYSIS": False
        }
    
    @staticmethod
    def balanced() -> Dict[str, Any]:
        """Balanced preset for general use."""
        return {
            "DETECTION_INTERVAL_FRAMES": 60,
            "MAX_DETECTIONS_PER_FRAME": 10,
            "DETECTION_TIMEOUT_SECONDS": 0.1,
            "ENABLE_GPU_ACCELERATION": False,
            "MAX_MEMORY_USAGE_MB": 1024,
            "ENABLE_BATCH_PROCESSING": True,
            "MIN_FACE_SIZE_PIXELS": 80,
            "MAX_DETECTION_HISTORY": 100,
            "ENABLE_LANDMARK_DETECTION": True,
            "ENABLE_ATTRIBUTE_ANALYSIS": False
        }


# Helper function to apply preset
def apply_face_detection_preset(preset_name: str) -> None:
    """Apply a predefined configuration preset."""
    presets = {
        "high_performance": FaceDetectionPresets.high_performance,
        "high_accuracy": FaceDetectionPresets.high_accuracy,
        "low_resource": FaceDetectionPresets.low_resource,
        "balanced": FaceDetectionPresets.balanced
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    preset_config = presets[preset_name]()
    
    # Apply preset values
    for key, value in preset_config.items():
        if hasattr(face_detection_settings, key):
            setattr(face_detection_settings, key, value)
    
    print(f"Applied {preset_name} preset to face detection settings")
