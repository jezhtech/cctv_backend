"""Celery configuration for background tasks."""
from celery import Celery
from .config import settings

# Create Celery instance
celery_app = Celery(
    "smart_attendance",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.services.camera.camera_service",
        "app.services.face_recognition.face_service",
        "app.services.streaming.stream_service",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    result_expires=3600,  # 1 hour
    result_persistent=True,
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    broker_transport_options={
        "visibility_timeout": 3600,
        "fanout_prefix": True,
        "fanout_patterns": True,
    },
)

# Task routing
celery_app.conf.task_routes = {
    "app.services.camera.*": {"queue": "camera"},
    "app.services.face_recognition.*": {"queue": "face_recognition"},
    "app.services.streaming.*": {"queue": "streaming"},
    "app.services.attendance.*": {"queue": "attendance"},
}

# Queue definitions
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_queues = {
    "default": {"exchange": "default", "routing_key": "default"},
    "camera": {"exchange": "camera", "routing_key": "camera"},
    "face_recognition": {"exchange": "face_recognition", "routing_key": "face_recognition"},
    "streaming": {"exchange": "streaming", "routing_key": "streaming"},
    "attendance": {"exchange": "attendance", "routing_key": "attendance"},
}

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "check-camera-health": {
        "task": "app.services.camera.camera_service.check_camera_health",
        "schedule": 60.0,  # Every minute
    },
    "cleanup-old-detections": {
        "task": "app.services.face_recognition.face_service.cleanup_old_detections",
        "schedule": 3600.0,  # Every hour
    },
    "update-attendance-records": {
        "task": "app.services.attendance.attendance_service.update_attendance_records",
        "schedule": 300.0,  # Every 5 minutes
    },
}

# Task annotations
celery_app.conf.task_annotations = {
    "app.services.camera.camera_service.*": {
        "rate_limit": "10/m",
        "time_limit": 300,
    },
    "app.services.face_recognition.face_service.*": {
        "rate_limit": "100/m",
        "time_limit": 600,
    },
    "app.services.streaming.stream_service.*": {
        "rate_limit": "50/m",
        "time_limit": 900,
    },
}

if __name__ == "__main__":
    celery_app.start()
