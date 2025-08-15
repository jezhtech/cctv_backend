"""Streaming service for handling RTSP camera streams directly in FastAPI."""
import cv2
import subprocess
import threading
import time
import asyncio
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger
import uuid
from datetime import datetime
import numpy as np
import json
from sqlalchemy import and_, func

from app.models.database.camera import Camera, CameraStream
from app.core.config import settings


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
        self.max_buffer_size = 60  # Keep last 60 frames for high FPS
        
        # In-memory frame storage (replaces Redis)
        self.latest_frame = None
        self.frame_timestamp = None
        
        # Stream info
        self.stream_info = {}
        self.detection_interval = 60  # Process every 60 frames for high FPS target
        
        # Face detection settings
        self.detection_interval = max(1, int(60 / camera.frame_rate))  # Detect every N frames for 60 FPS target
        self.last_detection_time = 0
        
        # Performance monitoring for high FPS
        self.performance_metrics = {
            'avg_frame_time': 0.0,
            'max_frame_time': 0.0,
            'min_frame_time': float('inf'),
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
                        
                        # Process frame for face detection
                        await self._process_frame_for_faces_async(frame_result)
                        
                        # Increment frame count
                        self.frame_count += 1
                        self.last_frame_time = datetime.utcnow()
                        
                        # Calculate FPS based on frame processing
                        frame_times.append(start_frame_time)
                        if len(frame_times) > 60:  # Increased buffer for higher FPS
                            frame_times.pop(0)
                        
                        if len(frame_times) > 1:
                            try:
                                self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                            except ZeroDivisionError:
                                self.fps = 0.0
                        
                        # Performance monitoring for high FPS optimization
                        frame_processing_time = time.time() - start_frame_time
                        self.performance_metrics['frame_time_history'].append(frame_processing_time)
                        if len(self.performance_metrics['frame_time_history']) > 100:
                            self.performance_metrics['frame_time_history'].pop(0)
                        
                        # Update performance metrics
                        if self.performance_metrics['frame_time_history']:
                            self.performance_metrics['avg_frame_time'] = sum(self.performance_metrics['frame_time_history']) / len(self.performance_metrics['frame_time_history'])
                            self.performance_metrics['max_frame_time'] = max(self.performance_metrics['frame_time_history'])
                            self.performance_metrics['min_frame_time'] = min(self.performance_metrics['frame_time_history'])
                        
                        # Log stream health every 60 frames (reduced logging overhead)
                        if self.frame_count % 60 == 0:
                            logger.debug(f"FastAPI stream healthy for camera {self.camera_id} - {self.frame_count} frames, FPS: {self.fps:.2f}, Avg frame time: {self.performance_metrics['avg_frame_time']*1000:.2f}ms")
                        
                        # Reset consecutive errors on successful frame
                        consecutive_errors = 0
                        
                        # Minimal delay for high frame rate streaming - reduced from 10ms to 1ms
                        # This allows for up to 1000 FPS theoretical maximum
                        await asyncio.sleep(0.001)  # Check every 1ms for high-performance streaming
                        
                except Exception as e:
                    logger.error(f"Error monitoring FastAPI stream for camera {self.camera_id}: {str(e)}")
                    consecutive_errors += 1
                    self.errors_count += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors for camera {self.camera_id}, stopping stream")
                        break
                    
                    await asyncio.sleep(0.01)  # Minimal delay for error recovery
                    
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
            if self.frame_count % 120 == 0:
                logger.debug(f"Stored frame {self.frame_count} in memory for camera {self.camera_id}")
                
        except Exception as e:
            logger.debug(f"Failed to store frame in memory: {str(e)}")
    
    async def _process_frame_for_faces_async(self, frame: np.ndarray):
        """Process frame for face detection and recognition asynchronously."""
        try:
            # Skip if too soon since last detection - reduced from 1 second to 0.1 seconds for higher FPS
            current_time = time.time()
            if current_time - self.last_detection_time < 0.1:  # Max 10 detections per second for high FPS
                return
            
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame received for camera {self.camera_id}")
                return
            
            # Simple face detection placeholder
            # For now, just log that we're processing frames
            if self.frame_count % 60 == 0:  # Log every 60 frames (reduced logging overhead)
                logger.debug(f"Processing frame {self.frame_count} for face detection in camera {self.camera_id}")
                logger.debug(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            self.last_detection_time = current_time
            
            # Update metrics every 60 frames (reduced overhead)
            if self.frame_count % 60 == 0:
                self._update_metrics()
            
        except Exception as e:
            logger.warning(f"Face detection processing error for camera {self.camera_id}: {str(e)}")
            # Don't crash the stream for face detection errors
    
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
                "latest_frame_timestamp": self.frame_timestamp.isoformat() if self.frame_timestamp else None
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
                'avg_frame_time_ms': round(self.performance_metrics['avg_frame_time'] * 1000, 2),
                'max_frame_time_ms': round(self.performance_metrics['max_frame_time'] * 1000, 2),
                'min_frame_time_ms': round(self.performance_metrics['min_frame_time'] * 1000, 2),
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
                "frame_buffer_size": len(self.frame_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error getting stream status for camera {self.camera_id}: {str(e)}")
            return {
                "is_running": False,
                "camera_id": str(self.camera_id),
                "error": str(e)
            }


# Global stream manager instance for FastAPI
class StreamManager:
    """Manages multiple camera streams directly in FastAPI process using in-memory storage."""
    
    def __init__(self):
        # Active stream processors
        self.active_streams: Dict[uuid.UUID, StreamProcessor] = {}
        self._lock = asyncio.Lock()
    
    async def start_stream(self, camera_id: uuid.UUID, camera: Camera) -> bool:
        """Start a new camera stream asynchronously."""
        async with self._lock:
            try:
                # Check if stream is already running
                if camera_id in self.active_streams:
                    logger.warning(f"Stream already running for camera {camera_id}")
                    return True
                
                # Log camera details for debugging
                logger.info(f"Starting FastAPI stream for camera {camera_id}: {camera.name}")
                logger.info(f"Camera IP: {camera.ip_address}:{camera.port}")
                logger.info(f"Camera path: {camera.path}")
                logger.info(f"Camera credentials: username={camera.username}, password={'*' * len(camera.password) if camera.password else 'None'}")
                
                # Create new stream processor
                stream_processor = StreamProcessor(camera_id, camera)
                if await stream_processor.start_async():
                    # Store in active streams
                    self.active_streams[camera_id] = stream_processor
                    
                    logger.info(f"Started FastAPI stream for camera {camera_id}")
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
                # Get stream processor
                stream_processor = self.active_streams.get(camera_id)
                if stream_processor:
                    # Stop the processor
                    await stream_processor.stop_async()
                    
                    # Remove from active streams
                    del self.active_streams[camera_id]
                
                logger.info(f"Stopped FastAPI stream for camera {camera_id}")
                return True
                    
            except Exception as e:
                logger.error(f"Error stopping FastAPI stream for camera {camera_id}: {str(e)}")
                return False
    
    def get_stream_status(self, camera_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get streaming status for a camera."""
        try:
            # Check local active streams
            if camera_id in self.active_streams:
                return self.active_streams[camera_id].get_status()
            
            logger.debug(f"No processor found for camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting stream status for camera {camera_id}: {str(e)}")
            return None
    
    def get_processor(self, camera_id: uuid.UUID) -> Optional[StreamProcessor]:
        """Get a specific processor instance."""
        return self.active_streams.get(camera_id)
    
    def is_stream_active(self, camera_id: uuid.UUID) -> bool:
        """Check if a stream is actually active for a camera."""
        try:
            # Check local active streams
            if camera_id in self.active_streams:
                return self.active_streams[camera_id].is_running
            
            logger.debug(f"No processor found for camera {camera_id}")
            return False
            
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
    
    def get_all_streams_status(self) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Get status of all active streams."""
        try:
            all_statuses = {}
            
            # Get local active streams
            for camera_id, processor in self.active_streams.items():
                all_statuses[camera_id] = processor.get_status()
            
            return all_statuses
            
        except Exception as e:
            logger.error(f"Error getting all stream statuses: {str(e)}")
            return {}
    
    async def stop_all_streams(self):
        """Stop all active streams."""
        async with self._lock:
            try:
                camera_ids = list(self.active_streams.keys())
                for camera_id in camera_ids:
                    await self.stop_stream(camera_id)
                logger.info(f"Stopped all {len(camera_ids)} active streams")
            except Exception as e:
                logger.error(f"Error stopping all streams: {str(e)}")


# Global FastAPI stream manager instance
stream_manager = StreamManager()
