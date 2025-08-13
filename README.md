# Smart Attendance Backend

A production-level FastAPI backend for RTSP camera management with facial recognition capabilities. This system can handle multiple cameras simultaneously, perform real-time face detection and recognition, and track attendance automatically.

## Features

- **RTSP Camera Management**: Add, configure, and manage multiple RTSP cameras
- **Real-time Streaming**: Handle multiple camera streams simultaneously
- **Face Recognition**: Advanced facial recognition using InsightFace
- **Attendance Tracking**: Automatic attendance logging with face recognition
- **User Management**: Comprehensive user and face embedding management
- **Production Ready**: Built with production-level architecture and best practices
- **Scalable**: Designed to handle multiple cameras and users efficiently
- **Monitoring**: Built-in health checks and performance monitoring

## Architecture

```
smart-attendance-backend/
├── app/
│   ├── api/                    # API endpoints
│   │   └── v1/                # API version 1
│   │       ├── cameras.py     # Camera management endpoints
│   │       └── users.py       # User and face recognition endpoints
│   ├── core/                   # Core application components
│   │   ├── config.py          # Configuration management
│   │   ├── database.py        # Database connection and sessions
│   │   └── celery_app.py      # Celery configuration for background tasks
│   ├── models/                 # Database models
│   │   └── database/          # SQLAlchemy models
│   │       ├── base.py        # Base model classes
│   │       ├── camera.py      # Camera and stream models
│   │       └── user.py        # User, face, and attendance models
│   ├── schemas/                # Pydantic schemas
│   │   ├── camera.py          # Camera-related schemas
│   │   └── user.py            # User and face recognition schemas
│   ├── services/               # Business logic services
│   │   ├── camera/            # Camera management service
│   │   ├── face_recognition/  # Face recognition service
│   │   ├── streaming/         # RTSP streaming service
│   │   └── user/              # User management service
│   └── main.py                # Main FastAPI application
├── tests/                      # Test suite
├── models/                     # ML model storage
├── logs/                       # Application logs
├── uploads/                    # File uploads
├── docker-compose.yml          # Docker services configuration
├── pyproject.toml             # Poetry dependencies and configuration
└── README.md                  # This file
```

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **PostgreSQL**: Primary database
- **Redis**: Caching and message broker
- **Celery**: Background task processing
- **InsightFace**: Advanced face recognition library
- **OpenCV**: Computer vision and image processing
- **FAISS**: Efficient similarity search for face embeddings
- **Poetry**: Dependency management
- **Docker**: Containerization

## Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- PostgreSQL 15+
- Redis 7+
- Docker and Docker Compose (optional)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-attendance-backend
   ```

2. **Install dependencies using Poetry**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Run database migrations**
   ```bash
   poetry run alembic upgrade head
   ```

6. **Start the application**
   ```bash
   poetry run python -m app.main
   ```

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key (at least 32 characters)
- `FACE_MODEL_PATH`: Path to InsightFace model
- `RTSP_FRAME_RATE`: Default frame rate for cameras
- `FACE_RECOGNITION_THRESHOLD`: Confidence threshold for face recognition

### Camera Configuration

Each camera requires:
- **Name**: Human-readable camera identifier
- **RTSP URL**: Stream URL (e.g., `rtsp://192.168.1.100:554/stream`)
- **Credentials**: Username/password if required
- **Frame Rate**: Processing frame rate (1-30 fps)
- **Resolution**: Camera resolution settings

## API Endpoints

### Camera Management

- `POST /api/v1/cameras/` - Create new camera
- `GET /api/v1/cameras/` - List all cameras
- `GET /api/v1/cameras/{id}` - Get camera details
- `PUT /api/v1/cameras/{id}` - Update camera
- `DELETE /api/v1/cameras/{id}` - Delete camera
- `POST /api/v1/cameras/test-connection` - Test RTSP connection
- `GET /api/v1/cameras/{id}/health` - Camera health check
- `POST /api/v1/cameras/{id}/start-stream` - Start streaming
- `POST /api/v1/cameras/{id}/stop-stream` - Stop streaming

### User Management

- `POST /api/v1/users/` - Create new user
- `GET /api/v1/users/` - List all users
- `GET /api/v1/users/{id}` - Get user details
- `PUT /api/v1/users/{id}` - Update user
- `DELETE /api/v1/users/{id}` - Delete user

### Face Recognition

- `POST /api/v1/users/face-recognition` - Recognize face from image
- `POST /api/v1/users/upload-face` - Upload face image for user
- `POST /api/v1/users/upload-face-file` - Upload face image file
- `GET /api/v1/users/{id}/face-embeddings` - Get user's face embeddings
- `DELETE /api/v1/users/face-embeddings/{id}` - Delete face embedding

### Attendance

- `POST /api/v1/users/attendance` - Create attendance record
- `GET /api/v1/users/attendance` - List attendance records
- `GET /api/v1/users/attendance/{id}` - Get attendance details
- `PUT /api/v1/users/attendance/{id}` - Update attendance
- `DELETE /api/v1/users/attendance/{id}` - Delete attendance

## Usage Examples

### Adding a New Camera

```python
import requests

# Test camera connection first
test_data = {
    "rtsp_url": "rtsp://192.168.1.100:554/stream",
    "username": "admin",
    "password": "password123",
    "timeout": 30
}

response = requests.post(
    "http://localhost:8000/api/v1/cameras/test-connection",
    json=test_data
)

if response.json()["success"]:
    # Create camera
    camera_data = {
        "name": "Main Entrance",
        "description": "Camera at main building entrance",
        "rtsp_url": "rtsp://192.168.1.100:554/stream",
        "username": "admin",
        "password": "password123",
        "frame_rate": 5,
        "location": "Main Building Entrance"
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/cameras/",
        json=camera_data
    )
    
    camera_id = response.json()["id"]
    
    # Start streaming
    requests.post(f"http://localhost:8000/api/v1/cameras/{camera_id}/start-stream")
```

### Adding a User with Face Data

```python
import requests
import base64

# Create user
user_data = {
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@company.com",
    "password": "securepassword123",
    "employee_id": "EMP001",
    "department": "Engineering"
}

response = requests.post(
    "http://localhost:8000/api/v1/users/",
    json=user_data
)

user_id = response.json()["id"]

# Upload face image
with open("john_face.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

face_data = {
    "user_id": user_id,
    "image_data": image_data,
    "is_primary": True,
    "source_description": "Initial photo"
}

response = requests.post(
    "http://localhost:8000/api/v1/users/upload-face",
    json=face_data
)
```

## Development

### Running Tests

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app
```

### Code Quality

```bash
# Format code
poetry run black app/

# Sort imports
poetry run isort app/

# Lint code
poetry run flake8 app/

# Type checking
poetry run mypy app/
```

### Database Migrations

```bash
# Create new migration
poetry run alembic revision --autogenerate -m "Description"

# Apply migrations
poetry run alembic upgrade head

# Rollback migration
poetry run alembic downgrade -1
```

## Production Deployment

### Docker Deployment

1. **Build and start services**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Scale workers**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --scale celery=3
   ```

### Environment Variables for Production

- Set `DEBUG=false`
- Use strong `SECRET_KEY`
- Configure production database and Redis URLs
- Set appropriate CORS origins
- Configure logging levels

### Monitoring

- **Health Check**: `/health` endpoint
- **Celery Flower**: Monitor background tasks at `/flower`
- **Application Logs**: Check `logs/app.log`
- **Database**: Monitor PostgreSQL performance
- **Redis**: Monitor cache and message broker

## Troubleshooting

### Common Issues

1. **Camera Connection Failed**
   - Verify RTSP URL is accessible
   - Check network connectivity
   - Verify credentials if required

2. **Face Recognition Not Working**
   - Ensure InsightFace model is downloaded
   - Check image quality and face visibility
   - Verify face embeddings are properly stored

3. **High Memory Usage**
   - Reduce frame rate for cameras
   - Limit number of simultaneous streams
   - Monitor Celery worker memory usage

4. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check connection string format
   - Verify database exists and is accessible

### Logs

Check application logs in `logs/app.log` for detailed error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact: rajasgh18@gmail.com

## Acknowledgments

- InsightFace for advanced face recognition capabilities
- FastAPI community for the excellent web framework
- OpenCV for computer vision tools
