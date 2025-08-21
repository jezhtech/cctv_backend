"""User service for managing users, face embeddings, and attendance."""
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from loguru import logger
import uuid
from datetime import datetime, timedelta
from passlib.context import CryptContext

from app.models.database.user import User, FaceEmbedding, Attendance
from app.schemas.user import UserCreate, UserUpdate, AttendanceCreate, AttendanceUpdate

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserService:
    """Service for managing users and related data."""
    
    def __init__(self):
        pass
    
    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    async def create_user(self, db: Session, user_data: UserCreate) -> User:
        """Create a new user."""
        try:
            # Check if email already exists
            existing_user = db.query(User).filter(User.email == user_data.email).first()
            if existing_user:
                raise ValueError("User with this email already exists")
            
            # Check if employee_id already exists
            if user_data.employee_id:
                existing_employee = db.query(User).filter(User.employee_id == user_data.employee_id).first()
                if existing_employee:
                    raise ValueError("User with this employee ID already exists")
            
            # Hash password
            hashed_password = self._hash_password(user_data.password)
            
            # Create user instance
            user_dict = user_data.dict()
            user_dict.pop("password")
            user_dict["password_hash"] = hashed_password
            
            user = User(**user_dict)
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"User created successfully: {user.id}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create user: {str(e)}")
            raise
    
    async def get_user(self, db: Session, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID."""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {str(e)}")
            return None
    
    async def get_users(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = False
    ) -> Tuple[List[User], int]:
        """Get paginated list of users."""
        try:
            query = db.query(User)
            
            if active_only:
                query = query.filter(User.is_active == True)
            
            total = query.count()
            users = query.offset(skip).limit(limit).all()
            
            return users, total
        except Exception as e:
            logger.error(f"Failed to get users: {str(e)}")
            return [], 0
    
    async def update_user(
        self, 
        db: Session, 
        user_id: uuid.UUID, 
        user_data: UserUpdate
    ) -> Optional[User]:
        """Update user information."""
        try:
            user = await self.get_user(db, user_id)
            if not user:
                return None
            
            # Check email uniqueness if changing
            if user_data.email and user_data.email != user.email:
                existing_user = db.query(User).filter(
                    and_(User.email == user_data.email, User.id != user_id)
                ).first()
                if existing_user:
                    raise ValueError("User with this email already exists")
            
            # Check employee_id uniqueness if changing
            if user_data.employee_id and user_data.employee_id != user.employee_id:
                existing_employee = db.query(User).filter(
                    and_(User.employee_id == user_data.employee_id, User.id != user_id)
                ).first()
                if existing_employee:
                    raise ValueError("User with this employee ID already exists")
            
            # Update fields
            update_data = user_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            
            logger.info(f"User updated successfully: {user_id}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update user {user_id}: {str(e)}")
            raise
    
    async def delete_user(self, db: Session, user_id: uuid.UUID) -> bool:
        """Hard delete user permanently."""
        try:
            user = await self.get_user(db, user_id)
            if not user:
                return False
            
            # Hard delete - permanently remove from database
            db.delete(user)
            db.commit()
            logger.info(f"User permanently deleted: {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete user {user_id}: {str(e)}")
            return False
    
    async def get_user_face_embeddings(
        self, 
        db: Session, 
        user_id: uuid.UUID
    ) -> List[FaceEmbedding]:
        """Get face embeddings for a user."""
        try:
            embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.user_id == user_id
            ).order_by(desc(FaceEmbedding.is_primary), desc(FaceEmbedding.created_at)).all()
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get face embeddings for user {user_id}: {str(e)}")
            return []
    
    async def delete_face_embedding(
        self, 
        db: Session, 
        embedding_id: uuid.UUID
    ) -> bool:
        """Delete a face embedding."""
        try:
            embedding = db.query(FaceEmbedding).filter(FaceEmbedding.id == embedding_id).first()
            if not embedding:
                return False
            
            db.delete(embedding)
            db.commit()
            
            logger.info(f"Face embedding deleted successfully: {embedding_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete face embedding {embedding_id}: {str(e)}")
            return False
    
    async def create_attendance(
        self, 
        db: Session, 
        attendance_data: AttendanceCreate
    ) -> Attendance:
        """Create a new attendance record."""
        try:
            # Check if user exists
            user = await self.get_user(db, attendance_data.user_id)
            if not user:
                raise ValueError("User not found")
            
            # Create attendance instance
            attendance = Attendance(**attendance_data.dict())
            db.add(attendance)
            db.commit()
            db.refresh(attendance)
            
            logger.info(f"Attendance record created successfully: {attendance.id}")
            return attendance
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create attendance: {str(e)}")
            raise
    
    async def get_attendance(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[uuid.UUID] = None,
        camera_id: Optional[uuid.UUID] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Tuple[List[Attendance], int]:
        """Get paginated list of attendance records."""
        try:
            query = db.query(Attendance).join(User)
            
            # Apply filters
            if user_id:
                query = query.filter(Attendance.user_id == user_id)
            
            if camera_id:
                query = query.filter(Attendance.camera_id == camera_id)
            
            if date_from:
                try:
                    from_date = datetime.fromisoformat(date_from)
                    query = query.filter(Attendance.created_at >= from_date)
                except ValueError:
                    logger.warning(f"Invalid date_from format: {date_from}")
            
            if date_to:
                try:
                    to_date = datetime.fromisoformat(date_to)
                    query = query.filter(Attendance.created_at <= to_date)
                except ValueError:
                    logger.warning(f"Invalid date_to format: {date_to}")
            
            total = query.count()
            attendances = query.offset(skip).limit(limit).order_by(desc(Attendance.created_at)).all()
            
            return attendances, total
        except Exception as e:
            logger.error(f"Failed to get attendance: {str(e)}")
            return [], 0
    
    async def get_attendance_record(
        self, 
        db: Session, 
        attendance_id: uuid.UUID
    ) -> Optional[Attendance]:
        """Get attendance record by ID."""
        try:
            attendance = db.query(Attendance).filter(Attendance.id == attendance_id).first()
            return attendance
        except Exception as e:
            logger.error(f"Failed to get attendance record {attendance_id}: {str(e)}")
            return None
    
    async def update_attendance(
        self,
        db: Session,
        attendance_id: uuid.UUID,
        attendance_data: AttendanceUpdate
    ) -> Optional[Attendance]:
        """Update attendance record."""
        try:
            attendance = await self.get_attendance_record(db, attendance_id)
            if not attendance:
                return None
            
            # Update fields
            update_data = attendance_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(attendance, field, value)
            
            attendance.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(attendance)
            
            logger.info(f"Attendance record updated successfully: {attendance_id}")
            return attendance
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update attendance {attendance_id}: {str(e)}")
            return None
    
    async def delete_attendance(
        self, 
        db: Session, 
        attendance_id: uuid.UUID
    ) -> bool:
        """Delete attendance record."""
        try:
            attendance = await self.get_attendance_record(db, attendance_id)
            if not attendance:
                return False
            
            db.delete(attendance)
            db.commit()
            
            logger.info(f"Attendance record deleted successfully: {attendance_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete attendance {attendance_id}: {str(e)}")
            return False
    
    async def get_user_statistics(self, db: Session, user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get user statistics."""
        try:
            user = await self.get_user(db, user_id)
            if not user:
                return None
            
            # Get attendance statistics
            total_attendances = db.query(Attendance).filter(Attendance.user_id == user_id).count()
            
            # Get this month's attendance
            this_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            monthly_attendances = db.query(Attendance).filter(
                and_(
                    Attendance.user_id == user_id,
                    Attendance.created_at >= this_month
                )
            ).count()
            
            # Get face embeddings count
            face_embeddings_count = db.query(FaceEmbedding).filter(
                FaceEmbedding.user_id == user_id
            ).count()
            
            # Get last attendance
            last_attendance = db.query(Attendance).filter(
                Attendance.user_id == user_id
            ).order_by(desc(Attendance.created_at)).first()
            
            return {
                "user_id": user_id,
                "total_attendances": total_attendances,
                "monthly_attendances": monthly_attendances,
                "face_embeddings_count": face_embeddings_count,
                "last_attendance": last_attendance.created_at if last_attendance else None,
                "has_face_data": face_embeddings_count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get user statistics {user_id}: {str(e)}")
            return None
    
    async def get_attendance_statistics(
        self, 
        db: Session, 
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get attendance statistics."""
        try:
            query = db.query(Attendance)
            
            # Apply date filters
            if date_from:
                try:
                    from_date = datetime.fromisoformat(date_from)
                    query = query.filter(Attendance.created_at >= from_date)
                except ValueError:
                    logger.warning(f"Invalid date_from format: {date_from}")
            
            if date_to:
                try:
                    to_date = datetime.fromisoformat(date_to)
                    query = query.filter(Attendance.created_at <= to_date)
                except ValueError:
                    logger.warning(f"Invalid date_to format: {date_to}")
            
            # Get statistics
            total_attendances = query.count()
            unique_users = query.distinct(Attendance.user_id).count()
            
            # Get today's statistics
            today = datetime.utcnow().date()
            today_attendances = db.query(Attendance).filter(
                func.date(Attendance.created_at) == today
            ).count()
            
            # Get this week's statistics
            week_ago = datetime.utcnow() - timedelta(days=7)
            weekly_attendances = db.query(Attendance).filter(
                Attendance.created_at >= week_ago
            ).count()
            
            # Get this month's statistics
            this_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            monthly_attendances = db.query(Attendance).filter(
                Attendance.created_at >= this_month
            ).count()
            
            return {
                "total_attendances": total_attendances,
                "unique_users": unique_users,
                "today_attendances": today_attendances,
                "weekly_attendances": weekly_attendances,
                "monthly_attendances": monthly_attendances,
                "date_range": {
                    "from": date_from,
                    "to": date_to
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get attendance statistics: {str(e)}")
            return {"error": str(e)}


# Global user service instance
user_service = UserService()
