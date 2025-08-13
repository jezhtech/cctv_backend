"""Legacy configuration file - imports from new structure."""
from app.core.config import settings

# For backward compatibility
__all__ = ['settings']
