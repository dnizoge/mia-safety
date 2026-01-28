"""
MIA Backend - Database Session Management
==========================================

Async SQLite database session management using aiosqlite.

For simplicity in MVP, we use raw SQL with aiosqlite.
For production, consider SQLAlchemy async ORM.
"""

import logging
from pathlib import Path
from typing import Optional
import aiosqlite

from backend_api.core.config import settings

logger = logging.getLogger("mia.api.db")

# Global database connection
_db_connection: Optional[aiosqlite.Connection] = None


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA = """
-- Violation records table
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    frame_number INTEGER,
    timestamp_seconds REAL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_violations_class ON violations(class_name);
CREATE INDEX IF NOT EXISTS idx_violations_created ON violations(created_at);
CREATE INDEX IF NOT EXISTS idx_violations_source ON violations(source);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

async def init_db() -> None:
    """
    Initialize database connection and create schema.
    
    Called during application startup.
    """
    global _db_connection
    
    # Extract database path from URL
    # Format: sqlite+aiosqlite:///./path/to/db.sqlite
    db_url = settings.DATABASE_URL
    
    if db_url.startswith("sqlite+aiosqlite:///"):
        db_path = db_url.replace("sqlite+aiosqlite:///", "")
    elif db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
    else:
        db_path = "data/mia_violations.db"
    
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Open connection
    _db_connection = await aiosqlite.connect(db_path)
    
    # Enable WAL mode for better concurrency
    await _db_connection.execute("PRAGMA journal_mode=WAL")
    
    # Create schema
    await _db_connection.executescript(SCHEMA)
    await _db_connection.commit()
    
    logger.info(f"Database initialized: {db_path}")


async def close_db() -> None:
    """
    Close database connection.
    
    Called during application shutdown.
    """
    global _db_connection
    
    if _db_connection is not None:
        await _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed")


async def get_db() -> aiosqlite.Connection:
    """
    Get database connection.
    
    Returns:
        Active database connection
        
    Raises:
        RuntimeError: If database not initialized
    """
    if _db_connection is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    return _db_connection
