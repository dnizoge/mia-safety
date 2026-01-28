"""
MIA Backend - Violation Service
================================

Service for logging and querying safety violations.

Handles:
- Logging new violations to database
- Querying violation history
- Generating statistics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from backend_api.db.session import get_db

logger = logging.getLogger("mia.api.violation_service")


class ViolationService:
    """
    Service for managing violation records.
    
    Provides CRUD operations and statistics for violations.
    """
    
    async def log_violation(
        self,
        source: str,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Log a new violation to the database.
        
        Args:
            source: Video source (file path or stream ID)
            class_name: Violation class name
            confidence: Detection confidence
            bbox: Bounding box (x1, y1, x2, y2)
            frame_number: Optional frame number
            timestamp: Optional timestamp in seconds
            
        Returns:
            ID of inserted record
        """
        try:
            db = await get_db()
            
            x1, y1, x2, y2 = bbox
            
            cursor = await db.execute(
                """
                INSERT INTO violations 
                (source, frame_number, timestamp_seconds, class_name, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source, frame_number, timestamp, class_name, confidence,
                 x1, y1, x2, y2)
            )
            await db.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Failed to log violation: {e}")
            return -1
    
    async def get_violations(
        self,
        page: int = 1,
        page_size: int = 50,
        class_name: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Get violations with pagination and filtering.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            class_name: Filter by class
            source: Filter by source
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Tuple of (total_count, violations_list)
        """
        try:
            db = await get_db()
            
            # Build query
            conditions = []
            params = []
            
            if class_name:
                conditions.append("class_name = ?")
                params.append(class_name)
            
            if source:
                conditions.append("source LIKE ?")
                params.append(f"%{source}%")
            
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("created_at < ?")
                params.append(end_date.isoformat())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM violations WHERE {where_clause}"
            cursor = await db.execute(count_query, params)
            total = (await cursor.fetchone())[0]
            
            # Get paginated results
            offset = (page - 1) * page_size
            query = f"""
                SELECT id, source, frame_number, timestamp_seconds, class_name,
                       confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, created_at
                FROM violations 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            
            cursor = await db.execute(query, params + [page_size, offset])
            rows = await cursor.fetchall()
            
            # Format results
            violations = []
            for row in rows:
                violations.append({
                    "id": row[0],
                    "source": row[1],
                    "frame_number": row[2],
                    "timestamp_seconds": row[3],
                    "class_name": row[4],
                    "confidence": row[5],
                    "bbox_x1": row[6],
                    "bbox_y1": row[7],
                    "bbox_x2": row[8],
                    "bbox_y2": row[9],
                    "created_at": row[10]
                })
            
            return total, violations
            
        except Exception as e:
            logger.error(f"Failed to get violations: {e}")
            return 0, []
    
    async def get_violation_by_id(self, violation_id: int) -> Optional[Dict[str, Any]]:
        """Get a single violation by ID."""
        try:
            db = await get_db()
            
            cursor = await db.execute(
                """
                SELECT id, source, frame_number, timestamp_seconds, class_name,
                       confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, created_at
                FROM violations WHERE id = ?
                """,
                (violation_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "id": row[0],
                "source": row[1],
                "frame_number": row[2],
                "timestamp_seconds": row[3],
                "class_name": row[4],
                "confidence": row[5],
                "bbox_x1": row[6],
                "bbox_y1": row[7],
                "bbox_x2": row[8],
                "bbox_y2": row[9],
                "created_at": row[10]
            }
            
        except Exception as e:
            logger.error(f"Failed to get violation {violation_id}: {e}")
            return None
    
    async def delete_violation(self, violation_id: int) -> bool:
        """Delete a violation by ID."""
        try:
            db = await get_db()
            
            cursor = await db.execute(
                "DELETE FROM violations WHERE id = ?",
                (violation_id,)
            )
            await db.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Failed to delete violation {violation_id}: {e}")
            return False
    
    async def clear_all_violations(self) -> int:
        """Delete all violations."""
        try:
            db = await get_db()
            
            cursor = await db.execute("DELETE FROM violations")
            await db.commit()
            
            return cursor.rowcount
            
        except Exception as e:
            logger.error(f"Failed to clear violations: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get violation statistics.
        
        Returns:
            Dictionary with aggregated statistics
        """
        try:
            db = await get_db()
            
            # Total violations
            cursor = await db.execute("SELECT COUNT(*) FROM violations")
            total = (await cursor.fetchone())[0]
            
            # Violations by class
            cursor = await db.execute(
                """
                SELECT class_name, COUNT(*) as count 
                FROM violations 
                GROUP BY class_name 
                ORDER BY count DESC
                """
            )
            rows = await cursor.fetchall()
            by_class = {row[0]: row[1] for row in rows}
            
            # Last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM violations WHERE created_at >= ?",
                (yesterday,)
            )
            last_24h = (await cursor.fetchone())[0]
            
            # Last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM violations WHERE created_at >= ?",
                (week_ago,)
            )
            last_7d = (await cursor.fetchone())[0]
            
            # Most common violation
            most_common = list(by_class.keys())[0] if by_class else None
            
            return {
                "total_violations": total,
                "violations_by_class": by_class,
                "violations_last_24h": last_24h,
                "violations_last_7d": last_7d,
                "most_common_violation": most_common
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_violations": 0,
                "violations_by_class": {},
                "violations_last_24h": 0,
                "violations_last_7d": 0,
                "most_common_violation": None
            }
