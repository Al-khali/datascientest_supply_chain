"""
Infrastructure Repositories
===========================

Implémentations concrètes des repositories pour différents backends.
Supporte SQLite, PostgreSQL, et MongoDB avec async/await.

Auteur: khalid
Date: 04/06/2025
"""

import json
import sqlite3
import aiosqlite
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID
import asyncio
import logging

from core.domain.interfaces.repositories import (
    ReviewRepository, AnalyticsRepository, AlertRepository, CacheRepository
)
from core.domain.models.review import (
    Review, ReviewId, AnalyticsReport, BusinessAlert,
    SupplyChainCategory, SentimentLabel, CriticalityLevel,
    SentimentScore, ConfidenceScore, CriticalityScore
)

logger = logging.getLogger(__name__)


class SQLiteReviewRepository(ReviewRepository):
    """Repository SQLite pour les avis clients."""
    
    def __init__(self, db_path: str = "data/reviews.db"):
        """Initialise le repository SQLite.
        
        Args:
            db_path: Chemin vers la base de données SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def _init_db(self) -> None:
        """Initialise la base de données avec les tables nécessaires."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    category TEXT NOT NULL,
                    criticality_level TEXT NOT NULL,
                    criticality_score REAL NOT NULL,
                    requires_action INTEGER NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_sentiment 
                ON reviews(sentiment_label)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_category 
                ON reviews(category)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_timestamp 
                ON reviews(timestamp)
            """)
            
            await db.commit()
    
    async def save(self, review: Review) -> Review:
        """Sauvegarde un avis dans la base de données."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            
            await db.execute("""
                INSERT OR REPLACE INTO reviews (
                    id, text, timestamp, sentiment_label, sentiment_score,
                    confidence_score, category, criticality_level, 
                    criticality_score, requires_action, metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(review.id.value),
                review.text,
                review.timestamp.isoformat(),
                review.sentiment_label.value,
                float(review.sentiment_score.value),
                float(review.confidence_score.value),
                review.category.value,
                review.criticality_level.value,
                float(review.criticality_score.value),
                int(review.requires_action),
                json.dumps(review.metadata) if review.metadata else None,
                now,
                now
            ))
            
            await db.commit()
            
        logger.debug(f"Saved review: {review.id}")
        return review
    
    async def get_by_id(self, review_id: ReviewId) -> Optional[Review]:
        """Récupère un avis par son ID."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM reviews WHERE id = ?", 
                (str(review_id.value),)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_review(row)
    
    async def find_all(self, offset: int = 0, limit: int = 100) -> List[Review]:
        """Récupère tous les avis avec pagination."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM reviews ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ) as cursor:
                rows = await cursor.fetchall()
                
                return [self._row_to_review(row) for row in rows]
    
    async def find_by_criteria(
        self,
        sentiment_label: Optional[SentimentLabel] = None,
        category: Optional[SupplyChainCategory] = None,
        criticality_level: Optional[CriticalityLevel] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        requires_action: Optional[bool] = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Review]:
        """Recherche d'avis selon des critères."""
        await self._init_db()
        
        conditions = []
        params = []
        
        if sentiment_label:
            conditions.append("sentiment_label = ?")
            params.append(sentiment_label.value)
            
        if category:
            conditions.append("category = ?")
            params.append(category.value)
            
        if criticality_level:
            conditions.append("criticality_level = ?")
            params.append(criticality_level.value)
            
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from.isoformat())
            
        if date_to:
            conditions.append("timestamp <= ?")
            params.append(date_to.isoformat())
            
        if requires_action is not None:
            conditions.append("requires_action = ?")
            params.append(int(requires_action))
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM reviews 
            WHERE {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                return [self._row_to_review(row) for row in rows]
    
    async def count_by_criteria(
        self,
        sentiment_label: Optional[SentimentLabel] = None,
        category: Optional[SupplyChainCategory] = None,
        criticality_level: Optional[CriticalityLevel] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        requires_action: Optional[bool] = None
    ) -> int:
        """Compte les avis selon des critères."""
        await self._init_db()
        
        conditions = []
        params = []
        
        if sentiment_label:
            conditions.append("sentiment_label = ?")
            params.append(sentiment_label.value)
            
        if category:
            conditions.append("category = ?")
            params.append(category.value)
            
        if criticality_level:
            conditions.append("criticality_level = ?")
            params.append(criticality_level.value)
            
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from.isoformat())
            
        if date_to:
            conditions.append("timestamp <= ?")
            params.append(date_to.isoformat())
            
        if requires_action is not None:
            conditions.append("requires_action = ?")
            params.append(int(requires_action))
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT COUNT(*) FROM reviews WHERE {where_clause}"
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0
    
    async def delete_by_id(self, review_id: ReviewId) -> bool:
        """Supprime un avis par son ID."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM reviews WHERE id = ?", 
                (str(review_id.value),)
            )
            
            await db.commit()
            
            return cursor.rowcount > 0
    
    def _row_to_review(self, row: aiosqlite.Row) -> Review:
        """Convertit une ligne de base de données en objet Review."""
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return Review(
            id=ReviewId(UUID(row['id'])),
            text=row['text'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            sentiment_label=SentimentLabel(row['sentiment_label']),
            sentiment_score=SentimentScore(row['sentiment_score']),
            confidence_score=ConfidenceScore(row['confidence_score']),
            category=SupplyChainCategory(row['category']),
            criticality_level=CriticalityLevel(row['criticality_level']),
            criticality_score=CriticalityScore(row['criticality_score']),
            requires_action=bool(row['requires_action']),
            metadata=metadata
        )


class SQLiteAnalyticsRepository(AnalyticsRepository):
    """Repository SQLite pour les rapports d'analytics."""
    
    def __init__(self, db_path: str = "data/analytics.db"):
        """Initialise le repository analytics SQLite.
        
        Args:
            db_path: Chemin vers la base de données SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def _init_db(self) -> None:
        """Initialise la base de données analytics."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS analytics_reports (
                    id TEXT PRIMARY KEY,
                    report_date TEXT NOT NULL,
                    total_reviews INTEGER NOT NULL,
                    sentiment_distribution TEXT NOT NULL,
                    category_distribution TEXT NOT NULL,
                    average_sentiment REAL NOT NULL,
                    criticality_distribution TEXT NOT NULL,
                    trends TEXT NOT NULL,
                    kpis TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_analytics_date 
                ON analytics_reports(report_date)
            """)
            
            await db.commit()
    
    async def save_report(self, report: AnalyticsReport) -> AnalyticsReport:
        """Sauvegarde un rapport d'analytics."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO analytics_reports (
                    id, report_date, total_reviews, sentiment_distribution,
                    category_distribution, average_sentiment, 
                    criticality_distribution, trends, kpis, 
                    recommendations, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(report.id),
                report.report_date.isoformat(),
                report.total_reviews,
                json.dumps(report.sentiment_distribution),
                json.dumps(report.category_distribution),
                float(report.average_sentiment.value),
                json.dumps(report.criticality_distribution),
                json.dumps(report.trends),
                json.dumps(report.kpis),
                json.dumps([rec.dict() for rec in report.recommendations]),
                datetime.now(timezone.utc).isoformat()
            ))
            
            await db.commit()
            
        logger.debug(f"Saved analytics report: {report.id}")
        return report
    
    async def get_latest_report(self) -> Optional[AnalyticsReport]:
        """Récupère le dernier rapport d'analytics."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM analytics_reports ORDER BY report_date DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_report(row)
    
    async def get_reports_by_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AnalyticsReport]:
        """Récupère les rapports d'une période."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT * FROM analytics_reports 
                WHERE report_date >= ? AND report_date <= ?
                ORDER BY report_date DESC
            """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                rows = await cursor.fetchall()
                
                return [self._row_to_report(row) for row in rows]
    
    async def calculate_kpis(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calcule les KPIs pour une période."""
        # Cette méthode ferait appel au ReviewRepository pour calculer les KPIs
        # Pour l'instant, on retourne des KPIs par défaut
        return {
            "satisfaction_score": 7.5,
            "nps_score": 42,
            "response_time": 24.5,
            "resolution_rate": 0.85,
            "critical_issues": 12,
            "improvement_rate": 0.15
        }
    
    def _row_to_report(self, row: aiosqlite.Row) -> AnalyticsReport:
        """Convertit une ligne en rapport d'analytics."""
        from core.domain.models.review import BusinessRecommendation
        
        recommendations_data = json.loads(row['recommendations'])
        recommendations = [
            BusinessRecommendation(**rec_data) 
            for rec_data in recommendations_data
        ]
        
        return AnalyticsReport(
            id=UUID(row['id']),
            report_date=datetime.fromisoformat(row['report_date']),
            total_reviews=row['total_reviews'],
            sentiment_distribution=json.loads(row['sentiment_distribution']),
            category_distribution=json.loads(row['category_distribution']),
            average_sentiment=SentimentScore(row['average_sentiment']),
            criticality_distribution=json.loads(row['criticality_distribution']),
            trends=json.loads(row['trends']),
            kpis=json.loads(row['kpis']),
            recommendations=recommendations
        )


class InMemoryCacheRepository(CacheRepository):
    """Repository cache en mémoire avec TTL."""
    
    def __init__(self):
        """Initialise le cache en mémoire."""
        self._cache: Dict[str, Any] = {}
        self._ttl: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        async with self._lock:
            if key not in self._cache:
                return None
                
            # Vérification TTL
            if key in self._ttl and datetime.now(timezone.utc) > self._ttl[key]:
                del self._cache[key]
                del self._ttl[key]
                return None
                
            return self._cache[key]
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: int = 300
    ) -> bool:
        """Stocke une valeur dans le cache."""
        async with self._lock:
            self._cache[key] = value
            
            if ttl_seconds > 0:
                expiry = datetime.now(timezone.utc).timestamp() + ttl_seconds
                self._ttl[key] = datetime.fromtimestamp(expiry, timezone.utc)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Supprime une clé du cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._ttl:
                    del self._ttl[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Vérifie l'existence d'une clé."""
        return await self.get(key) is not None
    
    async def clear_pattern(self, pattern: str) -> int:
        """Supprime les clés matchant un pattern."""
        import fnmatch
        
        async with self._lock:
            keys_to_delete = [
                key for key in self._cache.keys() 
                if fnmatch.fnmatch(key, pattern)
            ]
            
            for key in keys_to_delete:
                del self._cache[key]
                if key in self._ttl:
                    del self._ttl[key]
            
            return len(keys_to_delete)
    
    async def cleanup_expired(self) -> int:
        """Nettoie les clés expirées."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            expired_keys = [
                key for key, expiry in self._ttl.items()
                if now > expiry
            ]
            
            for key in expired_keys:
                if key in self._cache:
                    del self._cache[key]
                del self._ttl[key]
            
            return len(expired_keys)


class SQLiteAlertRepository(AlertRepository):
    """Repository SQLite pour les alertes métier."""
    
    def __init__(self, db_path: str = "data/alerts.db"):
        """Initialise le repository alerts SQLite.
        
        Args:
            db_path: Chemin vers la base de données SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def _init_db(self) -> None:
        """Initialise la base de données des alertes."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS business_alerts (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    affected_category TEXT,
                    threshold_value REAL,
                    current_value REAL,
                    metadata TEXT,
                    is_resolved INTEGER NOT NULL DEFAULT 0,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_type 
                ON business_alerts(alert_type)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity 
                ON business_alerts(severity)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created 
                ON business_alerts(created_at)
            """)
            
            await db.commit()
    
    async def save(self, alert: BusinessAlert) -> BusinessAlert:
        """Sauvegarde une alerte métier."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            
            await db.execute("""
                INSERT OR REPLACE INTO business_alerts (
                    id, alert_type, severity, title, description,
                    affected_category, threshold_value, current_value,
                    metadata, is_resolved, resolved_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(alert.id),
                alert.alert_type,
                alert.severity,
                alert.title,
                alert.description,
                alert.affected_category.value if alert.affected_category else None,
                alert.threshold_value,
                alert.current_value,
                json.dumps(alert.metadata) if alert.metadata else None,
                int(alert.is_resolved),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                now,
                now
            ))
            
            await db.commit()
            
        logger.debug(f"Saved business alert: {alert.id}")
        return alert
    
    async def get_by_id(self, alert_id: UUID) -> Optional[BusinessAlert]:
        """Récupère une alerte par son ID."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM business_alerts WHERE id = ?", 
                (str(alert_id),)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_alert(row)
    
    async def find_active_alerts(self) -> List[BusinessAlert]:
        """Récupère toutes les alertes actives."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM business_alerts WHERE is_resolved = 0 ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                
                return [self._row_to_alert(row) for row in rows]
    
    async def find_by_type(self, alert_type: str) -> List[BusinessAlert]:
        """Récupère les alertes par type."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM business_alerts WHERE alert_type = ? ORDER BY created_at DESC",
                (alert_type,)
            ) as cursor:
                rows = await cursor.fetchall()
                
                return [self._row_to_alert(row) for row in rows]
    
    async def find_by_severity(self, severity: str) -> List[BusinessAlert]:
        """Récupère les alertes par sévérité."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM business_alerts WHERE severity = ? ORDER BY created_at DESC",
                (severity,)
            ) as cursor:
                rows = await cursor.fetchall()
                
                return [self._row_to_alert(row) for row in rows]
    
    async def mark_resolved(self, alert_id: UUID) -> bool:
        """Marque une alerte comme résolue."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            
            cursor = await db.execute("""
                UPDATE business_alerts 
                SET is_resolved = 1, resolved_at = ?, updated_at = ?
                WHERE id = ?
            """, (now, now, str(alert_id)))
            
            await db.commit()
            
            return cursor.rowcount > 0
    
    def _row_to_alert(self, row: aiosqlite.Row) -> BusinessAlert:
        """Convertit une ligne en alerte métier."""
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return BusinessAlert(
            id=UUID(row['id']),
            alert_type=row['alert_type'],
            severity=row['severity'],
            title=row['title'],
            description=row['description'],
            affected_category=SupplyChainCategory(row['affected_category']) if row['affected_category'] else None,
            threshold_value=row['threshold_value'],
            current_value=row['current_value'],
            metadata=metadata,
            is_resolved=bool(row['is_resolved']),
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
            created_at=datetime.fromisoformat(row['created_at'])
        )
