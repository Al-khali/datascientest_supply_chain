"""
Domain Interfaces - Contrats de repositories
==========================================

Interfaces définissant les contrats pour l'accès aux données.
Suivent le principe d'inversion de dépendance (SOLID).

Auteur: khalid
Date: 04/06/2025
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from core.domain.models.review import (
    Review, ReviewId, AnalyticsReport, BusinessAlert,
    SupplyChainCategory, SentimentLabel, CriticalityLevel
)


class ReviewRepository(ABC):
    """Interface pour la persistance des avis."""
    
    @abstractmethod
    async def save(self, review: Review) -> Review:
        """Sauvegarde un avis."""
        pass
    
    @abstractmethod
    async def get_by_id(self, review_id: ReviewId) -> Optional[Review]:
        """Récupère un avis par ID."""
        pass
    
    @abstractmethod
    async def find_all(
        self, 
        offset: int = 0, 
        limit: int = 100
    ) -> List[Review]:
        """Récupère tous les avis avec pagination."""
        pass
    
    @abstractmethod
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
        """Recherche d'avis par critères."""
        pass
    
    @abstractmethod
    async def count_total(self) -> int:
        """Compte le nombre total d'avis."""
        pass
    
    @abstractmethod
    async def delete(self, review_id: ReviewId) -> bool:
        """Supprime un avis."""
        pass


class AnalyticsRepository(ABC):
    """Interface pour la persistance des analytics."""
    
    @abstractmethod
    async def save_report(self, report: AnalyticsReport) -> AnalyticsReport:
        """Sauvegarde un rapport d'analytics."""
        pass
    
    @abstractmethod
    async def get_latest_report(self) -> Optional[AnalyticsReport]:
        """Récupère le dernier rapport."""
        pass
    
    @abstractmethod
    async def get_reports_by_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AnalyticsReport]:
        """Récupère les rapports d'une période."""
        pass
    
    @abstractmethod
    async def calculate_kpis(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calcule les KPIs pour une période."""
        pass


class AlertRepository(ABC):
    """Interface pour la persistance des alertes."""
    
    @abstractmethod
    async def save(self, alert: BusinessAlert) -> BusinessAlert:
        """Sauvegarde une alerte."""
        pass
    
    @abstractmethod
    async def get_active_alerts(self) -> List[BusinessAlert]:
        """Récupère les alertes actives."""
        pass
    
    @abstractmethod
    async def get_by_category(
        self, 
        category: SupplyChainCategory
    ) -> List[BusinessAlert]:
        """Récupère les alertes par catégorie."""
        pass
    
    @abstractmethod
    async def acknowledge_alert(
        self, 
        alert_id: UUID, 
        user_id: str
    ) -> bool:
        """Acquitte une alerte."""
        pass
    
    @abstractmethod
    async def cleanup_old_alerts(self, days_old: int = 30) -> int:
        """Nettoie les anciennes alertes."""
        pass


class CacheRepository(ABC):
    """Interface pour le cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: int = 300
    ) -> bool:
        """Stocke une valeur dans le cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Supprime une clé du cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Vérifie l'existence d'une clé."""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Supprime les clés matchant un pattern."""
        pass
