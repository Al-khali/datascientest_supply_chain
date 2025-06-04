"""
Application Layer
================

Couche application pour orchestrer les use cases et exposer les APIs.
Point d'entrée pour les interfaces utilisateur (API REST, Dashboard).

Auteur: khalid
Date: 04/06/2025
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging

from core.services.dependency_injection import inject
from core.services.service_configuration import get_service_configuration

# Domain models
from core.domain.models.review import (
    Review, ReviewId, SentimentLabel, SupplyChainCategory,
    CriticalityLevel, AnalyticsReport, BusinessAlert, BusinessRecommendation
)

# Use cases
from core.usecases.sentiment_analysis import (
    ProcessReviewUseCase, AnalyzeTrendsUseCase, GenerateInsightsUseCase
)

# Repositories
from core.domain.interfaces.repositories import (
    ReviewRepository, AnalyticsRepository, AlertRepository
)

logger = logging.getLogger(__name__)


class ReviewAnalysisApplicationService:
    """Service application pour l'analyse des avis clients."""
    
    def __init__(self):
        """Initialise le service application."""
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialise le service avec les dépendances."""
        if not self._initialized:
            self.config = await get_service_configuration()
            self._initialized = True
    
    async def process_review(self, text: str, metadata: Dict[str, Any] = None) -> Review:
        """Traite un nouvel avis client.
        
        Args:
            text: Texte de l'avis
            metadata: Métadonnées additionnelles
            
        Returns:
            Avis traité avec analyse de sentiment
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Récupération du use case
            process_use_case = inject(ProcessReviewUseCase)
            
            # Traitement de l'avis
            review = await process_use_case.execute(text, metadata or {})
            
            logger.info(f"Processed review: {review.id} - {review.sentiment_label.value}")
            return review
            
        except Exception as e:
            logger.error(f"Error processing review: {e}")
            raise
    
    async def process_reviews_batch(
        self, 
        reviews_data: List[Dict[str, Any]]
    ) -> List[Review]:
        """Traite un lot d'avis de manière optimisée.
        
        Args:
            reviews_data: Liste des données d'avis
            
        Returns:
            Liste des avis traités
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            process_use_case = inject(ProcessReviewUseCase)
            
            # Traitement en parallèle avec limitation
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
            
            async def process_single(review_data):
                async with semaphore:
                    return await process_use_case.execute(
                        review_data["text"],
                        review_data.get("metadata", {})
                    )
            
            tasks = [process_single(data) for data in reviews_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des erreurs
            successful_reviews = [
                result for result in results 
                if isinstance(result, Review)
            ]
            
            errors = [
                result for result in results 
                if isinstance(result, Exception)
            ]
            
            if errors:
                logger.warning(f"Batch processing had {len(errors)} errors")
            
            logger.info(f"Batch processed: {len(successful_reviews)}/{len(reviews_data)} reviews")
            return successful_reviews
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    async def get_reviews(
        self,
        sentiment_filter: Optional[SentimentLabel] = None,
        category_filter: Optional[SupplyChainCategory] = None,
        criticality_filter: Optional[CriticalityLevel] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        requires_action: Optional[bool] = None,
        offset: int = 0,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Récupère les avis avec filtres et pagination.
        
        Returns:
            Dictionnaire avec reviews, total_count, et metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            review_repo = inject(ReviewRepository)
            
            # Récupération des avis
            reviews = await review_repo.find_by_criteria(
                sentiment_label=sentiment_filter,
                category=category_filter,
                criticality_level=criticality_filter,
                date_from=date_from,
                date_to=date_to,
                requires_action=requires_action,
                offset=offset,
                limit=limit
            )
            
            # Comptage total
            total_count = await review_repo.count_by_criteria(
                sentiment_label=sentiment_filter,
                category=category_filter,
                criticality_level=criticality_filter,
                date_from=date_from,
                date_to=date_to,
                requires_action=requires_action
            )
            
            return {
                "reviews": reviews,
                "total_count": total_count,
                "offset": offset,
                "limit": limit,
                "has_more": (offset + len(reviews)) < total_count
            }
            
        except Exception as e:
            logger.error(f"Error retrieving reviews: {e}")
            raise
    
    async def get_review_by_id(self, review_id: str) -> Optional[Review]:
        """Récupère un avis par son ID."""
        if not self._initialized:
            await self.initialize()
        
        try:
            review_repo = inject(ReviewRepository)
            return await review_repo.get_by_id(ReviewId(UUID(review_id)))
            
        except Exception as e:
            logger.error(f"Error retrieving review {review_id}: {e}")
            return None


class AnalyticsApplicationService:
    """Service application pour les analytics et reporting."""
    
    def __init__(self):
        """Initialise le service analytics."""
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialise le service avec les dépendances."""
        if not self._initialized:
            self.config = await get_service_configuration()
            self._initialized = True
    
    async def generate_analytics_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AnalyticsReport:
        """Génère un rapport d'analytics complet.
        
        Args:
            start_date: Date de début (défaut: 30 jours)
            end_date: Date de fin (défaut: maintenant)
            
        Returns:
            Rapport d'analytics complet
        """
        if not self._initialized:
            await self.initialize()
        
        # Dates par défaut
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        try:
            # Use case d'analyse des tendances
            trends_use_case = inject(AnalyzeTrendsUseCase)
            
            # Génération du rapport
            report = await trends_use_case.execute(start_date, end_date)
            
            logger.info(f"Generated analytics report for period {start_date} to {end_date}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            raise
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Récupère les données pour le dashboard exécutif.
        
        Returns:
            Données formatées pour le dashboard
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Rapport des 7 derniers jours
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)
            
            report = await self.generate_analytics_report(start_date, end_date)
            
            # Alertes actives
            alert_repo = inject(AlertRepository)
            active_alerts = await alert_repo.find_active_alerts()
            
            # Formatage pour le dashboard
            dashboard_data = {
                "overview": {
                    "total_reviews": report.total_reviews,
                    "average_sentiment": float(report.average_sentiment.value),
                    "satisfaction_score": report.kpis.get("satisfaction_score", 0),
                    "nps_score": report.kpis.get("nps_score", 0)
                },
                "sentiment_distribution": report.sentiment_distribution,
                "category_distribution": report.category_distribution,
                "criticality_distribution": report.criticality_distribution,
                "trends": report.trends,
                "active_alerts": [
                    {
                        "id": str(alert.id),
                        "title": alert.title,
                        "severity": alert.severity,
                        "category": alert.affected_category.value if alert.affected_category else None,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in active_alerts[:5]  # Top 5 alertes
                ],
                "recommendations": [
                    {
                        "title": rec.title,
                        "priority": rec.priority,
                        "category": rec.category.value if rec.category else None,
                        "estimated_impact": rec.estimated_impact
                    }
                    for rec in report.recommendations[:3]  # Top 3 recommandations
                ],
                "kpis": report.kpis,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            raise
    
    async def get_trends_analysis(
        self,
        category: Optional[SupplyChainCategory] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyse des tendances par catégorie.
        
        Args:
            category: Catégorie à analyser (toutes si None)
            days: Nombre de jours à analyser
            
        Returns:
            Analyse des tendances
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            trends_use_case = inject(AnalyzeTrendsUseCase)
            
            # Analyse spécifique à la catégorie si fournie
            if category:
                # Filtrage par catégorie (à implémenter dans le use case)
                analysis = await trends_use_case.execute_for_category(
                    start_date, end_date, category
                )
            else:
                # Analyse globale
                report = await trends_use_case.execute(start_date, end_date)
                analysis = {
                    "trends": report.trends,
                    "sentiment_evolution": report.sentiment_distribution,
                    "category_performance": report.category_distribution
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            raise


class InsightsApplicationService:
    """Service application pour la génération d'insights métier."""
    
    def __init__(self):
        """Initialise le service insights."""
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialise le service avec les dépendances."""
        if not self._initialized:
            self.config = await get_service_configuration()
            self._initialized = True
    
    async def generate_business_insights(
        self,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Génère des insights métier avancés.
        
        Args:
            analysis_type: Type d'analyse (comprehensive, quick, deep)
            
        Returns:
            Insights métier structurés
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            insights_use_case = inject(GenerateInsightsUseCase)
            
            # Génération des insights
            insights = await insights_use_case.execute(analysis_type)
            
            return {
                "insights": insights,
                "analysis_type": analysis_type,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise
    
    async def get_recommendations(
        self,
        priority_filter: Optional[str] = None,
        category_filter: Optional[SupplyChainCategory] = None
    ) -> List[BusinessRecommendation]:
        """Récupère les recommandations métier.
        
        Args:
            priority_filter: Filtre par priorité (HIGH, MEDIUM, LOW)
            category_filter: Filtre par catégorie
            
        Returns:
            Liste des recommandations filtrées
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Génération d'un rapport récent pour obtenir les recommandations
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)
            
            trends_use_case = inject(AnalyzeTrendsUseCase)
            report = await trends_use_case.execute(start_date, end_date)
            
            recommendations = report.recommendations
            
            # Filtrage par priorité
            if priority_filter:
                recommendations = [
                    rec for rec in recommendations 
                    if rec.priority == priority_filter.upper()
                ]
            
            # Filtrage par catégorie
            if category_filter:
                recommendations = [
                    rec for rec in recommendations 
                    if rec.category == category_filter
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error retrieving recommendations: {e}")
            raise
    
    async def get_alerts_summary(self) -> Dict[str, Any]:
        """Récupère un résumé des alertes actives.
        
        Returns:
            Résumé des alertes avec statistiques
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            alert_repo = inject(AlertRepository)
            
            # Alertes actives
            active_alerts = await alert_repo.find_active_alerts()
            
            # Statistiques par sévérité
            severity_counts = {}
            for alert in active_alerts:
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            
            # Statistiques par catégorie
            category_counts = {}
            for alert in active_alerts:
                if alert.affected_category:
                    cat = alert.affected_category.value
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            return {
                "total_active_alerts": len(active_alerts),
                "severity_distribution": severity_counts,
                "category_distribution": category_counts,
                "recent_alerts": [
                    {
                        "id": str(alert.id),
                        "title": alert.title,
                        "severity": alert.severity,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in active_alerts[:10]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error retrieving alerts summary: {e}")
            raise


# Services d'application globaux
_review_service: Optional[ReviewAnalysisApplicationService] = None
_analytics_service: Optional[AnalyticsApplicationService] = None
_insights_service: Optional[InsightsApplicationService] = None


async def get_review_service() -> ReviewAnalysisApplicationService:
    """Récupère le service d'analyse des avis."""
    global _review_service
    if _review_service is None:
        _review_service = ReviewAnalysisApplicationService()
        await _review_service.initialize()
    return _review_service


async def get_analytics_service() -> AnalyticsApplicationService:
    """Récupère le service d'analytics."""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsApplicationService()
        await _analytics_service.initialize()
    return _analytics_service


async def get_insights_service() -> InsightsApplicationService:
    """Récupère le service d'insights."""
    global _insights_service
    if _insights_service is None:
        _insights_service = InsightsApplicationService()
        await _insights_service.initialize()
    return _insights_service
