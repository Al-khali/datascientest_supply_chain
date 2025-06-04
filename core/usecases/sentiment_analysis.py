"""
Use Cases - Analyse de Sentiment
==============================

Cas d'usage métier pour l'analyse complète des avis clients.
Orchestre les services et applique la logique métier.

Auteur: khalid  
Date: 04/06/2025
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from core.domain.models.review import (
    Review, ReviewId, SentimentAnalysis, CategoryClassification,
    EntityExtraction, CriticalityLevel, BusinessAlert, SupplyChainCategory
)
from core.domain.interfaces.repositories import (
    ReviewRepository, AlertRepository, CacheRepository
)
from core.domain.interfaces.services import (
    SentimentAnalysisService, CategoryClassificationService,
    EntityExtractionService, RecommendationService, AnomalyDetectionService
)

logger = logging.getLogger(__name__)


class ProcessReviewUseCase:
    """Cas d'usage pour le traitement complet d'un avis."""
    
    def __init__(
        self,
        review_repository: ReviewRepository,
        alert_repository: AlertRepository,
        cache_repository: CacheRepository,
        sentiment_service: SentimentAnalysisService,
        category_service: CategoryClassificationService,
        entity_service: EntityExtractionService,
        recommendation_service: RecommendationService,
        anomaly_service: AnomalyDetectionService
    ):
        self.review_repository = review_repository
        self.alert_repository = alert_repository
        self.cache_repository = cache_repository
        self.sentiment_service = sentiment_service
        self.category_service = category_service
        self.entity_service = entity_service
        self.recommendation_service = recommendation_service
        self.anomaly_service = anomaly_service
    
    async def execute(self, review: Review) -> Review:
        """Traite complètement un avis client.
        
        Args:
            review: L'avis à traiter
            
        Returns:
            L'avis enrichi avec toutes les analyses
            
        Raises:
            ValueError: Si l'avis est invalide
            ProcessingError: Si une erreur survient pendant le traitement
        """
        try:
            logger.info(f"Début traitement avis {review.id.value}")
            
            # 1. Vérification cache
            cache_key = f"review_processed:{review.id.value}"
            cached = await self.cache_repository.get(cache_key)
            if cached:
                logger.info(f"Avis {review.id.value} trouvé en cache")
                return cached
            
            # 2. Analyse de sentiment
            sentiment_analysis = await self.sentiment_service.analyze_sentiment(
                review.content
            )
            review.add_sentiment_analysis(sentiment_analysis)
            
            # 3. Classification catégorie
            category_classification = await self.category_service.classify_category(
                review.content
            )
            review.add_category_classification(category_classification)
            
            # 4. Extraction d'entités
            entities = await self.entity_service.extract_entities(review.content)
            review.entities = entities
            
            # 5. Calcul criticité
            criticality_score = review.calculate_criticality()
            
            # 6. Détection anomalies
            is_anomaly = await self.anomaly_service.is_spam(review)
            review.is_anomaly = is_anomaly
            
            # 7. Génération recommandations
            if not is_anomaly:
                recommendations = await self.recommendation_service.generate_recommendations(
                    review
                )
                review.recommendations = recommendations
            
            # 8. Création d'alertes si nécessaire
            await self._create_alerts_if_needed(review)
            
            # 9. Sauvegarde
            saved_review = await self.review_repository.save(review)
            saved_review.mark_as_processed()
            
            # 10. Mise en cache
            await self.cache_repository.set(
                cache_key, 
                saved_review, 
                ttl_seconds=3600  # 1 heure
            )
            
            logger.info(f"Avis {review.id.value} traité avec succès")
            return saved_review
            
        except Exception as e:
            logger.error(f"Erreur traitement avis {review.id.value}: {str(e)}")
            raise ProcessingError(f"Failed to process review: {str(e)}")
    
    async def batch_process(self, reviews: List[Review]) -> List[Review]:
        """Traite plusieurs avis en parallèle.
        
        Args:
            reviews: Liste des avis à traiter
            
        Returns:
            Liste des avis traités
        """
        logger.info(f"Début traitement batch de {len(reviews)} avis")
        
        # Traitement par chunks pour éviter la surcharge
        chunk_size = 10
        processed_reviews = []
        
        for i in range(0, len(reviews), chunk_size):
            chunk = reviews[i:i + chunk_size]
            
            # Traitement concurrent du chunk
            tasks = [self.execute(review) for review in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des erreurs
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"Erreur traitement batch: {result}")
                else:
                    processed_reviews.append(result)
        
        logger.info(f"Traitement batch terminé: {len(processed_reviews)} avis traités")
        return processed_reviews
    
    async def _create_alerts_if_needed(self, review: Review) -> None:
        """Crée des alertes métier si nécessaire."""
        if not review.criticality_score:
            return
        
        criticality_level = review.get_criticality_level()
        
        # Alerte pour criticité élevée
        if criticality_level in [CriticalityLevel.HIGH, CriticalityLevel.CRITICAL]:
            alert = BusinessAlert(
                level=criticality_level,
                category=review.category_classification.category,
                title=f"Avis critique détecté - {review.category_classification.category.value}",
                description=f"Avis avec score de criticité {review.criticality_score.value:.1f}",
                recommended_actions=[rec.action for rec in review.recommendations[:3]]
            )
            
            await self.alert_repository.save(alert)
            logger.info(f"Alerte créée pour avis {review.id.value}")


class AnalyzeTrendsUseCase:
    """Cas d'usage pour l'analyse des tendances."""
    
    def __init__(
        self,
        review_repository: ReviewRepository,
        cache_repository: CacheRepository
    ):
        self.review_repository = review_repository
        self.cache_repository = cache_repository
    
    async def execute(
        self, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Analyse les tendances sur une période.
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Dictionnaire avec les tendances identifiées
        """
        cache_key = f"trends_analysis:{days}d"
        cached = await self.cache_repository.get(cache_key)
        if cached:
            return cached
        
        from datetime import timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Récupération des avis de la période
        reviews = await self.review_repository.find_by_criteria(
            date_from=start_date,
            date_to=end_date,
            limit=10000
        )
        
        if not reviews:
            return {"error": "No reviews found for the period"}
        
        # Calcul des tendances
        trends = await self._calculate_trends(reviews, days)
        
        # Mise en cache
        await self.cache_repository.set(
            cache_key, 
            trends, 
            ttl_seconds=1800  # 30 minutes
        )
        
        return trends
    
    async def _calculate_trends(
        self, 
        reviews: List[Review], 
        days: int
    ) -> Dict[str, Any]:
        """Calcule les tendances à partir des avis."""
        
        # Tendances de sentiment
        sentiments = [
            r.sentiment_analysis.score.value 
            for r in reviews 
            if r.sentiment_analysis
        ]
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Distribution par catégorie
        categories = {}
        for review in reviews:
            if review.category_classification:
                cat = review.category_classification.category.value
                categories[cat] = categories.get(cat, 0) + 1
        
        # Tendance volume
        volume_trend = len(reviews) / days if days > 0 else 0
        
        # Top mots-clés négatifs
        negative_reviews = [
            r for r in reviews 
            if r.sentiment_analysis and r.sentiment_analysis.score.value < -0.3
        ]
        
        return {
            "period_days": days,
            "total_reviews": len(reviews),
            "average_sentiment": round(avg_sentiment, 3),
            "volume_per_day": round(volume_trend, 1),
            "category_distribution": categories,
            "negative_reviews_count": len(negative_reviews),
            "negative_percentage": round(
                len(negative_reviews) / len(reviews) * 100, 1
            ) if reviews else 0,
            "requires_attention": len(negative_reviews) > len(reviews) * 0.3
        }


class GenerateInsightsUseCase:
    """Cas d'usage pour la génération d'insights business."""
    
    def __init__(
        self,
        review_repository: ReviewRepository,
        recommendation_service: RecommendationService
    ):
        self.review_repository = review_repository
        self.recommendation_service = recommendation_service
    
    async def execute(self) -> Dict[str, Any]:
        """Génère des insights business actionnables."""
        
        # Top avis critiques non traités
        critical_reviews = await self.review_repository.find_by_criteria(
            criticality_level=CriticalityLevel.CRITICAL,
            requires_action=True,
            limit=50
        )
        
        # Insights par catégorie
        category_insights = {}
        for category in SupplyChainCategory:
            category_reviews = await self.review_repository.find_by_criteria(
                category=category,
                limit=1000
            )
            
            if category_reviews:
                negative_count = sum(
                    1 for r in category_reviews 
                    if r.sentiment_analysis and r.sentiment_analysis.score.value < -0.2
                )
                
                category_insights[category.value] = {
                    "total_reviews": len(category_reviews),
                    "negative_reviews": negative_count,
                    "negative_percentage": round(
                        negative_count / len(category_reviews) * 100, 1
                    ),
                    "needs_attention": negative_count > len(category_reviews) * 0.25
                }
        
        return {
            "critical_reviews_pending": len(critical_reviews),
            "category_insights": category_insights,
            "top_issues": await self._extract_top_issues(critical_reviews),
            "recommended_actions": await self._get_priority_actions(critical_reviews)
        }
    
    async def _extract_top_issues(self, reviews: List[Review]) -> List[str]:
        """Extrait les problèmes principaux."""
        issues = []
        for review in reviews[:10]:  # Top 10
            if review.recommendations:
                issues.extend([rec.action for rec in review.recommendations[:2]])
        
        # Déduplication et tri par fréquence
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True)[:5]
    
    async def _get_priority_actions(self, reviews: List[Review]) -> List[Dict[str, Any]]:
        """Récupère les actions prioritaires."""
        actions = []
        
        for review in reviews[:5]:  # Top 5 critiques
            if review.recommendations:
                for rec in review.recommendations[:1]:  # Action principale
                    actions.append({
                        "action": rec.action,
                        "priority": rec.priority.value,
                        "category": review.category_classification.category.value if review.category_classification else "unknown",
                        "estimated_impact": rec.estimated_impact,
                        "review_id": review.id.value
                    })
        
        return actions


class ProcessingError(Exception):
    """Exception pour les erreurs de traitement."""
    pass
