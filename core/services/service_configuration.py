"""
Service Configuration
====================

Configuration et enregistrement des services dans le container DI.
Point d'entrée pour l'initialisation de l'architecture Clean.

Auteur: khalid
Date: 04/06/2025
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path

from core.services.dependency_injection import get_container
from core.services.config import get_config

# Domain interfaces
from core.domain.interfaces.repositories import (
    ReviewRepository, AnalyticsRepository, AlertRepository, CacheRepository
)
from core.domain.interfaces.services import (
    SentimentAnalysisService, CategoryClassificationService,
    RecommendationService, NotificationService
)

# Infrastructure implementations
from core.infrastructure.repositories import (
    SQLiteReviewRepository, SQLiteAnalyticsRepository,
    SQLiteAlertRepository, InMemoryCacheRepository
)
from core.infrastructure.services import (
    BERTSentimentAnalysisService, MLCategoryClassificationService,
    BusinessRecommendationService, EmailNotificationService,
    ModelConfig
)

# Use cases
from core.usecases.sentiment_analysis import (
    ProcessReviewUseCase, AnalyzeTrendsUseCase, GenerateInsightsUseCase
)

logger = logging.getLogger(__name__)


class ServiceConfiguration:
    """Configuration des services et repositories."""
    
    def __init__(self):
        """Initialise la configuration des services."""
        self.config = get_config()
        self.container = get_container()
        self._configured = False
    
    async def configure_services(self) -> None:
        """Configure et enregistre tous les services."""
        if self._configured:
            return
            
        logger.info("Configuring enterprise services...")
        
        try:
            # Configuration des repositories
            await self._configure_repositories()
            
            # Configuration des services métier
            await self._configure_business_services()
            
            # Configuration des use cases
            await self._configure_use_cases()
            
            # Initialisation des services critiques
            await self._initialize_critical_services()
            
            self._configured = True
            logger.info("Enterprise services configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure services: {e}")
            raise
    
    async def _configure_repositories(self) -> None:
        """Configure les repositories."""
        # Paths des bases de données
        data_dir = Path(self.config.data_directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Repository des avis
        review_db_path = data_dir / "reviews.db"
        self.container.register_singleton(
            ReviewRepository,
            SQLiteReviewRepository,
            lambda instance: self._init_repository(instance, str(review_db_path))
        )
        
        # Repository des analytics
        analytics_db_path = data_dir / "analytics.db"
        self.container.register_singleton(
            AnalyticsRepository,
            SQLiteAnalyticsRepository,
            lambda instance: self._init_repository(instance, str(analytics_db_path))
        )
        
        # Repository des alertes
        alerts_db_path = data_dir / "alerts.db"
        self.container.register_singleton(
            AlertRepository,
            SQLiteAlertRepository,
            lambda instance: self._init_repository(instance, str(alerts_db_path))
        )
        
        # Repository du cache
        self.container.register_singleton(
            CacheRepository,
            InMemoryCacheRepository
        )
        
        logger.debug("Repositories configured")
    
    async def _configure_business_services(self) -> None:
        """Configure les services métier."""
        # Configuration des modèles ML
        model_config = ModelConfig(
            sentiment_model_name=self.config.ml_models.sentiment_model,
            category_model_path=str(Path(self.config.data_directory) / "models" / "category_classifier.pkl"),
            device=self.config.ml_models.device,
            max_length=512,
            batch_size=self.config.ml_models.batch_size,
            cache_predictions=True
        )
        
        # Service d'analyse de sentiment
        self.container.register_singleton(
            SentimentAnalysisService,
            BERTSentimentAnalysisService,
            lambda instance: setattr(instance, 'config', model_config)
        )
        
        # Service de classification de catégories
        self.container.register_singleton(
            CategoryClassificationService,
            MLCategoryClassificationService,
            lambda instance: setattr(instance, 'config', model_config)
        )
        
        # Service de recommandations
        self.container.register_singleton(
            RecommendationService,
            BusinessRecommendationService
        )
        
        # Service de notifications
        smtp_config = {
            "host": self.config.notifications.smtp_host,
            "port": self.config.notifications.smtp_port,
            "username": self.config.notifications.smtp_username,
            "password": self.config.notifications.smtp_password,
            "use_tls": self.config.notifications.smtp_use_tls
        }
        
        self.container.register_singleton(
            NotificationService,
            EmailNotificationService,
            lambda instance: setattr(instance, 'smtp_config', smtp_config)
        )
        
        logger.debug("Business services configured")
    
    async def _configure_use_cases(self) -> None:
        """Configure les use cases métier."""
        # Use case de traitement des avis
        self.container.register_transient(
            ProcessReviewUseCase,
            ProcessReviewUseCase
        )
        
        # Use case d'analyse des tendances
        self.container.register_transient(
            AnalyzeTrendsUseCase,
            AnalyzeTrendsUseCase
        )
        
        # Use case de génération d'insights
        self.container.register_transient(
            GenerateInsightsUseCase,
            GenerateInsightsUseCase
        )
        
        logger.debug("Use cases configured")
    
    async def _initialize_critical_services(self) -> None:
        """Initialise les services critiques."""
        # Initialisation du service de sentiment (chargement des modèles)
        sentiment_service = self.container.resolve(SentimentAnalysisService)
        await sentiment_service.initialize()
        
        # Initialisation du service de classification
        category_service = self.container.resolve(CategoryClassificationService)
        await category_service.initialize()
        
        logger.debug("Critical services initialized")
    
    def _init_repository(self, instance: Any, db_path: str) -> None:
        """Initialise un repository avec son chemin de base."""
        if hasattr(instance, 'db_path'):
            instance.db_path = Path(db_path)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé des services."""
        health_status = {
            "configured": self._configured,
            "services": {},
            "repositories": {},
            "overall_status": "healthy"
        }
        
        try:
            # Vérification des repositories
            review_repo = self.container.resolve(ReviewRepository)
            health_status["repositories"]["reviews"] = "healthy"
            
            analytics_repo = self.container.resolve(AnalyticsRepository)
            health_status["repositories"]["analytics"] = "healthy"
            
            cache_repo = self.container.resolve(CacheRepository)
            health_status["repositories"]["cache"] = "healthy"
            
            # Vérification des services
            sentiment_service = self.container.resolve(SentimentAnalysisService)
            model_info = await sentiment_service.get_model_info()
            health_status["services"]["sentiment"] = {
                "status": "healthy" if model_info["initialized"] else "initializing",
                "model": model_info["model_name"]
            }
            
            category_service = self.container.resolve(CategoryClassificationService)
            category_info = await category_service.get_model_info()
            health_status["services"]["category"] = {
                "status": "healthy" if category_info["initialized"] else "initializing",
                "has_ml_model": category_info["has_ml_model"]
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def dispose(self) -> None:
        """Libère les ressources."""
        try:
            await self.container.dispose()
            self._configured = False
            logger.info("Services disposed successfully")
        except Exception as e:
            logger.error(f"Error disposing services: {e}")


# Instance globale de configuration
_service_config: ServiceConfiguration = None


async def get_service_configuration() -> ServiceConfiguration:
    """Récupère l'instance globale de configuration des services."""
    global _service_config
    
    if _service_config is None:
        _service_config = ServiceConfiguration()
        await _service_config.configure_services()
    
    return _service_config


async def initialize_enterprise_services() -> None:
    """Point d'entrée pour l'initialisation des services enterprise."""
    logger.info("Initializing enterprise architecture...")
    
    config = await get_service_configuration()
    health = await config.health_check()
    
    if health["overall_status"] == "healthy":
        logger.info("Enterprise services initialized successfully")
        logger.info(f"Services status: {health['services']}")
        logger.info(f"Repositories status: {health['repositories']}")
    else:
        logger.error("Enterprise services initialization failed")
        raise RuntimeError(f"Service initialization failed: {health.get('error', 'Unknown error')}")


async def shutdown_enterprise_services() -> None:
    """Arrêt propre des services enterprise."""
    global _service_config
    
    if _service_config:
        await _service_config.dispose()
        _service_config = None
    
    logger.info("Enterprise services shutdown completed")


# Context manager pour les services
class EnterpriseServicesContext:
    """Context manager pour la gestion du cycle de vie des services."""
    
    async def __aenter__(self):
        """Initialise les services."""
        await initialize_enterprise_services()
        return await get_service_configuration()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ferme les services."""
        await shutdown_enterprise_services()


def get_enterprise_services() -> EnterpriseServicesContext:
    """Crée un context manager pour les services enterprise."""
    return EnterpriseServicesContext()
