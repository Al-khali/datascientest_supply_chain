"""
Domain Services Interfaces
==========================

Interfaces pour les services métier de l'analyse de sentiment.
Définissent les contrats pour l'intelligence artificielle.

Auteur: khalid
Date: 04/06/2025
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.domain.models.review import (
    Review, SentimentAnalysis, CategoryClassification, 
    EntityExtraction, BusinessRecommendation
)


class SentimentAnalysisService(ABC):
    """Interface pour le service d'analyse de sentiment."""
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyse le sentiment d'un texte.
        
        Args:
            text: Le texte à analyser
            
        Returns:
            Résultat de l'analyse de sentiment
            
        Raises:
            ValueError: Si le texte est invalide
            ServiceUnavailableError: Si le service est indisponible
        """
        pass
    
    @abstractmethod
    async def batch_analyze_sentiment(
        self, 
        texts: List[str]
    ) -> List[SentimentAnalysis]:
        """Analyse le sentiment de plusieurs textes en batch.
        
        Args:
            texts: Liste des textes à analyser
            
        Returns:
            Liste des résultats d'analyse
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle utilisé."""
        pass


class CategoryClassificationService(ABC):
    """Interface pour la classification par catégorie."""
    
    @abstractmethod
    async def classify_category(self, text: str) -> CategoryClassification:
        """Classifie un texte par catégorie supply chain.
        
        Args:
            text: Le texte à classifier
            
        Returns:
            Résultat de la classification
        """
        pass
    
    @abstractmethod
    async def batch_classify(
        self, 
        texts: List[str]
    ) -> List[CategoryClassification]:
        """Classification en batch."""
        pass


class EntityExtractionService(ABC):
    """Interface pour l'extraction d'entités."""
    
    @abstractmethod
    async def extract_entities(self, text: str) -> List[EntityExtraction]:
        """Extrait les entités nommées d'un texte.
        
        Args:
            text: Le texte à analyser
            
        Returns:
            Liste des entités extraites
        """
        pass
    
    @abstractmethod
    async def extract_business_entities(
        self, 
        text: str
    ) -> Dict[str, List[str]]:
        """Extrait les entités métier spécifiques à la supply chain.
        
        Returns:
            Dictionnaire des entités par type (brands, products, etc.)
        """
        pass


class RecommendationService(ABC):
    """Interface pour la génération de recommandations."""
    
    @abstractmethod
    async def generate_recommendations(
        self, 
        review: Review
    ) -> List[BusinessRecommendation]:
        """Génère des recommandations business pour un avis.
        
        Args:
            review: L'avis analysé
            
        Returns:
            Liste des recommandations actionnables
        """
        pass
    
    @abstractmethod
    async def get_category_recommendations(
        self, 
        category: str,
        sentiment_score: float
    ) -> List[str]:
        """Génère des recommandations par catégorie."""
        pass


class AnomalyDetectionService(ABC):
    """Interface pour la détection d'anomalies."""
    
    @abstractmethod
    async def detect_anomalies(
        self, 
        reviews: List[Review]
    ) -> List[Review]:
        """Détecte les anomalies dans une liste d'avis.
        
        Args:
            reviews: Liste des avis à analyser
            
        Returns:
            Liste des avis marqués comme anomalies
        """
        pass
    
    @abstractmethod
    async def is_spam(self, review: Review) -> bool:
        """Détermine si un avis est du spam."""
        pass
    
    @abstractmethod
    async def calculate_quality_score(self, review: Review) -> float:
        """Calcule un score de qualité pour l'avis."""
        pass


class TopicModelingService(ABC):
    """Interface pour la modélisation de topics."""
    
    @abstractmethod
    async def extract_topics(
        self, 
        texts: List[str], 
        n_topics: int = 8
    ) -> Dict[str, Any]:
        """Extrait les topics d'une collection de textes.
        
        Args:
            texts: Liste des textes
            n_topics: Nombre de topics à extraire
            
        Returns:
            Dictionnaire avec les topics et leurs mots-clés
        """
        pass
    
    @abstractmethod
    async def get_trending_topics(
        self, 
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Récupère les topics tendance."""
        pass


class MLModelService(ABC):
    """Interface pour la gestion des modèles ML."""
    
    @abstractmethod
    async def load_model(self, model_name: str, version: str) -> bool:
        """Charge un modèle spécifique."""
        pass
    
    @abstractmethod
    async def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Récupère les métriques d'un modèle."""
        pass
    
    @abstractmethod
    async def retrain_model(
        self, 
        model_name: str, 
        training_data: List[Dict]
    ) -> bool:
        """Relance l'entraînement d'un modèle."""
        pass
    
    @abstractmethod
    async def validate_model_drift(self, model_name: str) -> Dict[str, Any]:
        """Valide la dérive d'un modèle."""
        pass
