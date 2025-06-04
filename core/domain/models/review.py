"""
Domain Models - Entités métier centrales
======================================

Modèles de domaine pour l'analyse de satisfaction client supply chain.
Représentent les concepts métier fondamentaux sans dépendances techniques.

Auteur: khalid
Date: 04/06/2025
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Fonction helper pour obtenir l'heure UTC actuelle."""
    return datetime.now(timezone.utc)


class SentimentLabel(str, Enum):
    """Labels de sentiment standardisés."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SupplyChainCategory(str, Enum):
    """Catégories supply chain métier."""
    DELIVERY_LOGISTICS = "delivery_logistics"
    PRODUCT_QUALITY = "product_quality"
    CUSTOMER_SERVICE = "customer_service"
    PAYMENT_TRANSACTION = "payment_transaction"
    DIGITAL_INTERFACE = "digital_interface"
    OTHER = "other"


class ReviewSource(str, Enum):
    """Sources des avis clients."""
    TRUSTPILOT = "trustpilot"
    GOOGLE_REVIEWS = "google_reviews"
    AMAZON = "amazon"
    INTERNAL_SURVEY = "internal_survey"
    SOCIAL_MEDIA = "social_media"


class CriticalityLevel(str, Enum):
    """Niveaux de criticité métier."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ReviewId:
    """Value Object pour l'identifiant d'avis."""
    value: str
    
    def __post_init__(self) -> None:
        if not self.value or len(self.value.strip()) == 0:
            raise ValueError("Review ID cannot be empty")


@dataclass(frozen=True)
class SentimentScore:
    """Value Object pour le score de sentiment."""
    value: float
    
    def __post_init__(self) -> None:
        if not -1.0 <= self.value <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")


@dataclass(frozen=True)
class ConfidenceScore:
    """Value Object pour le score de confiance."""
    value: float
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class CriticalityScore:
    """Value Object pour le score de criticité."""
    value: float
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 100.0:
            raise ValueError("Criticality score must be between 0.0 and 100.0")


@dataclass
class EntityExtraction:
    """Entité extraite du texte."""
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int


@dataclass
class SentimentAnalysis:
    """Résultat d'analyse de sentiment."""
    score: SentimentScore
    label: SentimentLabel
    confidence: ConfidenceScore
    emotions: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_version: str = "unknown"


@dataclass
class CategoryClassification:
    """Classification par catégorie supply chain."""
    category: SupplyChainCategory
    confidence: ConfidenceScore
    keywords_found: List[str] = field(default_factory=list)
    business_impact: str = "low"


@dataclass
class ReviewMetadata:
    """Métadonnées d'un avis client."""
    source: ReviewSource
    collected_at: datetime
    rating: Optional[int] = None
    author_id: Optional[str] = None
    verified_purchase: bool = False
    location: Optional[str] = None
    product_category: Optional[str] = None
    
    def __post_init__(self) -> None:
        if self.rating is not None and not 1 <= self.rating <= 5:
            raise ValueError("Rating must be between 1 and 5")


@dataclass
class BusinessRecommendation:
    """Recommandation business actionnaire."""
    action: str
    priority: CriticalityLevel
    estimated_impact: str
    timeframe: str
    responsible_team: str
    estimated_cost: Optional[float] = None


@dataclass
class Review:
    """Entité principale - Avis client."""
    id: ReviewId
    content: str
    metadata: ReviewMetadata
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    
    # Résultats d'analyse (optionnels)
    sentiment_analysis: Optional[SentimentAnalysis] = None
    category_classification: Optional[CategoryClassification] = None
    entities: List[EntityExtraction] = field(default_factory=list)
    criticality_score: Optional[CriticalityScore] = None
    recommendations: List[BusinessRecommendation] = field(default_factory=list)
    
    # Flags métier
    is_processed: bool = False
    is_anomaly: bool = False
    requires_immediate_action: bool = False
    
    def __post_init__(self) -> None:
        if not self.content or len(self.content.strip()) == 0:
            raise ValueError("Review content cannot be empty")
        
        if len(self.content) > 10000:
            raise ValueError("Review content too long (max 10000 characters)")
    
    def mark_as_processed(self) -> None:
        """Marque l'avis comme traité."""
        self.is_processed = True
        self.updated_at = datetime.now(timezone.utc)
    
    def add_sentiment_analysis(self, analysis: SentimentAnalysis) -> None:
        """Ajoute l'analyse de sentiment."""
        self.sentiment_analysis = analysis
        self.updated_at = datetime.now(timezone.utc)
    
    def add_category_classification(self, classification: CategoryClassification) -> None:
        """Ajoute la classification par catégorie."""
        self.category_classification = classification
        self.updated_at = datetime.now(timezone.utc)
    
    def calculate_criticality(self) -> CriticalityScore:
        """Calcule le score de criticité métier."""
        if not self.sentiment_analysis or not self.category_classification:
            return CriticalityScore(0.0)
        
        base_score = 0.0
        
        # Impact du sentiment
        sentiment_value = self.sentiment_analysis.score.value
        if sentiment_value <= -0.5:
            base_score += 40.0
        elif sentiment_value <= -0.2:
            base_score += 25.0
        elif sentiment_value <= 0.2:
            base_score += 10.0
        
        # Impact de la catégorie
        category_weights = {
            SupplyChainCategory.DELIVERY_LOGISTICS: 25.0,
            SupplyChainCategory.CUSTOMER_SERVICE: 20.0,
            SupplyChainCategory.PRODUCT_QUALITY: 18.0,
            SupplyChainCategory.PAYMENT_TRANSACTION: 15.0,
            SupplyChainCategory.DIGITAL_INTERFACE: 10.0,
            SupplyChainCategory.OTHER: 5.0
        }
        
        base_score += category_weights.get(self.category_classification.category, 5.0)
        
        # Impact de la longueur (avis détaillés plus critiques)
        if len(self.content) > 500:
            base_score += 15.0
        elif len(self.content) > 200:
            base_score += 8.0
        
        # Impact de la note (si disponible)
        if self.metadata.rating:
            if self.metadata.rating <= 2:
                base_score += 20.0
            elif self.metadata.rating == 3:
                base_score += 10.0
        
        # Normalisation 0-100
        final_score = min(100.0, max(0.0, base_score))
        
        self.criticality_score = CriticalityScore(final_score)
        
        # Flag pour action immédiate
        if final_score >= 80.0:
            self.requires_immediate_action = True
        
        return self.criticality_score
    
    def get_criticality_level(self) -> CriticalityLevel:
        """Retourne le niveau de criticité."""
        if not self.criticality_score:
            self.calculate_criticality()
        
        score = self.criticality_score.value
        
        if score >= 80.0:
            return CriticalityLevel.CRITICAL
        elif score >= 60.0:
            return CriticalityLevel.HIGH
        elif score >= 30.0:
            return CriticalityLevel.MEDIUM
        else:
            return CriticalityLevel.LOW


@dataclass
class AnalyticsReport:
    """Rapport d'analytics métier."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utc_now)
    period_start: datetime = field(default_factory=utc_now)
    period_end: datetime = field(default_factory=utc_now)
    
    # KPIs métier
    total_reviews: int = 0
    average_sentiment: float = 0.0
    nps_score: float = 0.0
    satisfaction_index: float = 0.0
    
    # Distribution par catégorie
    category_distribution: Dict[SupplyChainCategory, int] = field(default_factory=dict)
    sentiment_distribution: Dict[SentimentLabel, int] = field(default_factory=dict)
    criticality_distribution: Dict[CriticalityLevel, int] = field(default_factory=dict)
    
    # Tendances
    sentiment_trend_7d: float = 0.0
    volume_trend_7d: float = 0.0
    
    # Top insights
    top_negative_categories: List[str] = field(default_factory=list)
    most_critical_reviews: List[ReviewId] = field(default_factory=list)
    trending_keywords: List[str] = field(default_factory=list)


@dataclass
class BusinessAlert:
    """Alerte métier."""
    level: CriticalityLevel
    category: SupplyChainCategory
    title: str
    description: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utc_now)
    recommended_actions: List[str] = field(default_factory=list)
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def acknowledge(self, user_id: str) -> None:
        """Acquitte l'alerte."""
        self.is_acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now(timezone.utc)
