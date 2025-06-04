"""
Tests unitaires pour les modèles de domaine
==========================================

Te        with pytest.raises(ValueError):
            ReviewMetadata(
                source=ReviewSource.TRUSTPILOT,
                collected_at=datetime.now(timezone.utc),
                rating=0
            )
        
        with pytest.raises(ValueError):
            ReviewMetadata(
                source=ReviewSource.TRUSTPILOT,
                collected_at=datetime.now(timezone.utc),
                rating=6
            )avec fixtures, mocks et validation des règles métier.

Auteur: khalid
Date: 04/06/2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from core.domain.models.review import (
    Review, ReviewId, SentimentScore, ConfidenceScore, CriticalityScore,
    SentimentAnalysis, CategoryClassification, ReviewMetadata,
    BusinessRecommendation, AnalyticsReport, BusinessAlert,
    SentimentLabel, SupplyChainCategory, ReviewSource, CriticalityLevel
)


class TestValueObjects:
    """Tests pour les Value Objects."""
    
    def test_review_id_valid(self) -> None:
        """Test création ReviewId valide."""
        review_id = ReviewId("REV-123")
        assert review_id.value == "REV-123"
    
    def test_review_id_empty_raises_error(self) -> None:
        """Test ReviewId vide lève une erreur."""
        with pytest.raises(ValueError, match="Review ID cannot be empty"):
            ReviewId("")
        
        with pytest.raises(ValueError):
            ReviewId("   ")
    
    def test_sentiment_score_valid_range(self) -> None:
        """Test SentimentScore dans la plage valide."""
        score = SentimentScore(0.5)
        assert score.value == 0.5
        
        # Limites
        assert SentimentScore(-1.0).value == -1.0
        assert SentimentScore(1.0).value == 1.0
    
    def test_sentiment_score_invalid_range(self) -> None:
        """Test SentimentScore hors plage lève erreur."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1.0 and 1.0"):
            SentimentScore(-1.1)
        
        with pytest.raises(ValueError):
            SentimentScore(1.1)
    
    def test_confidence_score_valid_range(self) -> None:
        """Test ConfidenceScore valide."""
        score = ConfidenceScore(0.85)
        assert score.value == 0.85
    
    def test_confidence_score_invalid_range(self) -> None:
        """Test ConfidenceScore invalide."""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            ConfidenceScore(-0.1)
    
    def test_criticality_score_valid_range(self) -> None:
        """Test CriticalityScore valide."""
        score = CriticalityScore(75.5)
        assert score.value == 75.5
    
    def test_criticality_score_invalid_range(self) -> None:
        """Test CriticalityScore invalide."""
        with pytest.raises(ValueError, match="Criticality score must be between 0.0 and 100.0"):
            CriticalityScore(100.1)


class TestReviewMetadata:
    """Tests pour les métadonnées d'avis."""
    
    def test_metadata_creation_valid(self) -> None:
        """Test création métadonnées valides."""
        metadata = ReviewMetadata(
            source=ReviewSource.TRUSTPILOT,
            collected_at=datetime.now(timezone.utc),
            rating=4,
            verified_purchase=True
        )
        
        assert metadata.source == ReviewSource.TRUSTPILOT
        assert metadata.rating == 4
        assert metadata.verified_purchase is True
    
    def test_metadata_invalid_rating(self) -> None:
        """Test rating invalide."""
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            ReviewMetadata(
                source=ReviewSource.TRUSTPILOT,
                collected_at=datetime.now(timezone.utc),
                rating=0
            )
        
        with pytest.raises(ValueError):
            ReviewMetadata(
                source=ReviewSource.TRUSTPILOT,
                collected_at=datetime.now(timezone.utc),
                rating=6
            )


class TestReview:
    """Tests pour l'entité Review."""
    
    @pytest.fixture
    def sample_metadata(self) -> ReviewMetadata:
        """Fixture pour métadonnées d'exemple."""
        return ReviewMetadata(
            source=ReviewSource.TRUSTPILOT,
            collected_at=datetime.now(timezone.utc),
            rating=2,
            verified_purchase=True
        )
    
    @pytest.fixture
    def sample_sentiment(self) -> SentimentAnalysis:
        """Fixture pour analyse de sentiment."""
        return SentimentAnalysis(
            score=SentimentScore(-0.7),
            label=SentimentLabel.NEGATIVE,
            confidence=ConfidenceScore(0.9),
            model_version="bert-v1.0"
        )
    
    @pytest.fixture
    def sample_category(self) -> CategoryClassification:
        """Fixture pour classification catégorie."""
        return CategoryClassification(
            category=SupplyChainCategory.DELIVERY_LOGISTICS,
            confidence=ConfidenceScore(0.8),
            keywords_found=["livraison", "retard", "délai"]
        )
    
    def test_review_creation_valid(self, sample_metadata: ReviewMetadata) -> None:
        """Test création review valide."""
        review = Review(
            id=ReviewId("REV-001"),
            content="Livraison très en retard, très déçu",
            metadata=sample_metadata
        )
        
        assert review.id.value == "REV-001"
        assert "retard" in review.content
        assert review.is_processed is False
        assert review.requires_immediate_action is False
    
    def test_review_empty_content_raises_error(self, sample_metadata: ReviewMetadata) -> None:
        """Test contenu vide lève erreur."""
        with pytest.raises(ValueError, match="Review content cannot be empty"):
            Review(
                id=ReviewId("REV-001"),
                content="",
                metadata=sample_metadata
            )
        
        with pytest.raises(ValueError):
            Review(
                id=ReviewId("REV-001"),
                content="   ",
                metadata=sample_metadata
            )
    
    def test_review_content_too_long_raises_error(self, sample_metadata: ReviewMetadata) -> None:
        """Test contenu trop long lève erreur."""
        long_content = "x" * 10001
        
        with pytest.raises(ValueError, match="Review content too long"):
            Review(
                id=ReviewId("REV-001"),
                content=long_content,
                metadata=sample_metadata
            )
    
    def test_mark_as_processed(self, sample_metadata: ReviewMetadata) -> None:
        """Test marquage comme traité."""
        review = Review(
            id=ReviewId("REV-001"),
            content="Test content",
            metadata=sample_metadata
        )
        
        original_time = review.updated_at
        review.mark_as_processed()
        
        assert review.is_processed is True
        assert review.updated_at > original_time
    
    def test_add_sentiment_analysis(
        self, 
        sample_metadata: ReviewMetadata,
        sample_sentiment: SentimentAnalysis
    ) -> None:
        """Test ajout analyse sentiment."""
        review = Review(
            id=ReviewId("REV-001"),
            content="Test content",
            metadata=sample_metadata
        )
        
        original_time = review.updated_at
        review.add_sentiment_analysis(sample_sentiment)
        
        assert review.sentiment_analysis == sample_sentiment
        assert review.updated_at > original_time
    
    def test_calculate_criticality_complete(
        self,
        sample_metadata: ReviewMetadata,
        sample_sentiment: SentimentAnalysis,
        sample_category: CategoryClassification
    ) -> None:
        """Test calcul criticité avec données complètes."""
        review = Review(
            id=ReviewId("REV-001"),
            content="Livraison très en retard, vraiment déçu de ce service, jamais vu ça auparavant dans ma vie, c'est inacceptable pour une entreprise de cette taille et réputation",
            metadata=sample_metadata
        )
        
        review.add_sentiment_analysis(sample_sentiment)
        review.add_category_classification(sample_category)
        
        criticality = review.calculate_criticality()
        
        # Score attendu:
        # - Sentiment très négatif (-0.7): 40 points
        # - Catégorie livraison: 25 points  
        # - Contenu long (>200 chars): 8 points
        # - Rating 2: 20 points
        # Total: 93 points
        
        assert criticality.value >= 80.0  # Score élevé
        assert review.requires_immediate_action is True
        assert review.get_criticality_level() == CriticalityLevel.CRITICAL
    
    def test_calculate_criticality_without_data(self, sample_metadata: ReviewMetadata) -> None:
        """Test calcul criticité sans données d'analyse."""
        review = Review(
            id=ReviewId("REV-001"),
            content="Test content",
            metadata=sample_metadata
        )
        
        criticality = review.calculate_criticality()
        assert criticality.value == 0.0
    
    def test_get_criticality_level_ranges(
        self,
        sample_metadata: ReviewMetadata,
        sample_sentiment: SentimentAnalysis,
        sample_category: CategoryClassification
    ) -> None:
        """Test niveaux de criticité selon les scores."""
        review = Review(
            id=ReviewId("REV-001"),
            content="Test",
            metadata=sample_metadata
        )
        
        # Test différents scores
        test_cases = [
            (CriticalityScore(90.0), CriticalityLevel.CRITICAL),
            (CriticalityScore(70.0), CriticalityLevel.HIGH),
            (CriticalityScore(45.0), CriticalityLevel.MEDIUM),
            (CriticalityScore(15.0), CriticalityLevel.LOW),
        ]
        
        for score, expected_level in test_cases:
            review.criticality_score = score
            assert review.get_criticality_level() == expected_level


class TestAnalyticsReport:
    """Tests pour les rapports d'analytics."""
    
    def test_analytics_report_creation(self) -> None:
        """Test création rapport analytics."""
        report = AnalyticsReport(
            total_reviews=1000,
            average_sentiment=0.2,
            nps_score=35.5
        )
        
        assert report.total_reviews == 1000
        assert report.average_sentiment == 0.2
        assert report.nps_score == 35.5
        assert isinstance(report.id, type(uuid4()))
    
    def test_analytics_report_distributions(self) -> None:
        """Test distributions dans le rapport."""
        report = AnalyticsReport()
        
        # Test distribution par catégorie
        report.category_distribution = {
            SupplyChainCategory.DELIVERY_LOGISTICS: 150,
            SupplyChainCategory.PRODUCT_QUALITY: 100,
            SupplyChainCategory.CUSTOMER_SERVICE: 75
        }
        
        assert len(report.category_distribution) == 3
        assert report.category_distribution[SupplyChainCategory.DELIVERY_LOGISTICS] == 150


class TestBusinessAlert:
    """Tests pour les alertes business."""
    
    def test_business_alert_creation(self) -> None:
        """Test création alerte business."""
        alert = BusinessAlert(
            level=CriticalityLevel.CRITICAL,
            category=SupplyChainCategory.DELIVERY_LOGISTICS,
            title="Problème livraison critique",
            description="Augmentation significative des plaintes livraison"
        )
        
        assert alert.level == CriticalityLevel.CRITICAL
        assert alert.category == SupplyChainCategory.DELIVERY_LOGISTICS
        assert alert.is_acknowledged is False
        assert alert.acknowledged_by is None
    
    def test_alert_acknowledge(self) -> None:
        """Test acquittement d'alerte."""
        alert = BusinessAlert(
            level=CriticalityLevel.HIGH,
            category=SupplyChainCategory.CUSTOMER_SERVICE,
            title="Test Alert",
            description="Test Description"
        )
        
        alert.acknowledge("user123")
        
        assert alert.is_acknowledged is True
        assert alert.acknowledged_by == "user123"
        assert alert.acknowledged_at is not None
        assert isinstance(alert.acknowledged_at, datetime)


class TestBusinessRecommendation:
    """Tests pour les recommandations business."""
    
    def test_recommendation_creation(self) -> None:
        """Test création recommandation."""
        recommendation = BusinessRecommendation(
            action="Améliorer les délais de livraison",
            priority=CriticalityLevel.HIGH,
            estimated_impact="Réduction de 30% des plaintes",
            timeframe="2 semaines",
            responsible_team="Logistique",
            estimated_cost=5000.0
        )
        
        assert recommendation.action == "Améliorer les délais de livraison"
        assert recommendation.priority == CriticalityLevel.HIGH
        assert recommendation.estimated_cost == 5000.0


@pytest.mark.integration
class TestIntegrationScenarios:
    """Tests d'intégration pour les scénarios complets."""
    
    def test_complete_review_processing_scenario(self) -> None:
        """Test scénario complet de traitement d'avis."""
        # 1. Création avis
        metadata = ReviewMetadata(
            source=ReviewSource.TRUSTPILOT,
            collected_at=datetime.now(timezone.utc),
            rating=1,
            verified_purchase=True
        )
        
        review = Review(
            id=ReviewId("REV-SCENARIO-001"),
            content="Livraison catastrophique, 2 semaines de retard, produit endommagé, service client injoignable, je ne recommande absolument pas",
            metadata=metadata
        )
        
        # 2. Ajout analyses
        sentiment = SentimentAnalysis(
            score=SentimentScore(-0.9),
            label=SentimentLabel.NEGATIVE,
            confidence=ConfidenceScore(0.95)
        )
        
        category = CategoryClassification(
            category=SupplyChainCategory.DELIVERY_LOGISTICS,
            confidence=ConfidenceScore(0.85),
            keywords_found=["livraison", "retard", "endommagé"]
        )
        
        review.add_sentiment_analysis(sentiment)
        review.add_category_classification(category)
        
        # 3. Calcul criticité
        criticality = review.calculate_criticality()
        
        # 4. Vérifications
        assert review.is_processed is False  # Pas encore marqué
        assert criticality.value >= 80.0  # Critique ou très critique
        assert review.requires_immediate_action is True
        assert review.get_criticality_level() in [CriticalityLevel.HIGH, CriticalityLevel.CRITICAL]
        
        # 5. Marquage traité
        review.mark_as_processed()
        assert review.is_processed is True
    
    def test_multiple_reviews_analytics_scenario(self) -> None:
        """Test scénario analytics avec plusieurs avis."""
        reviews = []
        
        # Création de plusieurs avis avec différents niveaux
        test_data = [
            ("Excellent service", 5, 0.8, SentimentLabel.POSITIVE),
            ("Correct mais peut mieux faire", 3, 0.1, SentimentLabel.NEUTRAL),
            ("Très déçu du service", 2, -0.6, SentimentLabel.NEGATIVE),
            ("Catastrophique", 1, -0.9, SentimentLabel.NEGATIVE),
        ]
        
        for i, (content, rating, sentiment_score, sentiment_label) in enumerate(test_data):
            metadata = ReviewMetadata(
                source=ReviewSource.TRUSTPILOT,
                collected_at=datetime.now(timezone.utc),
                rating=rating
            )
            
            review = Review(
                id=ReviewId(f"REV-MULTI-{i:03d}"),
                content=content,
                metadata=metadata
            )
            
            sentiment = SentimentAnalysis(
                score=SentimentScore(sentiment_score),
                label=sentiment_label,
                confidence=ConfidenceScore(0.8)
            )
            
            review.add_sentiment_analysis(sentiment)
            reviews.append(review)
        
        # Calcul des statistiques
        avg_sentiment = sum(
            r.sentiment_analysis.score.value for r in reviews
        ) / len(reviews)
        
        negative_count = sum(
            1 for r in reviews 
            if r.sentiment_analysis.label == SentimentLabel.NEGATIVE
        )
        
        # Vérifications
        assert len(reviews) == 4
        assert avg_sentiment < 0  # Moyenne négative
        assert negative_count == 2  # 2 avis négatifs
        assert negative_count / len(reviews) == 0.5  # 50% négatifs
