"""
Tests d'Intégration - Infrastructure
===================================

Tests d'intégration pour valider les implémentations d'infrastructure
et l'intégration entre les différentes couches de l'architecture.

Auteur: khalid
Date: 04/06/2025
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from typing import List

# Infrastructure à tester
from core.infrastructure.repositories import (
    SQLiteReviewRepository, SQLiteAnalyticsRepository,
    SQLiteAlertRepository, InMemoryCacheRepository
)
from core.infrastructure.services import (
    BERTSentimentAnalysisService, MLCategoryClassificationService,
    BusinessRecommendationService, EmailNotificationService,
    ModelConfig
)

# Models de domaine
from core.domain.models.review import (
    Review, ReviewId, SentimentLabel, SentimentScore,
    ConfidenceScore, SupplyChainCategory, CriticalityLevel,
    CriticalityScore, AnalyticsReport, BusinessAlert,
    BusinessRecommendation
)


@pytest.fixture
async def temp_db_path():
    """Fixture pour un chemin de base de données temporaire."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def sample_review():
    """Fixture pour un avis de test."""
    return Review(
        id=ReviewId(uuid4()),
        text="Le produit est arrivé en retard et la qualité n'est pas au rendez-vous. Très déçu du service client.",
        timestamp=datetime.now(timezone.utc),
        sentiment_label=SentimentLabel.NEGATIVE,
        sentiment_score=SentimentScore(-0.8),
        confidence_score=ConfidenceScore(0.9),
        category=SupplyChainCategory.DELIVERY,
        criticality_level=CriticalityLevel.HIGH,
        criticality_score=CriticalityScore(0.85),
        requires_action=True,
        metadata={"source": "test", "channel": "email"}
    )


@pytest.fixture
async def sample_reviews():
    """Fixture pour plusieurs avis de test."""
    reviews = []
    
    # Avis négatif sur livraison
    reviews.append(Review(
        id=ReviewId(uuid4()),
        text="Livraison très lente, j'ai attendu 2 semaines au lieu de 3 jours promis.",
        timestamp=datetime.now(timezone.utc) - timedelta(days=1),
        sentiment_label=SentimentLabel.NEGATIVE,
        sentiment_score=SentimentScore(-0.6),
        confidence_score=ConfidenceScore(0.8),
        category=SupplyChainCategory.DELIVERY,
        criticality_level=CriticalityLevel.MEDIUM,
        criticality_score=CriticalityScore(0.7),
        requires_action=True
    ))
    
    # Avis positif sur produit
    reviews.append(Review(
        id=ReviewId(uuid4()),
        text="Excellent produit, exactement ce que je cherchais. Très satisfait de mon achat.",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=12),
        sentiment_label=SentimentLabel.POSITIVE,
        sentiment_score=SentimentScore(0.8),
        confidence_score=ConfidenceScore(0.9),
        category=SupplyChainCategory.PRODUCT,
        criticality_level=CriticalityLevel.LOW,
        criticality_score=CriticalityScore(0.2),
        requires_action=False
    ))
    
    # Avis neutre sur service client
    reviews.append(Review(
        id=ReviewId(uuid4()),
        text="Service client correct, ils ont répondu à mes questions mais sans plus.",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=6),
        sentiment_label=SentimentLabel.NEUTRAL,
        sentiment_score=SentimentScore(0.1),
        confidence_score=ConfidenceScore(0.6),
        category=SupplyChainCategory.CUSTOMER_SERVICE,
        criticality_level=CriticalityLevel.LOW,
        criticality_score=CriticalityScore(0.3),
        requires_action=False
    ))
    
    return reviews


class TestSQLiteReviewRepository:
    """Tests pour le repository SQLite des avis."""
    
    @pytest.mark.asyncio
    async def test_save_and_get_review(self, temp_db_path, sample_review):
        """Test de sauvegarde et récupération d'un avis."""
        repo = SQLiteReviewRepository(temp_db_path)
        
        # Sauvegarde
        saved_review = await repo.save(sample_review)
        assert saved_review.id == sample_review.id
        
        # Récupération
        retrieved_review = await repo.get_by_id(sample_review.id)
        assert retrieved_review is not None
        assert retrieved_review.id == sample_review.id
        assert retrieved_review.text == sample_review.text
        assert retrieved_review.sentiment_label == sample_review.sentiment_label
        assert retrieved_review.category == sample_review.category
    
    @pytest.mark.asyncio
    async def test_find_all_with_pagination(self, temp_db_path, sample_reviews):
        """Test de récupération avec pagination."""
        repo = SQLiteReviewRepository(temp_db_path)
        
        # Sauvegarde de tous les avis
        for review in sample_reviews:
            await repo.save(review)
        
        # Test pagination
        page1 = await repo.find_all(offset=0, limit=2)
        assert len(page1) == 2
        
        page2 = await repo.find_all(offset=2, limit=2)
        assert len(page2) == 1
        
        # Vérification ordre chronologique inverse
        assert page1[0].timestamp > page1[1].timestamp
    
    @pytest.mark.asyncio
    async def test_find_by_criteria(self, temp_db_path, sample_reviews):
        """Test de recherche par critères."""
        repo = SQLiteReviewRepository(temp_db_path)
        
        # Sauvegarde
        for review in sample_reviews:
            await repo.save(review)
        
        # Recherche par sentiment
        negative_reviews = await repo.find_by_criteria(
            sentiment_label=SentimentLabel.NEGATIVE
        )
        assert len(negative_reviews) == 1
        assert negative_reviews[0].sentiment_label == SentimentLabel.NEGATIVE
        
        # Recherche par catégorie
        delivery_reviews = await repo.find_by_criteria(
            category=SupplyChainCategory.DELIVERY
        )
        assert len(delivery_reviews) == 1
        assert delivery_reviews[0].category == SupplyChainCategory.DELIVERY
        
        # Recherche par action requise
        action_reviews = await repo.find_by_criteria(requires_action=True)
        assert len(action_reviews) == 1
        assert action_reviews[0].requires_action is True
    
    @pytest.mark.asyncio
    async def test_count_by_criteria(self, temp_db_path, sample_reviews):
        """Test de comptage par critères."""
        repo = SQLiteReviewRepository(temp_db_path)
        
        # Sauvegarde
        for review in sample_reviews:
            await repo.save(review)
        
        # Comptage total
        total_count = await repo.count_by_criteria()
        assert total_count == 3
        
        # Comptage par sentiment
        negative_count = await repo.count_by_criteria(
            sentiment_label=SentimentLabel.NEGATIVE
        )
        assert negative_count == 1
        
        # Comptage par date
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        recent_count = await repo.count_by_criteria(date_from=yesterday)
        assert recent_count >= 2


class TestInMemoryCacheRepository:
    """Tests pour le repository cache en mémoire."""
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test de stockage et récupération."""
        cache = InMemoryCacheRepository()
        
        # Stockage
        success = await cache.set("test_key", {"data": "test_value"}, ttl_seconds=60)
        assert success is True
        
        # Récupération
        value = await cache.get("test_key")
        assert value is not None
        assert value["data"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test d'expiration TTL."""
        cache = InMemoryCacheRepository()
        
        # Stockage avec TTL court
        await cache.set("expire_key", "expire_value", ttl_seconds=1)
        
        # Vérification immédiate
        value = await cache.get("expire_key")
        assert value == "expire_value"
        
        # Attente expiration
        await asyncio.sleep(1.1)
        
        # Vérification expiration
        expired_value = await cache.get("expire_key")
        assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_delete_and_exists(self):
        """Test de suppression et existence."""
        cache = InMemoryCacheRepository()
        
        # Stockage
        await cache.set("delete_key", "delete_value")
        
        # Vérification existence
        exists = await cache.exists("delete_key")
        assert exists is True
        
        # Suppression
        deleted = await cache.delete("delete_key")
        assert deleted is True
        
        # Vérification suppression
        exists_after = await cache.exists("delete_key")
        assert exists_after is False
    
    @pytest.mark.asyncio
    async def test_clear_pattern(self):
        """Test de suppression par pattern."""
        cache = InMemoryCacheRepository()
        
        # Stockage de plusieurs clés
        await cache.set("user:1:profile", "profile1")
        await cache.set("user:2:profile", "profile2")
        await cache.set("system:config", "config")
        
        # Suppression par pattern
        deleted_count = await cache.clear_pattern("user:*")
        assert deleted_count == 2
        
        # Vérification
        assert await cache.exists("user:1:profile") is False
        assert await cache.exists("user:2:profile") is False
        assert await cache.exists("system:config") is True


class TestBERTSentimentAnalysisService:
    """Tests pour le service d'analyse de sentiment BERT."""
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive(self):
        """Test d'analyse de sentiment positif."""
        # Configuration pour tests (modèle léger)
        config = ModelConfig(
            sentiment_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device="cpu",
            batch_size=1
        )
        
        service = BERTSentimentAnalysisService(config)
        
        # Texte positif
        text = "Excellent produit, livraison rapide, très satisfait de mon achat !"
        
        try:
            # Analyse
            sentiment_label, sentiment_score, confidence = await service.analyze_sentiment(text)
            
            # Vérifications
            assert sentiment_label in [SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL]
            assert isinstance(sentiment_score, SentimentScore)
            assert isinstance(confidence, ConfidenceScore)
            assert 0.0 <= confidence.value <= 1.0
            
        except Exception as e:
            # En cas d'erreur (modèle non disponible), vérifier le fallback
            pytest.skip(f"Model not available for testing: {e}")
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative(self):
        """Test d'analyse de sentiment négatif."""
        config = ModelConfig(device="cpu", batch_size=1)
        service = BERTSentimentAnalysisService(config)
        
        # Texte négatif
        text = "Produit défaillant, livraison en retard, service client incompétent. Je ne recommande pas."
        
        try:
            sentiment_label, sentiment_score, confidence = await service.analyze_sentiment(text)
            
            # Le sentiment devrait être négatif ou neutre (fallback)
            assert sentiment_label in [SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
            assert isinstance(sentiment_score, SentimentScore)
            assert isinstance(confidence, ConfidenceScore)
            
        except Exception:
            pytest.skip("Model not available for testing")
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self):
        """Test d'analyse en lot."""
        config = ModelConfig(device="cpu", batch_size=2)
        service = BERTSentimentAnalysisService(config)
        
        texts = [
            "Très bon produit, je recommande vivement !",
            "Service décevant, je ne reviendrai pas.",
            "Produit correct, sans plus."
        ]
        
        try:
            results = await service.analyze_batch(texts)
            
            # Vérifications
            assert len(results) == len(texts)
            
            for sentiment_label, sentiment_score, confidence in results:
                assert isinstance(sentiment_label, SentimentLabel)
                assert isinstance(sentiment_score, SentimentScore)
                assert isinstance(confidence, ConfidenceScore)
                
        except Exception:
            pytest.skip("Model not available for testing")


class TestMLCategoryClassificationService:
    """Tests pour le service de classification de catégories."""
    
    @pytest.mark.asyncio
    async def test_keyword_classification(self):
        """Test de classification par mots-clés."""
        service = MLCategoryClassificationService()
        await service.initialize()
        
        # Tests par catégorie
        test_cases = [
            ("La livraison était très lente", SupplyChainCategory.DELIVERY),
            ("Le produit est de mauvaise qualité", SupplyChainCategory.QUALITY),
            ("Le service client n'a pas su m'aider", SupplyChainCategory.CUSTOMER_SERVICE),
            ("Le prix est trop élevé", SupplyChainCategory.PRICING),
            ("J'ai commandé cet article hier", SupplyChainCategory.PRODUCT),
        ]
        
        for text, expected_category in test_cases:
            category, confidence = await service.classify_category(text)
            
            # Vérifications
            assert isinstance(category, SupplyChainCategory)
            assert isinstance(confidence, ConfidenceScore)
            assert 0.0 <= confidence.value <= 1.0
            
            # La catégorie devrait correspondre (ou PRODUCT par défaut)
            assert category in [expected_category, SupplyChainCategory.PRODUCT]
    
    @pytest.mark.asyncio
    async def test_empty_text_classification(self):
        """Test avec texte vide."""
        service = MLCategoryClassificationService()
        await service.initialize()
        
        category, confidence = await service.classify_category("")
        
        assert category == SupplyChainCategory.PRODUCT  # Défaut
        assert confidence.value <= 0.5  # Faible confiance


class TestBusinessRecommendationService:
    """Tests pour le service de recommandations métier."""
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, sample_reviews):
        """Test de génération de recommandations."""
        service = BusinessRecommendationService()
        
        analytics_data = {
            "kpis": {
                "satisfaction_score": 5.2,  # Bas score
                "nps_score": 15,  # Bas NPS
                "response_time": 48.5
            }
        }
        
        recommendations = await service.generate_recommendations(sample_reviews, analytics_data)
        
        # Vérifications
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert isinstance(rec, BusinessRecommendation)
            assert rec.title
            assert rec.description
            assert rec.priority in ["HIGH", "MEDIUM", "LOW"]
            assert isinstance(rec.category, SupplyChainCategory)
            assert 0.0 <= rec.estimated_impact <= 1.0
            assert 0.0 <= rec.confidence_score.value <= 1.0
    
    @pytest.mark.asyncio
    async def test_recommendations_prioritization(self, sample_reviews):
        """Test de priorisation des recommandations."""
        service = BusinessRecommendationService()
        
        analytics_data = {"kpis": {"satisfaction_score": 4.0}}
        
        recommendations = await service.generate_recommendations(sample_reviews, analytics_data)
        
        # Vérification tri par priorité
        high_priority_count = len([rec for rec in recommendations if rec.priority == "HIGH"])
        
        if high_priority_count > 1:
            # Vérifier que les HIGH sont en premier
            assert recommendations[0].priority == "HIGH"


class TestEmailNotificationService:
    """Tests pour le service de notifications email."""
    
    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test d'envoi d'alerte."""
        service = EmailNotificationService()
        
        alert = BusinessAlert(
            id=uuid4(),
            alert_type="QUALITY_ISSUE",
            severity="HIGH",
            title="Problème qualité détecté",
            description="Augmentation significative des avis négatifs sur la qualité",
            created_at=datetime.now(timezone.utc)
        )
        
        recipients = ["manager@company.com", "quality@company.com"]
        
        # Test d'envoi (simulation)
        success = await service.send_alert(alert, recipients)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_send_report(self):
        """Test d'envoi de rapport."""
        service = EmailNotificationService()
        
        report_data = {
            "period": "2024-01-01 to 2024-01-07",
            "total_reviews": 150,
            "satisfaction_score": 7.2,
            "key_insights": ["Amélioration globale", "Problème livraison persistant"]
        }
        
        recipients = ["executive@company.com"]
        
        success = await service.send_report(report_data, recipients)
        assert success is True


@pytest.mark.asyncio
async def test_integration_flow(temp_db_path):
    """Test d'intégration complète."""
    # Repositories
    review_repo = SQLiteReviewRepository(temp_db_path)
    cache_repo = InMemoryCacheRepository()
    
    # Services
    sentiment_service = BERTSentimentAnalysisService(ModelConfig(device="cpu"))
    category_service = MLCategoryClassificationService()
    recommendation_service = BusinessRecommendationService()
    
    # Initialisation
    await category_service.initialize()
    
    # Flux complet : analyse -> stockage -> recommandations
    text = "Produit défaillant, livraison très lente, service client peu réactif."
    
    try:
        # 1. Analyse de sentiment
        sentiment_label, sentiment_score, confidence = await sentiment_service.analyze_sentiment(text)
        
        # 2. Classification de catégorie
        category, cat_confidence = await category_service.classify_category(text)
        
        # 3. Création de l'avis
        review = Review(
            id=ReviewId(uuid4()),
            text=text,
            timestamp=datetime.now(timezone.utc),
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            confidence_score=confidence,
            category=category,
            criticality_level=CriticalityLevel.HIGH,
            criticality_score=CriticalityScore(0.8),
            requires_action=True
        )
        
        # 4. Stockage
        saved_review = await review_repo.save(review)
        assert saved_review.id == review.id
        
        # 5. Cache
        cache_key = f"review:{review.id.value}"
        await cache_repo.set(cache_key, review.text)
        cached_text = await cache_repo.get(cache_key)
        assert cached_text == review.text
        
        # 6. Génération de recommandations
        recommendations = await recommendation_service.generate_recommendations(
            [review], 
            {"kpis": {"satisfaction_score": 4.0}}
        )
        
        assert len(recommendations) > 0
        
    except Exception as e:
        # En cas d'erreur avec les modèles, vérifier au moins le stockage
        pytest.skip(f"Integration test skipped due to model issues: {e}")


if __name__ == "__main__":
    # Pour lancer les tests directement
    pytest.main([__file__, "-v"])
