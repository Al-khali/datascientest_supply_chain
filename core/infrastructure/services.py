"""
Infrastructure Services
======================

Implémentations concrètes des services métier et ML.
Intègre les modèles d'IA et la logique de traitement.

Auteur: khalid
Date: 04/06/2025
"""

import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import joblib

# ML Libraries
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Business logic
from core.domain.interfaces.services import (
    SentimentAnalysisService, CategoryClassificationService,
    RecommendationService, NotificationService
)
from core.domain.models.review import (
    Review, SentimentLabel, SentimentScore, ConfidenceScore,
    SupplyChainCategory, CriticalityLevel, CriticalityScore,
    BusinessRecommendation, BusinessAlert
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration des modèles ML."""
    sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    category_model_path: str = "models/category_classifier.pkl"
    device: str = "cpu"  # "cuda" si GPU disponible
    max_length: int = 512
    batch_size: int = 32
    cache_predictions: bool = True


class BERTSentimentAnalysisService(SentimentAnalysisService):
    """Service d'analyse de sentiment avec BERT multilingual."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialise le service d'analyse de sentiment.
        
        Args:
            config: Configuration du modèle
        """
        self.config = config or ModelConfig()
        self._sentiment_pipeline: Optional[Pipeline] = None
        self._tokenizer = None
        self._model = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialise les modèles ML de manière asynchrone."""
        if self._initialized:
            return
            
        logger.info("Initializing BERT sentiment analysis model...")
        
        try:
            # Chargement du modèle dans un thread séparé pour éviter le blocage
            loop = asyncio.get_event_loop()
            
            self._tokenizer, self._model, self._sentiment_pipeline = await loop.run_in_executor(
                None, self._load_models
            )
            
            self._initialized = True
            logger.info("BERT sentiment model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment model: {e}")
            raise
    
    def _load_models(self) -> Tuple[Any, Any, Pipeline]:
        """Charge les modèles ML de manière synchrone."""
        # Chargement du tokenizer et modèle BERT
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.sentiment_model_name,
            use_fast=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.sentiment_model_name
        )
        
        # Pipeline d'analyse de sentiment
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.config.device == "cuda" and torch.cuda.is_available() else -1,
            return_all_scores=True
        )
        
        return tokenizer, model, sentiment_pipeline
    
    async def analyze_sentiment(self, text: str) -> Tuple[SentimentLabel, SentimentScore, ConfidenceScore]:
        """Analyse le sentiment d'un texte."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Preprocessing du texte
            cleaned_text = self._preprocess_text(text)
            
            # Analyse asynchrone
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, self._sentiment_pipeline, cleaned_text
            )
            
            # Traitement des résultats
            sentiment_data = results[0]  # Premier élément car return_all_scores=True
            
            # Mapping des labels BERT vers nos enums
            label_mapping = {
                "POSITIVE": SentimentLabel.POSITIVE,
                "NEGATIVE": SentimentLabel.NEGATIVE,
                "NEUTRAL": SentimentLabel.NEUTRAL,
                "1 star": SentimentLabel.NEGATIVE,
                "2 stars": SentimentLabel.NEGATIVE,
                "3 stars": SentimentLabel.NEUTRAL,
                "4 stars": SentimentLabel.POSITIVE,
                "5 stars": SentimentLabel.POSITIVE
            }
            
            # Détermination du sentiment dominant
            best_prediction = max(sentiment_data, key=lambda x: x['score'])
            bert_label = best_prediction['label']
            confidence = best_prediction['score']
            
            # Conversion vers nos types
            sentiment_label = label_mapping.get(bert_label, SentimentLabel.NEUTRAL)
            
            # Calcul du score de sentiment (-1 à 1)
            if sentiment_label == SentimentLabel.POSITIVE:
                sentiment_score = confidence
            elif sentiment_label == SentimentLabel.NEGATIVE:
                sentiment_score = -confidence
            else:
                sentiment_score = 0.0
            
            return (
                sentiment_label,
                SentimentScore(sentiment_score),
                ConfidenceScore(confidence)
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Fallback vers sentiment neutre
            return (
                SentimentLabel.NEUTRAL,
                SentimentScore(0.0),
                ConfidenceScore(0.5)
            )
    
    async def analyze_batch(self, texts: List[str]) -> List[Tuple[SentimentLabel, SentimentScore, ConfidenceScore]]:
        """Analyse un batch de textes pour de meilleures performances."""
        if not self._initialized:
            await self.initialize()
        
        results = []
        
        # Traitement par chunks pour éviter les problèmes de mémoire
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            try:
                # Preprocessing du batch
                cleaned_batch = [self._preprocess_text(text) for text in batch]
                
                # Analyse asynchrone du batch
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    None, self._sentiment_pipeline, cleaned_batch
                )
                
                # Traitement des résultats du batch
                for text_results in batch_results:
                    sentiment_data = text_results
                    
                    best_prediction = max(sentiment_data, key=lambda x: x['score'])
                    bert_label = best_prediction['label']
                    confidence = best_prediction['score']
                    
                    label_mapping = {
                        "POSITIVE": SentimentLabel.POSITIVE,
                        "NEGATIVE": SentimentLabel.NEGATIVE,
                        "NEUTRAL": SentimentLabel.NEUTRAL,
                        "1 star": SentimentLabel.NEGATIVE,
                        "2 stars": SentimentLabel.NEGATIVE,
                        "3 stars": SentimentLabel.NEUTRAL,
                        "4 stars": SentimentLabel.POSITIVE,
                        "5 stars": SentimentLabel.POSITIVE
                    }
                    
                    sentiment_label = label_mapping.get(bert_label, SentimentLabel.NEUTRAL)
                    
                    if sentiment_label == SentimentLabel.POSITIVE:
                        sentiment_score = confidence
                    elif sentiment_label == SentimentLabel.NEGATIVE:
                        sentiment_score = -confidence
                    else:
                        sentiment_score = 0.0
                    
                    results.append((
                        sentiment_label,
                        SentimentScore(sentiment_score),
                        ConfidenceScore(confidence)
                    ))
                    
            except Exception as e:
                logger.error(f"Error in batch sentiment analysis: {e}")
                # Fallback pour le batch entier
                for _ in batch:
                    results.append((
                        SentimentLabel.NEUTRAL,
                        SentimentScore(0.0),
                        ConfidenceScore(0.5)
                    ))
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing du texte avant analyse."""
        if not text or not isinstance(text, str):
            return ""
        
        # Nettoyage basique
        text = text.strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalisation des espaces
        
        # Troncature si trop long
        if len(text) > self.config.max_length * 4:  # Approximation de tokens
            text = text[:self.config.max_length * 4]
        
        return text
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle."""
        return {
            "model_name": self.config.sentiment_model_name,
            "device": self.config.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "initialized": self._initialized
        }


class MLCategoryClassificationService(CategoryClassificationService):
    """Service de classification des catégories supply chain."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialise le service de classification.
        
        Args:
            config: Configuration du modèle
        """
        self.config = config or ModelConfig()
        self._classifier = None
        self._vectorizer = None
        self._initialized = False
        
        # Mapping par mots-clés pour bootstrapping
        self._keyword_mapping = {
            SupplyChainCategory.DELIVERY: [
                "livraison", "delivery", "expédition", "délai", "retard", 
                "transport", "colis", "réception", "arrivée"
            ],
            SupplyChainCategory.QUALITY: [
                "qualité", "quality", "défaut", "cassé", "abîmé", 
                "défectueux", "problème", "mauvais"
            ],
            SupplyChainCategory.CUSTOMER_SERVICE: [
                "service", "support", "aide", "réponse", "contact",
                "conseiller", "assistance", "réclamation"
            ],
            SupplyChainCategory.PRICING: [
                "prix", "price", "coût", "cher", "tarif", "promotion",
                "remise", "facture", "paiement"
            ],
            SupplyChainCategory.PRODUCT: [
                "produit", "product", "article", "item", "commande",
                "stock", "disponible", "référence"
            ]
        }
    
    async def initialize(self) -> None:
        """Initialise le modèle de classification."""
        if self._initialized:
            return
            
        logger.info("Initializing category classification model...")
        
        try:
            # Tentative de chargement d'un modèle pré-entraîné
            model_path = Path(self.config.category_model_path)
            
            if model_path.exists():
                loop = asyncio.get_event_loop()
                self._classifier, self._vectorizer = await loop.run_in_executor(
                    None, self._load_trained_model, model_path
                )
                logger.info("Loaded pre-trained category model")
            else:
                logger.info("No pre-trained model found, using keyword-based classification")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize category model: {e}")
            self._initialized = True  # Continue avec keyword-based
    
    def _load_trained_model(self, model_path: Path) -> Tuple[Any, Any]:
        """Charge un modèle pré-entraîné."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['classifier'], model_data['vectorizer']
    
    async def classify_category(self, text: str) -> Tuple[SupplyChainCategory, ConfidenceScore]:
        """Classifie le texte dans une catégorie supply chain."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self._classifier and self._vectorizer:
                # Utilisation du modèle ML si disponible
                return await self._ml_classify(text)
            else:
                # Fallback vers classification par mots-clés
                return self._keyword_classify(text)
                
        except Exception as e:
            logger.error(f"Error in category classification: {e}")
            return SupplyChainCategory.PRODUCT, ConfidenceScore(0.3)
    
    async def _ml_classify(self, text: str) -> Tuple[SupplyChainCategory, ConfidenceScore]:
        """Classification ML si modèle disponible."""
        cleaned_text = self._preprocess_text(text)
        
        loop = asyncio.get_event_loop()
        
        # Vectorisation
        text_vector = await loop.run_in_executor(
            None, self._vectorizer.transform, [cleaned_text]
        )
        
        # Prédiction
        prediction = await loop.run_in_executor(
            None, self._classifier.predict_proba, text_vector
        )
        
        # Récupération de la classe et probabilité
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Mapping vers enum (à adapter selon l'entraînement)
        categories = list(SupplyChainCategory)
        category = categories[class_idx] if class_idx < len(categories) else SupplyChainCategory.PRODUCT
        
        return category, ConfidenceScore(confidence)
    
    def _keyword_classify(self, text: str) -> Tuple[SupplyChainCategory, ConfidenceScore]:
        """Classification par mots-clés."""
        text_lower = text.lower()
        
        category_scores = {}
        
        for category, keywords in self._keyword_mapping.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                # Normalisation par le nombre de mots-clés
                category_scores[category] = score / len(keywords)
        
        if not category_scores:
            return SupplyChainCategory.PRODUCT, ConfidenceScore(0.3)
        
        # Catégorie avec le score le plus élevé
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        return best_category[0], ConfidenceScore(min(0.9, best_category[1] * 2))
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing pour la classification."""
        if not text or not isinstance(text, str):
            return ""
        
        # Nettoyage
        text = text.lower().strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        
        return text
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Informations sur le modèle de classification."""
        return {
            "model_path": self.config.category_model_path,
            "has_ml_model": self._classifier is not None,
            "uses_keywords": self._classifier is None,
            "categories": [cat.value for cat in SupplyChainCategory],
            "initialized": self._initialized
        }


class BusinessRecommendationService(RecommendationService):
    """Service de génération de recommandations métier."""
    
    def __init__(self):
        """Initialise le service de recommandations."""
        self._rules_engine = self._build_rules_engine()
    
    def _build_rules_engine(self) -> Dict[str, Any]:
        """Construit le moteur de règles métier."""
        return {
            "sentiment_thresholds": {
                "critical": -0.7,
                "warning": -0.4,
                "good": 0.6
            },
            "category_weights": {
                SupplyChainCategory.DELIVERY: 1.2,
                SupplyChainCategory.QUALITY: 1.3,
                SupplyChainCategory.CUSTOMER_SERVICE: 1.1,
                SupplyChainCategory.PRICING: 1.0,
                SupplyChainCategory.PRODUCT: 1.0
            },
            "action_templates": {
                "improve_delivery": {
                    "title": "Améliorer les délais de livraison",
                    "description": "Optimiser la chaîne logistique pour réduire les délais",
                    "priority": "HIGH",
                    "estimated_impact": 0.25,
                    "estimated_cost": "MEDIUM"
                },
                "quality_control": {
                    "title": "Renforcer le contrôle qualité",
                    "description": "Mettre en place des contrôles supplémentaires",
                    "priority": "HIGH",
                    "estimated_impact": 0.30,
                    "estimated_cost": "MEDIUM"
                },
                "customer_service_training": {
                    "title": "Formation service client",
                    "description": "Améliorer la formation des équipes support",
                    "priority": "MEDIUM",
                    "estimated_impact": 0.20,
                    "estimated_cost": "LOW"
                }
            }
        }
    
    async def generate_recommendations(
        self, 
        reviews: List[Review],
        analytics_data: Dict[str, Any]
    ) -> List[BusinessRecommendation]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []
        
        try:
            # Analyse des patterns
            patterns = await self._analyze_patterns(reviews)
            
            # Génération de recommandations par catégorie
            category_recommendations = await self._generate_category_recommendations(patterns)
            recommendations.extend(category_recommendations)
            
            # Recommandations basées sur les tendances
            trend_recommendations = await self._generate_trend_recommendations(analytics_data)
            recommendations.extend(trend_recommendations)
            
            # Priorisation et déduplication
            recommendations = self._prioritize_recommendations(recommendations)
            
            return recommendations[:10]  # Limite à 10 recommandations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_default_recommendations(reviews)
    
    async def _analyze_patterns(self, reviews: List[Review]) -> Dict[str, Any]:
        """Analyse les patterns dans les avis."""
        patterns = {
            "sentiment_by_category": {},
            "critical_issues": [],
            "frequent_issues": {},
            "improvement_opportunities": []
        }
        
        # Analyse par catégorie
        category_sentiments = {}
        category_counts = {}
        
        for review in reviews:
            category = review.category
            sentiment_score = review.sentiment_score.value
            
            if category not in category_sentiments:
                category_sentiments[category] = []
                category_counts[category] = 0
            
            category_sentiments[category].append(sentiment_score)
            category_counts[category] += 1
            
            # Identification des problèmes critiques
            if (review.sentiment_label == SentimentLabel.NEGATIVE and 
                review.criticality_level == CriticalityLevel.HIGH):
                patterns["critical_issues"].append({
                    "category": category,
                    "sentiment_score": sentiment_score,
                    "text_excerpt": review.text[:100]
                })
        
        # Calcul des moyennes par catégorie
        for category, scores in category_sentiments.items():
            patterns["sentiment_by_category"][category] = {
                "average_sentiment": np.mean(scores),
                "count": len(scores),
                "negative_ratio": len([s for s in scores if s < -0.2]) / len(scores)
            }
        
        return patterns
    
    async def _generate_category_recommendations(
        self, 
        patterns: Dict[str, Any]
    ) -> List[BusinessRecommendation]:
        """Génère des recommandations par catégorie."""
        recommendations = []
        
        for category, data in patterns["sentiment_by_category"].items():
            avg_sentiment = data["average_sentiment"]
            negative_ratio = data["negative_ratio"]
            
            # Recommandations selon les seuils
            if avg_sentiment < self._rules_engine["sentiment_thresholds"]["critical"]:
                if category == SupplyChainCategory.DELIVERY:
                    template = self._rules_engine["action_templates"]["improve_delivery"]
                elif category == SupplyChainCategory.QUALITY:
                    template = self._rules_engine["action_templates"]["quality_control"]
                elif category == SupplyChainCategory.CUSTOMER_SERVICE:
                    template = self._rules_engine["action_templates"]["customer_service_training"]
                else:
                    continue
                
                recommendation = BusinessRecommendation(
                    title=template["title"],
                    description=f"{template['description']} (Score actuel: {avg_sentiment:.2f})",
                    priority=template["priority"],
                    category=category,
                    estimated_impact=template["estimated_impact"],
                    estimated_cost=template["estimated_cost"],
                    confidence_score=ConfidenceScore(0.8),
                    supporting_data={
                        "average_sentiment": avg_sentiment,
                        "negative_ratio": negative_ratio,
                        "sample_count": data["count"]
                    }
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_trend_recommendations(
        self, 
        analytics_data: Dict[str, Any]
    ) -> List[BusinessRecommendation]:
        """Génère des recommandations basées sur les tendances."""
        recommendations = []
        
        # Recommandations basées sur les KPIs
        if "kpis" in analytics_data:
            kpis = analytics_data["kpis"]
            
            if kpis.get("satisfaction_score", 0) < 6.0:
                recommendations.append(BusinessRecommendation(
                    title="Plan d'amélioration satisfaction globale",
                    description="Mise en place d'un plan d'action pour améliorer la satisfaction client",
                    priority="HIGH",
                    category=SupplyChainCategory.CUSTOMER_SERVICE,
                    estimated_impact=0.35,
                    estimated_cost="HIGH",
                    confidence_score=ConfidenceScore(0.9),
                    supporting_data={"current_satisfaction": kpis.get("satisfaction_score")}
                ))
            
            if kpis.get("nps_score", 0) < 20:
                recommendations.append(BusinessRecommendation(
                    title="Programme d'amélioration NPS",
                    description="Actions ciblées pour améliorer le Net Promoter Score",
                    priority="MEDIUM",
                    category=SupplyChainCategory.CUSTOMER_SERVICE,
                    estimated_impact=0.25,
                    estimated_cost="MEDIUM",
                    confidence_score=ConfidenceScore(0.75),
                    supporting_data={"current_nps": kpis.get("nps_score")}
                ))
        
        return recommendations
    
    def _prioritize_recommendations(
        self, 
        recommendations: List[BusinessRecommendation]
    ) -> List[BusinessRecommendation]:
        """Priorise et déduplique les recommandations."""
        # Déduplication par titre
        unique_recommendations = {}
        for rec in recommendations:
            if rec.title not in unique_recommendations:
                unique_recommendations[rec.title] = rec
            else:
                # Garde celle avec le plus haut impact estimé
                if rec.estimated_impact > unique_recommendations[rec.title].estimated_impact:
                    unique_recommendations[rec.title] = rec
        
        # Tri par priorité et impact
        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        sorted_recommendations = sorted(
            unique_recommendations.values(),
            key=lambda x: (
                priority_order.get(x.priority, 0),
                x.estimated_impact,
                x.confidence_score.value
            ),
            reverse=True
        )
        
        return sorted_recommendations
    
    def _get_default_recommendations(self, reviews: List[Review]) -> List[BusinessRecommendation]:
        """Recommandations par défaut en cas d'erreur."""
        return [
            BusinessRecommendation(
                title="Analyse approfondie requise",
                description="Effectuer une analyse détaillée des retours clients",
                priority="MEDIUM",
                category=SupplyChainCategory.CUSTOMER_SERVICE,
                estimated_impact=0.15,
                estimated_cost="LOW",
                confidence_score=ConfidenceScore(0.6),
                supporting_data={"total_reviews": len(reviews)}
            )
        ]


class EmailNotificationService(NotificationService):
    """Service de notifications par email."""
    
    def __init__(self, smtp_config: Dict[str, Any] = None):
        """Initialise le service de notifications.
        
        Args:
            smtp_config: Configuration SMTP
        """
        self.smtp_config = smtp_config or {
            "host": os.getenv("SMTP_HOST", "localhost"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME", ""),
            "password": os.getenv("SMTP_PASSWORD", ""),
            "use_tls": os.getenv("SMTP_TLS", "true").lower() == "true"
        }
        
    async def send_alert(self, alert: BusinessAlert, recipients: List[str]) -> bool:
        """Envoie une alerte par email."""
        try:
            # Pour l'instant, on log l'alerte
            # Dans un vrai système, on utiliserait smtplib ou un service comme SendGrid
            logger.info(f"ALERT NOTIFICATION: {alert.title}")
            logger.info(f"Recipients: {recipients}")
            logger.info(f"Severity: {alert.severity}")
            logger.info(f"Description: {alert.description}")
            
            # Simulation d'envoi
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
            return False
    
    async def send_report(self, report_data: Dict[str, Any], recipients: List[str]) -> bool:
        """Envoie un rapport par email."""
        try:
            logger.info(f"REPORT NOTIFICATION: Analytics Report")
            logger.info(f"Recipients: {recipients}")
            logger.info(f"Report data keys: {list(report_data.keys())}")
            
            # Simulation d'envoi
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send report notification: {e}")
            return False
    
    async def send_recommendation(
        self, 
        recommendations: List[BusinessRecommendation], 
        recipients: List[str]
    ) -> bool:
        """Envoie des recommandations par email."""
        try:
            logger.info(f"RECOMMENDATIONS NOTIFICATION: {len(recommendations)} recommendations")
            logger.info(f"Recipients: {recipients}")
            
            for rec in recommendations:
                logger.info(f"- {rec.title} (Priority: {rec.priority})")
            
            # Simulation d'envoi
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send recommendations notification: {e}")
            return False
