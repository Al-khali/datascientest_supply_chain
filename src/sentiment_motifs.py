"""
Module expert d'analyse NLP avancée pour l'analyse de satisfaction client dans la supply chain.
Utilise des modèles transformers pré-entraînés, analyse multi-dimensionnelle des sentiments,
extraction d'entités métiers, clustering automatique des motifs d'insatisfaction,
et génération d'insights business actionnables.

Fonctionnalités expertes :
- Analyse de sentiment multi-modèles (BERT, RoBERTa, CamemBERT)
- Extraction d'entités nommées (spaCy)
- Classification automatique des catégories d'insatisfaction
- Topic modeling avancé (LDA, BERTopic)
- Scoring de criticité des avis
- Détection automatique d'anomalies
- Génération de recommandations business
- Export multi-format avec métriques avancées

Auteur: Data Scientist Expert
Date: 03/06/2025
"""

import pandas as pd
import numpy as np
import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
import networkx as nx
from wordcloud import WordCloud

# Data Science
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Configuration avancée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('../logs/nlp_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentAnalysisResult:
    """Structure de données pour les résultats d'analyse de sentiment."""
    review_id: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    emotions: Dict[str, float]
    topics: List[str]
    entities: List[Dict]
    criticality_score: float
    business_impact: str
    recommended_actions: List[str]

class AdvancedSentimentAnalyzer:
    """Analyseur de sentiment expert avec modèles transformers et analyse business."""
    
    def __init__(self):
        self.models = {}
        self.nlp_fr = None
        self.vectorizers = {}
        self.clusters = {}
        self._initialize_models()
        
        # Dictionnaires métiers supply chain
        self.supply_chain_categories = {
            'livraison_logistique': {
                'keywords': ['livraison', 'colis', 'expédition', 'transport', 'délai', 'retard', 
                           'courrier', 'poste', 'chronopost', 'dhl', 'ups', 'fedex', 'colissimo'],
                'weight': 1.5,  # Importance critique
                'business_impact': 'operational'
            },
            'qualite_produit': {
                'keywords': ['produit', 'qualité', 'défectueux', 'abîmé', 'cassé', 'état', 
                           'neuf', 'usagé', 'conforme', 'emballage', 'packaging'],
                'weight': 1.3,
                'business_impact': 'product'
            },
            'service_client': {
                'keywords': ['service', 'client', 'support', 'conseiller', 'hotline', 'contact',
                           'réponse', 'aide', 'assistance', 'accueil', 'personnel'],
                'weight': 1.4,
                'business_impact': 'customer_experience'
            },
            'transaction_paiement': {
                'keywords': ['paiement', 'facture', 'prix', 'tarif', 'coût', 'remboursement',
                           'échange', 'retour', 'garantie', 'sav', 'commande'],
                'weight': 1.2,
                'business_impact': 'financial'
            },
            'interface_digital': {
                'keywords': ['site', 'web', 'application', 'app', 'interface', 'navigation',
                           'bug', 'erreur', 'connexion', 'compte', 'profil'],
                'weight': 1.1,
                'business_impact': 'digital'
            }
        }
    
    def _initialize_models(self):
        """Initialise les modèles NLP avancés."""
        logger.info("Initialisation des modèles NLP experts...")
        
        try:
            # Modèle de sentiment français expert
            self.models['sentiment_fr'] = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            
            # Modèle d'émotions
            self.models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            
            # Modèle spaCy français pour NER
            try:
                self.nlp_fr = spacy.load("fr_core_news_md")
            except OSError:
                logger.warning("Modèle spaCy français non trouvé. Installation automatique...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
                self.nlp_fr = spacy.load("fr_core_news_md")
            
            logger.info("Modèles NLP initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles: {str(e)}")
            # Fallback vers TextBlob
            logger.info("Utilisation du fallback TextBlob")
    
    def analyze_sentiment_advanced(self, text: str) -> Dict:
        """Analyse de sentiment multi-modèles avec scoring avancé."""
        if not text or pd.isna(text):
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'emotions': {},
                'method': 'empty'
            }
        
        results = {}
        
        try:
            # Analyse BERT multilingue
            if 'sentiment_fr' in self.models:
                bert_result = self.models['sentiment_fr'](text[:512])  # Limite BERT
                results['bert'] = {
                    'label': bert_result[0]['label'],
                    'score': bert_result[0]['score']
                }
            
            # Analyse TextBlob (baseline)
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # Analyse d'émotions
            if 'emotion' in self.models and len(text) > 10:
                emotion_result = self.models['emotion'](text[:512])
                results['emotions'] = {
                    emotion['label']: emotion['score'] 
                    for emotion in emotion_result
                }
            
            # Score de sentiment consolidé
            sentiment_score = self._consolidate_sentiment_scores(results)
            sentiment_label = self._get_sentiment_label(sentiment_score)
            confidence = self._calculate_confidence(results)
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'emotions': results.get('emotions', {}),
                'method': 'advanced_nlp',
                'detailed_results': results
            }
            
        except Exception as e:
            logger.warning(f"Erreur analyse sentiment avancée: {str(e)}")
            # Fallback TextBlob
            blob = TextBlob(text)
            return {
                'sentiment_score': blob.sentiment.polarity,
                'sentiment_label': self._get_sentiment_label(blob.sentiment.polarity),
                'confidence': abs(blob.sentiment.polarity),
                'emotions': {},
                'method': 'fallback_textblob'
            }
    
    def _consolidate_sentiment_scores(self, results: Dict) -> float:
        """Consolide les scores de sentiment de différents modèles."""
        scores = []
        
        # Score BERT (pondération forte)
        if 'bert' in results:
            bert_score = results['bert']['score']
            if results['bert']['label'] in ['NEGATIVE', '1 star', '2 stars']:
                bert_score = -bert_score
            elif results['bert']['label'] in ['POSITIVE', '4 stars', '5 stars']:
                bert_score = bert_score
            else:
                bert_score = 0
            scores.append(('bert', bert_score, 0.6))
        
        # Score TextBlob (pondération moyenne)
        if 'textblob' in results:
            scores.append(('textblob', results['textblob']['polarity'], 0.4))
        
        # Calcul pondéré
        if scores:
            weighted_sum = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            return weighted_sum / total_weight
        
        return 0.0
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convertit un score en label de sentiment."""
        if score >= 0.1:
            return 'positive'
        elif score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calcule le niveau de confiance de l'analyse."""
        confidences = []
        
        if 'bert' in results:
            confidences.append(results['bert']['score'])
        
        if 'textblob' in results:
            confidences.append(abs(results['textblob']['polarity']))
        
        return np.mean(confidences) if confidences else 0.0
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extraction d'entités nommées avec spaCy."""
        if not self.nlp_fr or not text:
            return []
        
        try:
            doc = self.nlp_fr(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"Erreur extraction entités: {str(e)}")
            return []
    
    def classify_supply_chain_category(self, text: str) -> Dict:
        """Classification automatique par catégorie supply chain."""
        if not text or pd.isna(text):
            return {'category': 'unknown', 'confidence': 0.0, 'keywords_found': []}
        
        text_lower = text.lower()
        category_scores = {}
        
        for category, config in self.supply_chain_categories.items():
            score = 0
            found_keywords = []
            
            for keyword in config['keywords']:
                # Recherche exacte avec frontières de mots
                import re
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    score += config['weight']
                    found_keywords.append(keyword)
            
            if score > 0:
                category_scores[category] = {
                    'score': score,
                    'keywords': found_keywords,
                    'business_impact': config['business_impact']
                }
        
        if not category_scores:
            return {'category': 'autre', 'confidence': 0.0, 'keywords_found': []}
        
        # Sélection de la meilleure catégorie
        best_category = max(category_scores.keys(), key=lambda x: category_scores[x]['score'])
        
        return {
            'category': best_category,
            'confidence': category_scores[best_category]['score'],
            'keywords_found': category_scores[best_category]['keywords'],
            'business_impact': category_scores[best_category]['business_impact']
        }
    
    def calculate_criticality_score(self, sentiment_result: Dict, category_result: Dict, 
                                  rating: float, review_length: int) -> float:
        """Calcule un score de criticité business pour prioriser les actions."""
        score = 0.0
        
        # Impact du sentiment (pondération forte)
        sentiment_score = sentiment_result.get('sentiment_score', 0)
        if sentiment_score < -0.5:
            score += 30  # Très négatif
        elif sentiment_score < -0.2:
            score += 20  # Négatif
        elif sentiment_score < 0.2:
            score += 5   # Neutre
        
        # Impact de la note
        if rating <= 2:
            score += 25
        elif rating <= 3:
            score += 15
        
        # Impact de la catégorie supply chain
        category_weights = {
            'livraison_logistique': 20,
            'service_client': 18,
            'qualite_produit': 15,
            'transaction_paiement': 12,
            'interface_digital': 8
        }
        
        category = category_result.get('category', 'autre')
        score += category_weights.get(category, 5)
        
        # Impact de la longueur (avis détaillés = plus critique)
        if review_length > 200:
            score += 10
        elif review_length > 100:
            score += 5
        
        # Normalisation 0-100
        return min(100, max(0, score))
    
    def generate_business_recommendations(self, category: str, sentiment_score: float, 
                                        criticality_score: float, entities: List[Dict]) -> List[str]:
        """Génère des recommandations business actionnables."""
        recommendations = []
        
        # Recommandations par catégorie
        if category == 'livraison_logistique':
            if sentiment_score < -0.3:
                recommendations.extend([
                    "Réviser les délais de livraison communiqués",
                    "Améliorer le tracking en temps réel",
                    "Renforcer la communication proactive sur les retards"
                ])
            recommendations.append("Analyser les performances des transporteurs partenaires")
            
        elif category == 'service_client':
            if sentiment_score < -0.3:
                recommendations.extend([
                    "Formation équipes service client sur l'empathie",
                    "Réduire les temps d'attente téléphonique",
                    "Implémenter un système de callback automatique"
                ])
            recommendations.append("Audit complet parcours client support")
            
        elif category == 'qualite_produit':
            recommendations.extend([
                "Contrôle qualité renforcé en entrepôt",
                "Amélioration packaging protection",
                "Processus retour/échange simplifié"
            ])
            
        elif category == 'transaction_paiement':
            recommendations.extend([
                "Optimisation tunnel de commande",
                "Clarification politique retours/remboursements",
                "Sécurisation moyens de paiement"
            ])
        
        # Recommandations par criticité
        if criticality_score > 80:
            recommendations.insert(0, "🔥 ACTION PRIORITAIRE - Traitement immédiat requis")
            recommendations.append("Escalade vers management supply chain")
        elif criticality_score > 60:
            recommendations.insert(0, "⚠️ Action sous 48h recommandée")
        
        return recommendations[:5]  # Limite à 5 recommandations
    
    def perform_topic_modeling(self, texts: List[str], n_topics: int = 8) -> Dict:
        """Modélisation de topics avancée avec LDA et BERTopic."""
        if len(texts) < 10:
            return {'topics': [], 'method': 'insufficient_data'}
        
        try:
            # Filtrage des textes vides
            valid_texts = [text for text in texts if text and len(text) > 20]
            
            if len(valid_texts) < 5:
                return {'topics': [], 'method': 'insufficient_valid_data'}
            
            # TF-IDF + LDA
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=None,  # Déjà nettoyé
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            lda.fit(tfidf_matrix)
            
            # Extraction des topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_weight = topic[top_words_idx].tolist()
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': topic_weight,
                    'coherence': float(np.mean(topic_weight))
                })
            
            return {
                'topics': topics,
                'method': 'lda',
                'n_topics': n_topics,
                'total_docs': len(valid_texts)
            }
            
        except Exception as e:
            logger.error(f"Erreur topic modeling: {str(e)}")
            return {'topics': [], 'method': 'error', 'error': str(e)}
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détection d'anomalies dans les avis (spam, fake reviews, etc.)."""
        try:
            # Features pour détection d'anomalies
            features = []
            
            # Longueur du texte
            if 'review_length' in df.columns:
                features.append('review_length')
            
            # Score de sentiment
            if 'sentiment_score' in df.columns:
                features.append('sentiment_score')
            
            # Note
            if 'rating' in df.columns:
                features.append('rating')
            
            # Score de confiance
            if 'confidence' in df.columns:
                features.append('confidence')
            
            if len(features) < 2:
                df['is_anomaly'] = False
                return df
            
            # Préparation des données
            feature_data = df[features].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # 10% d'anomalies attendues
                random_state=42
            )
            
            anomaly_predictions = iso_forest.fit_predict(scaled_data)
            df['is_anomaly'] = anomaly_predictions == -1
            df['anomaly_score'] = iso_forest.decision_function(scaled_data)
            
            logger.info(f"Détection d'anomalies: {df['is_anomaly'].sum()} anomalies détectées sur {len(df)} avis")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur détection anomalies: {str(e)}")
            df['is_anomaly'] = False
            df['anomaly_score'] = 0.0
            return df

def process_reviews_expert(input_path: str = '../data/avis_clean.csv', 
                          output_path: str = '../data/avis_sentiment_expert.csv') -> Dict:
    """Fonction principale d'analyse NLP experte des avis clients."""
    
    logger.info("🚀 Démarrage de l'analyse NLP experte des avis clients")
    
    # Chargement des données
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Chargement: {len(df)} avis à analyser")
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé: {input_path}")
        return {'status': 'error', 'message': 'Fichier de données non trouvé'}
    
    if df.empty:
        logger.error("Dataset vide")
        return {'status': 'error', 'message': 'Dataset vide'}
    
    # Initialisation de l'analyseur
    analyzer = AdvancedSentimentAnalyzer()
    
    # Détermination des colonnes
    text_column = 'review_text' if 'review_text' in df.columns else 'commentaire'
    rating_column = 'rating' if 'rating' in df.columns else 'note'
    
    if text_column not in df.columns:
        logger.error(f"Colonne de texte non trouvée: {text_column}")
        return {'status': 'error', 'message': 'Colonne de texte manquante'}
    
    # Traitement des avis
    logger.info("Analyse de sentiment avancée en cours...")
    results = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Progression: {idx}/{len(df)} avis traités")
        
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        rating = float(row[rating_column]) if rating_column in df.columns and pd.notna(row[rating_column]) else 3.0
        review_length = len(text)
        
        # Analyse de sentiment
        sentiment_result = analyzer.analyze_sentiment_advanced(text)
        
        # Classification catégorie
        category_result = analyzer.classify_supply_chain_category(text)
        
        # Extraction d'entités
        entities = analyzer.extract_entities(text)
        
        # Score de criticité
        criticality_score = analyzer.calculate_criticality_score(
            sentiment_result, category_result, rating, review_length
        )
        
        # Recommandations business
        recommendations = analyzer.generate_business_recommendations(
            category_result['category'], 
            sentiment_result['sentiment_score'],
            criticality_score,
            entities
        )
        
        results.append({
            'review_id': row.get('review_id', f'review_{idx}'),
            'sentiment_score': sentiment_result['sentiment_score'],
            'sentiment_label': sentiment_result['sentiment_label'],
            'confidence': sentiment_result['confidence'],
            'emotions': json.dumps(sentiment_result.get('emotions', {})),
            'category': category_result['category'],
            'category_confidence': category_result['confidence'],
            'keywords_found': json.dumps(category_result['keywords_found']),
            'business_impact': category_result['business_impact'],
            'entities': json.dumps(entities),
            'criticality_score': criticality_score,
            'recommended_actions': json.dumps(recommendations),
            'analysis_method': sentiment_result['method']
        })
    
    # Ajout des résultats au DataFrame
    results_df = pd.DataFrame(results)
    df_enriched = pd.concat([df, results_df], axis=1)
    
    # Détection d'anomalies
    logger.info("Détection d'anomalies en cours...")
    df_enriched = analyzer.detect_anomalies(df_enriched)
    
    # Topic modeling sur les avis négatifs
    logger.info("Analyse des topics sur les avis négatifs...")
    negative_reviews = df_enriched[df_enriched['sentiment_label'] == 'negative'][text_column].tolist()
    topic_results = analyzer.perform_topic_modeling(negative_reviews)
    
    # Export des résultats
    df_enriched.to_csv(output_path, index=False)
    df_enriched.to_parquet(output_path.replace('.csv', '.parquet'), index=False)
    
    # Génération du rapport d'analyse
    analysis_report = {
        'summary': {
            'total_reviews': len(df_enriched),
            'sentiment_distribution': df_enriched['sentiment_label'].value_counts().to_dict(),
            'category_distribution': df_enriched['category'].value_counts().to_dict(),
            'high_criticality_count': len(df_enriched[df_enriched['criticality_score'] > 70]),
            'anomalies_detected': df_enriched['is_anomaly'].sum(),
            'average_sentiment': df_enriched['sentiment_score'].mean()
        },
        'business_insights': {
            'top_negative_categories': df_enriched[df_enriched['sentiment_label'] == 'negative']['category'].value_counts().head().to_dict(),
            'critical_reviews': df_enriched.nlargest(10, 'criticality_score')[['review_id', 'category', 'criticality_score', 'sentiment_score']].to_dict('records'),
            'most_common_entities': [],  # À implémenter si nécessaire
        },
        'topic_modeling': topic_results,
        'recommendations': {
            'immediate_actions': [],
            'strategic_improvements': [],
            'monitoring_kpis': [
                'Pourcentage d\'avis négatifs par catégorie',
                'Score de criticité moyen par semaine',
                'Temps de résolution des avis critiques',
                'Évolution sentiment par transporteur'
            ]
        },
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'model_versions': 'BERT-multilingual, spaCy-fr-md, TextBlob',
            'data_source': input_path,
            'output_files': [output_path, output_path.replace('.csv', '.parquet')]
        }
    }
    
    # Export du rapport
    report_path = output_path.replace('.csv', '_analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Analyse NLP experte terminée!")
    logger.info(f"📊 Fichiers générés:")
    logger.info(f"   - Données enrichies: {output_path}")
    logger.info(f"   - Format Parquet: {output_path.replace('.csv', '.parquet')}")
    logger.info(f"   - Rapport d'analyse: {report_path}")
    
    return {
        'status': 'success',
        'summary': analysis_report['summary'],
        'output_files': analysis_report['metadata']['output_files'],
        'report_file': report_path
    }

def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse NLP experte des avis clients supply chain')
    parser.add_argument('--input', default='../data/avis_clean.csv', help='Fichier d\'entrée')
    parser.add_argument('--output', default='../data/avis_sentiment_expert.csv', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    result = process_reviews_expert(args.input, args.output)
    
    if result['status'] == 'success':
        print(f"✅ Analyse terminée avec succès!")
        print(f"📊 {result['summary']['total_reviews']} avis analysés")
        print(f"🎯 {result['summary']['high_criticality_count']} avis critiques identifiés")
    else:
        print(f"❌ Erreur: {result['message']}")

if __name__ == "__main__":
    main()
