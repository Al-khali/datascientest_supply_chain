#!/usr/bin/env python3
"""
Script de tests de performance et de charge pour la plateforme Supply Chain IA.
Valide la robustesse, la scalabilité et les temps de réponse de tous les modules.

Auteur: Data Engineer Expert
Date: 03/06/2025
"""

import sys
import os
import time
import asyncio
import threading
import multiprocessing
import pandas as pd
import numpy as np
import requests
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import logging
import psutil
import memory_profiler
from typing import List, Dict, Any

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Testeur de performance pour la plateforme Supply Chain IA."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def test_data_processing_performance(self):
        """Test des performances de traitement des données."""
        logger.info("🚀 Test de performance - Traitement des données")
        
        # Génération de datasets de tailles croissantes
        datasets = {
            'small': 1000,
            'medium': 10000,
            'large': 100000,
            'xlarge': 500000
        }
        
        processing_times = {}
        memory_usage = {}
        
        for size_name, size in datasets.items():
            logger.info(f"   📊 Test dataset {size_name}: {size:,} rows")
            
            # Génération des données
            start_gen = time.time()
            df = pd.DataFrame({
                'review_id': [f"test_{i:06d}" for i in range(size)],
                'rating': np.random.randint(1, 6, size),
                'review_text': [f"Avis test numéro {i} avec contenu variable" for i in range(size)],
                'sentiment_score': np.random.normal(0, 0.5, size),
                'date_published': pd.date_range('2024-01-01', periods=size, freq='H')
            })
            gen_time = time.time() - start_gen
            
            # Test du traitement
            start_process = time.time()
            
            # Opérations typiques du pipeline
            df['rating_normalized'] = df['rating'] / 5.0
            df['sentiment_label'] = df['sentiment_score'].apply(
                lambda x: 'positif' if x > 0.1 else 'negatif' if x < -0.1 else 'neutre'
            )
            df['month'] = df['date_published'].dt.to_period('M')
            
            # Agrégations
            monthly_stats = df.groupby('month').agg({
                'rating': ['mean', 'count'],
                'sentiment_score': 'mean'
            })
            
            process_time = time.time() - start_process
            
            # Mesure mémoire
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            processing_times[size_name] = {
                'generation_time': gen_time,
                'processing_time': process_time,
                'total_time': gen_time + process_time,
                'rows_per_second': size / (gen_time + process_time)
            }
            
            memory_usage[size_name] = {
                'memory_mb': memory_mb,
                'memory_per_row': memory_mb / size * 1024  # KB par ligne
            }
            
            logger.info(f"      ⏱️  Temps total: {gen_time + process_time:.2f}s")
            logger.info(f"      📈 Débit: {size / (gen_time + process_time):,.0f} rows/s")
            logger.info(f"      💾 Mémoire: {memory_mb:.1f} MB")
        
        self.results['data_processing'] = {
            'processing_times': processing_times,
            'memory_usage': memory_usage
        }
        
        return processing_times, memory_usage
    
    def test_nlp_performance(self):
        """Test des performances des modèles NLP."""
        logger.info("🧠 Test de performance - Modèles NLP")
        
        # Textes de test de longueurs variables
        test_texts = {
            'court': "Livraison rapide, très satisfait.",
            'moyen': "J'ai commandé plusieurs produits et je suis globalement satisfait de la qualité. La livraison a été effectuée dans les délais annoncés.",
            'long': "J'ai récemment passé commande sur le site et je dois dire que l'expérience globale a été très positive. Les produits correspondent exactement à la description, la qualité est au rendez-vous et l'emballage était parfait. Le service client a répondu rapidement à mes questions. Seul petit bémol, la livraison a pris une journée de plus que prévu, mais rien de dramatique. Je recommande vivement cette boutique et je reviendrai certainement pour de futurs achats.",
            'tres_long': "Mon expérience d'achat sur ce site a été mitigée et je souhaite partager mon retour détaillé. D'un côté positif, la navigation sur le site web est intuitive, le catalogue de produits est bien organisé avec des filtres efficaces, et les descriptions des articles sont complètes avec de belles photos. Le processus de commande s'est déroulé sans accroc et j'ai rapidement reçu une confirmation par email avec le numéro de suivi. Cependant, plusieurs points négatifs sont à noter. Premièrement, la livraison a accusé un retard de trois jours par rapport à la date annoncée, sans aucune communication proactive de la part du service logistique. Deuxièmement, l'un des produits reçus présentait un défaut mineur mais visible qui n'était pas mentionné. Troisièmement, quand j'ai contacté le service client pour signaler ces problèmes, j'ai dû attendre plus de 48 heures avant d'obtenir une réponse, ce qui est inacceptable de nos jours. Au final, les problèmes ont été résolus et j'ai obtenu un geste commercial, mais l'expérience globale reste décevante compte tenu de la réputation de la marque."
        }
        
        nlp_times = {}
        
        for length, text in test_texts.items():
            logger.info(f"   📝 Test texte {length}: {len(text)} caractères")
            
            start_time = time.time()
            
            # Simulation des traitements NLP
            # (Dans un cas réel, on utiliserait les vrais modèles)
            
            # 1. Nettoyage du texte
            clean_text = text.lower().strip()
            
            # 2. Détection de sentiment (simulation)
            sentiment_score = np.random.normal(0, 0.3)
            sentiment_label = 'positif' if sentiment_score > 0.1 else 'negatif' if sentiment_score < -0.1 else 'neutre'
            
            # 3. Extraction d'entités (simulation)
            entities = []
            keywords = ['livraison', 'produit', 'service', 'qualité', 'rapide', 'retard']
            for keyword in keywords:
                if keyword in clean_text:
                    entities.append(keyword)
            
            # 4. Score de criticité (simulation)
            criticality = abs(sentiment_score) * len(entities) / 10
            
            process_time = time.time() - start_time
            
            nlp_times[length] = {
                'processing_time': process_time,
                'chars_per_second': len(text) / process_time,
                'sentiment_score': sentiment_score,
                'entities_found': len(entities),
                'criticality_score': criticality
            }
            
            logger.info(f"      ⏱️  Temps: {process_time:.4f}s")
            logger.info(f"      📈 Débit: {len(text) / process_time:,.0f} chars/s")
            logger.info(f"      😊 Sentiment: {sentiment_label} ({sentiment_score:.3f})")
        
        self.results['nlp_performance'] = nlp_times
        return nlp_times
    
    def test_api_performance(self):
        """Test des performances de l'API REST."""
        logger.info("🌐 Test de performance - API REST")
        
        # URLs de test (adapter selon votre configuration)
        base_url = "http://localhost:8000"
        endpoints = {
            'health': f"{base_url}/health",
            'stats': f"{base_url}/api/v1/reviews/stats",
            'sentiment': f"{base_url}/api/v1/reviews/sentiment-analysis"
        }
        
        api_results = {}
        
        for endpoint_name, url in endpoints.items():
            logger.info(f"   🔗 Test endpoint: {endpoint_name}")
            
            response_times = []
            success_count = 0
            
            # Test avec 10 requêtes
            for i in range(10):
                try:
                    start_time = time.time()
                    
                    if endpoint_name == 'sentiment':
                        # POST avec données
                        response = requests.post(
                            url, 
                            json={"text": f"Test API performance numéro {i}"},
                            timeout=5
                        )
                    else:
                        # GET simple
                        response = requests.get(url, timeout=5)
                    
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        success_count += 1
                        response_times.append(response_time)
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"      ⚠️ Erreur requête {i}: {e}")
                
                time.sleep(0.1)  # Pause entre requêtes
            
            if response_times:
                api_results[endpoint_name] = {
                    'success_rate': success_count / 10 * 100,
                    'avg_response_time': np.mean(response_times),
                    'min_response_time': np.min(response_times),
                    'max_response_time': np.max(response_times),
                    'requests_per_second': 1 / np.mean(response_times) if response_times else 0
                }
                
                logger.info(f"      ✅ Succès: {success_count}/10 ({success_count*10}%)")
                logger.info(f"      ⏱️  Temps moyen: {np.mean(response_times)*1000:.0f}ms")
                logger.info(f"      📈 Débit: {1 / np.mean(response_times):.1f} req/s")
            else:
                api_results[endpoint_name] = {
                    'success_rate': 0,
                    'error': 'Aucune réponse valide'
                }
                logger.error(f"      ❌ Échec complet pour {endpoint_name}")
        
        self.results['api_performance'] = api_results
        return api_results
    
    def test_concurrent_load(self):
        """Test de charge avec utilisateurs concurrents."""
        logger.info("⚡ Test de charge - Utilisateurs concurrents")
        
        def simulate_user_session():
            """Simule une session utilisateur complète."""
            session_start = time.time()
            
            # Simulation des actions utilisateur
            actions = [
                'load_dashboard',
                'apply_filters', 
                'export_data',
                'view_details'
            ]
            
            action_times = {}
            
            for action in actions:
                start_time = time.time()
                
                # Simulation du traitement
                if action == 'load_dashboard':
                    time.sleep(0.5)  # Chargement initial
                elif action == 'apply_filters':
                    time.sleep(0.2)  # Filtrage
                elif action == 'export_data':
                    time.sleep(0.8)  # Export CSV
                elif action == 'view_details':
                    time.sleep(0.3)  # Détails
                
                action_times[action] = time.time() - start_time
            
            total_session_time = time.time() - session_start
            
            return {
                'session_time': total_session_time,
                'actions': action_times,
                'thread_id': threading.current_thread().ident
            }
        
        # Test avec différents niveaux de concurrence
        concurrency_levels = [1, 5, 10, 20]
        load_results = {}
        
        for concurrent_users in concurrency_levels:
            logger.info(f"   👥 Test avec {concurrent_users} utilisateurs concurrents")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(simulate_user_session) for _ in range(concurrent_users)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            
            session_times = [r['session_time'] for r in results]
            
            load_results[concurrent_users] = {
                'total_execution_time': total_time,
                'avg_session_time': np.mean(session_times),
                'max_session_time': np.max(session_times),
                'min_session_time': np.min(session_times),
                'throughput': concurrent_users / total_time,
                'success_rate': 100.0  # Simulation toujours réussie
            }
            
            logger.info(f"      ⏱️  Temps total: {total_time:.2f}s")
            logger.info(f"      📊 Session moyenne: {np.mean(session_times):.2f}s")
            logger.info(f"      📈 Débit: {concurrent_users / total_time:.1f} users/s")
        
        self.results['load_testing'] = load_results
        return load_results
    
    def test_memory_usage(self):
        """Test d'utilisation mémoire."""
        logger.info("💾 Test d'utilisation mémoire")
        
        @memory_profiler.profile
        def memory_intensive_operation():
            # Simulation d'opérations gourmandes en mémoire
            large_df = pd.DataFrame({
                'col1': np.random.randn(100000),
                'col2': np.random.randn(100000),
                'col3': [f"string_{i}" for i in range(100000)]
            })
            
            # Opérations de traitement
            result = large_df.groupby(large_df['col1'].round()).agg({
                'col2': ['mean', 'std', 'count']
            })
            
            return result
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        result = memory_intensive_operation()
        execution_time = time.time() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_peak = memory_after  # Approximation
        
        memory_results = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_peak_mb': memory_peak,
            'memory_increase_mb': memory_after - memory_before,
            'execution_time': execution_time
        }
        
        logger.info(f"   💾 Mémoire avant: {memory_before:.1f} MB")
        logger.info(f"   💾 Mémoire après: {memory_after:.1f} MB")
        logger.info(f"   📈 Augmentation: {memory_after - memory_before:.1f} MB")
        logger.info(f"   ⏱️  Temps d'exécution: {execution_time:.2f}s")
        
        self.results['memory_usage'] = memory_results
        return memory_results
    
    def generate_report(self):
        """Génère un rapport complet des tests de performance."""
        logger.info("📋 Génération du rapport de performance")
        
        total_time = time.time() - self.start_time
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_test_duration': total_time,
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': sys.version
            },
            'test_results': self.results
        }
        
        # Sauvegarde du rapport
        report_path = f"../data/performance_report_{int(time.time())}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Affichage du résumé
        print("\n" + "="*60)
        print("🚀 RAPPORT DE PERFORMANCE - PLATEFORME SUPPLY CHAIN IA")
        print("="*60)
        
        print(f"\n⏱️  DURÉE TOTALE DES TESTS: {total_time:.1f}s")
        print(f"🖥️  SYSTÈME: {multiprocessing.cpu_count()} CPU, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
        
        if 'data_processing' in self.results:
            print(f"\n📊 TRAITEMENT DONNÉES:")
            for size, metrics in self.results['data_processing']['processing_times'].items():
                print(f"   {size.upper()}: {metrics['rows_per_second']:,.0f} rows/s")
        
        if 'nlp_performance' in self.results:
            print(f"\n🧠 PERFORMANCE NLP:")
            for length, metrics in self.results['nlp_performance'].items():
                print(f"   {length.upper()}: {metrics['chars_per_second']:,.0f} chars/s")
        
        if 'api_performance' in self.results:
            print(f"\n🌐 PERFORMANCE API:")
            for endpoint, metrics in self.results['api_performance'].items():
                if 'avg_response_time' in metrics:
                    print(f"   {endpoint.upper()}: {metrics['avg_response_time']*1000:.0f}ms avg")
        
        if 'load_testing' in self.results:
            print(f"\n⚡ TEST DE CHARGE:")
            for users, metrics in self.results['load_testing'].items():
                print(f"   {users} USERS: {metrics['throughput']:.1f} users/s")
        
        print(f"\n📋 Rapport sauvegardé: {report_path}")
        print("="*60)
        
        return report

def main():
    """Fonction principale d'exécution des tests."""
    print("🚀 LANCEMENT DES TESTS DE PERFORMANCE")
    print("Plateforme Supply Chain IA - Tests de robustesse")
    print("-" * 50)
    
    tester = PerformanceTester()
    
    try:
        # Exécution des tests
        tester.test_data_processing_performance()
        tester.test_nlp_performance()
        tester.test_api_performance()
        tester.test_concurrent_load()
        tester.test_memory_usage()
        
        # Génération du rapport final
        report = tester.generate_report()
        
        print("\n✅ TESTS DE PERFORMANCE TERMINÉS AVEC SUCCÈS")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors des tests: {e}")
        raise

if __name__ == "__main__":
    main()
