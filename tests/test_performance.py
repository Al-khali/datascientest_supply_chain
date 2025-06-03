"""
Tests de performance pour la plateforme Supply Chain AI
Mesure les temps de réponse, throughput et utilisation des ressources

Auteur: Data Engineer / Data Scientist Expert
Date: 03/06/2025
"""

import pytest
import asyncio
import time
import psutil
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import statistics

# Configuration des tests
API_BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"
TEST_DURATION = 60  # secondes
CONCURRENT_USERS = 10

class PerformanceTestSuite:
    """Suite de tests de performance pour la plateforme."""
    
    def __init__(self):
        self.results = []
        self.system_metrics = []
    
    def measure_system_resources(self) -> Dict:
        """Mesure l'utilisation des ressources système."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': time.time()
        }
    
    def test_api_response_time(self, endpoint: str, method: str = "GET", data: Dict = None) -> float:
        """Mesure le temps de réponse d'un endpoint API."""
        start_time = time.time()
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
            elif method == "POST":
                response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            self.results.append({
                'endpoint': endpoint,
                'method': method,
                'response_time': response_time,
                'status_code': response.status_code,
                'success': response.status_code < 400,
                'timestamp': start_time
            })
            
            return response_time
            
        except Exception as e:
            end_time = time.time()
            self.results.append({
                'endpoint': endpoint,
                'method': method,
                'response_time': end_time - start_time,
                'status_code': 0,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
            return end_time - start_time
    
    def load_test_endpoint(self, endpoint: str, concurrent_users: int = 10, duration: int = 60):
        """Test de charge sur un endpoint."""
        print(f"🚀 Test de charge: {endpoint} ({concurrent_users} utilisateurs, {duration}s)")
        
        def worker():
            end_time = time.time() + duration
            while time.time() < end_time:
                self.test_api_response_time(endpoint)
                time.sleep(0.1)  # Petit délai entre les requêtes
        
        # Lancement des threads
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]
            
            # Surveillance des ressources système
            start_time = time.time()
            while time.time() - start_time < duration:
                self.system_metrics.append(self.measure_system_resources())
                time.sleep(5)
            
            # Attendre la fin de tous les threads
            for future in futures:
                future.result()
    
    def generate_performance_report(self) -> Dict:
        """Génère un rapport de performance détaillé."""
        if not self.results:
            return {"error": "Aucun résultat de test disponible"}
        
        df = pd.DataFrame(self.results)
        
        # Statistiques globales
        successful_requests = df[df['success'] == True]
        total_requests = len(df)
        success_rate = len(successful_requests) / total_requests * 100
        
        # Temps de réponse
        response_times = successful_requests['response_time'].tolist()
        
        report = {
            'summary': {
                'total_requests': total_requests,
                'successful_requests': len(successful_requests),
                'success_rate': round(success_rate, 2),
                'test_duration': round(df['timestamp'].max() - df['timestamp'].min(), 2)
            },
            'response_times': {
                'mean': round(statistics.mean(response_times), 3) if response_times else 0,
                'median': round(statistics.median(response_times), 3) if response_times else 0,
                'min': round(min(response_times), 3) if response_times else 0,
                'max': round(max(response_times), 3) if response_times else 0,
                'p95': round(sorted(response_times)[int(len(response_times) * 0.95)], 3) if response_times else 0,
                'p99': round(sorted(response_times)[int(len(response_times) * 0.99)], 3) if response_times else 0
            },
            'throughput': {
                'requests_per_second': round(total_requests / (df['timestamp'].max() - df['timestamp'].min()), 2) if total_requests > 1 else 0
            }
        }
        
        # Métriques système
        if self.system_metrics:
            metrics_df = pd.DataFrame(self.system_metrics)
            report['system_resources'] = {
                'avg_cpu_percent': round(metrics_df['cpu_percent'].mean(), 2),
                'max_cpu_percent': round(metrics_df['cpu_percent'].max(), 2),
                'avg_memory_percent': round(metrics_df['memory_percent'].mean(), 2),
                'max_memory_percent': round(metrics_df['memory_percent'].max(), 2)
            }
        
        # Analyse par endpoint
        endpoint_stats = {}
        for endpoint in df['endpoint'].unique():
            endpoint_data = df[df['endpoint'] == endpoint]
            endpoint_success = endpoint_data[endpoint_data['success'] == True]
            
            if len(endpoint_success) > 0:
                endpoint_stats[endpoint] = {
                    'requests': len(endpoint_data),
                    'success_rate': round(len(endpoint_success) / len(endpoint_data) * 100, 2),
                    'avg_response_time': round(endpoint_success['response_time'].mean(), 3),
                    'p95_response_time': round(endpoint_success['response_time'].quantile(0.95), 3)
                }
        
        report['endpoints'] = endpoint_stats
        
        return report

# Tests pytest
class TestAPIPerformance:
    """Tests de performance pour l'API."""
    
    @pytest.fixture(scope="class")
    def performance_suite(self):
        return PerformanceTestSuite()
    
    def test_health_endpoint_response_time(self, performance_suite):
        """Test du temps de réponse de l'endpoint health."""
        response_time = performance_suite.test_api_response_time("/health")
        assert response_time < 0.1, f"Temps de réponse trop élevé: {response_time}s"
    
    def test_kpis_endpoint_response_time(self, performance_suite):
        """Test du temps de réponse de l'endpoint KPIs."""
        response_time = performance_suite.test_api_response_time("/api/v1/kpis")
        assert response_time < 0.5, f"Temps de réponse trop élevé: {response_time}s"
    
    def test_sentiment_analysis_response_time(self, performance_suite):
        """Test du temps de réponse de l'analyse de sentiment."""
        test_data = {"text": "Ce produit est fantastique!"}
        response_time = performance_suite.test_api_response_time("/api/v1/analyze", "POST", test_data)
        assert response_time < 2.0, f"Temps de réponse trop élevé: {response_time}s"
    
    @pytest.mark.slow
    def test_api_load_test(self, performance_suite):
        """Test de charge sur l'API."""
        performance_suite.load_test_endpoint("/health", concurrent_users=5, duration=30)
        
        report = performance_suite.generate_performance_report()
        
        # Assertions sur les performances
        assert report['summary']['success_rate'] > 95, "Taux de succès insuffisant"
        assert report['response_times']['p95'] < 1.0, "95e percentile trop élevé"
        assert report['throughput']['requests_per_second'] > 10, "Throughput insuffisant"

def run_performance_benchmark():
    """Lance un benchmark complet de performance."""
    print("🚀 Démarrage du benchmark de performance Supply Chain AI")
    print("=" * 60)
    
    suite = PerformanceTestSuite()
    
    # Tests des endpoints principaux
    endpoints_to_test = [
        "/health",
        "/api/v1/kpis",
        "/api/v1/reviews/stats"
    ]
    
    print("📊 Test des temps de réponse individuels...")
    for endpoint in endpoints_to_test:
        response_time = suite.test_api_response_time(endpoint)
        print(f"  {endpoint}: {response_time:.3f}s")
    
    # Test de charge
    print("\n🔥 Test de charge (30 secondes)...")
    suite.load_test_endpoint("/health", concurrent_users=CONCURRENT_USERS, duration=30)
    
    # Génération du rapport
    report = suite.generate_performance_report()
    
    print("\n📈 RAPPORT DE PERFORMANCE")
    print("=" * 60)
    print(f"Requêtes totales: {report['summary']['total_requests']}")
    print(f"Taux de succès: {report['summary']['success_rate']}%")
    print(f"Throughput: {report['throughput']['requests_per_second']} req/s")
    print(f"Temps de réponse moyen: {report['response_times']['mean']}s")
    print(f"95e percentile: {report['response_times']['p95']}s")
    
    if 'system_resources' in report:
        print(f"CPU moyen: {report['system_resources']['avg_cpu_percent']}%")
        print(f"Mémoire moyenne: {report['system_resources']['avg_memory_percent']}%")
    
    print("\n📋 Performance par endpoint:")
    for endpoint, stats in report['endpoints'].items():
        print(f"  {endpoint}: {stats['avg_response_time']}s (success: {stats['success_rate']}%)")
    
    # Sauvegarde du rapport
    import json
    with open('performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Rapport détaillé sauvegardé: performance_report.json")
    return report

if __name__ == "__main__":
    run_performance_benchmark()
