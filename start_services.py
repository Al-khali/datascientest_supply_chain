#!/usr/bin/env python3
"""
Script de démarrage automatisé pour la plateforme
Analyse de Satisfaction Client Supply Chain

Auteur: Data Engineer / Data Scientist Expert
Date: 03/06/2025
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path
from typing import List, Dict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Gestionnaire de services pour la plateforme."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services: Dict[str, subprocess.Popen] = {}
        self.running = True
        
        # Configuration des services
        self.service_config = {
            'api': {
                'command': [sys.executable, '-m', 'uvicorn', 'src.api:app', '--reload', '--host', '0.0.0.0', '--port', '8000'],
                'description': 'API FastAPI',
                'url': 'http://localhost:8000/docs',
                'healthcheck': 'http://localhost:8000/health'
            },
            'dashboard': {
                'command': [sys.executable, '-m', 'streamlit', 'run', 'src/dashboard_expert.py', '--server.port', '8501'],
                'description': 'Dashboard Streamlit Expert',
                'url': 'http://localhost:8501',
                'healthcheck': None
            },
            'basic_dashboard': {
                'command': [sys.executable, '-m', 'streamlit', 'run', 'src/dashboard.py', '--server.port', '8502'],
                'description': 'Dashboard Streamlit Basique',
                'url': 'http://localhost:8502',
                'healthcheck': None
            }
        }
    
    def check_port_available(self, port: int) -> bool:
        """Vérifie si un port est disponible."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def start_service(self, service_name: str) -> bool:
        """Démarre un service spécifique."""
        config = self.service_config.get(service_name)
        if not config:
            logger.error("❌ Service inconnu: %s", service_name)
            return False
        
        try:
            # Vérification des ports
            if service_name == 'api' and not self.check_port_available(8000):
                logger.warning("⚠️  Port 8000 occupé, l'API pourrait ne pas démarrer")
            elif service_name == 'dashboard' and not self.check_port_available(8501):
                logger.warning("⚠️  Port 8501 occupé, le dashboard pourrait ne pas démarrer")
            elif service_name == 'basic_dashboard' and not self.check_port_available(8502):
                logger.warning("⚠️  Port 8502 occupé, le dashboard basique pourrait ne pas démarrer")
            
            # Démarrage du service
            process = subprocess.Popen(
                config['command'],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.services[service_name] = process
            logger.info("✅ Service démarré: %s (PID: %d)", config['description'], process.pid)
            logger.info("🌐 URL: %s", config['url'])
            
            return True
            
        except Exception as e:
            logger.error("❌ Erreur démarrage %s: %s", service_name, e)
            return False
    
    def stop_service(self, service_name: str):
        """Arrête un service spécifique."""
        if service_name in self.services:
            process = self.services[service_name]
            process.terminate()
            try:
                process.wait(timeout=5)
                logger.info("✅ Service arrêté: %s", service_name)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("⚠️  Service forcé d'arrêter: %s", service_name)
            del self.services[service_name]
    
    def stop_all_services(self):
        """Arrête tous les services."""
        logger.info("🛑 Arrêt de tous les services...")
        for service_name in list(self.services.keys()):
            self.stop_service(service_name)
        self.running = False
    
    def signal_handler(self, signum, frame):
        """Gestionnaire de signaux pour arrêt propre."""
        logger.info("📡 Signal reçu (%d), arrêt en cours...", signum)
        self.stop_all_services()
        sys.exit(0)
    
    def monitor_services(self):
        """Surveille l'état des services."""
        while self.running:
            for service_name, process in list(self.services.items()):
                if process.poll() is not None:
                    logger.error("❌ Service arrêté inopinément: %s", service_name)
                    # Optionnel: redémarrage automatique
                    # self.start_service(service_name)
            time.sleep(5)
    
    def display_status(self):
        """Affiche l'état des services."""
        print("\n" + "="*60)
        print("🚀 PLATEFORME SUPPLY CHAIN SATISFACTION CLIENT")
        print("="*60)
        
        for service_name, config in self.service_config.items():
            if service_name in self.services:
                process = self.services[service_name]
                status = "🟢 ACTIF" if process.poll() is None else "🔴 ARRÊTÉ"
                print(f"{config['description']:<25} {status:<10} {config['url']}")
            else:
                print(f"{config['description']:<25} {'⚪ ARRÊTÉ':<10} {config['url']}")
        
        print("="*60)
        print("📖 Documentation complète: README_EXPERT.md")
        print("🔧 Configuration: .env")
        print("📊 Tests: pytest tests/ -v")
        print("🐳 Docker: docker-compose up")
        print("="*60 + "\n")
    
    def run_interactive_menu(self):
        """Interface interactive pour gérer les services."""
        while self.running:
            try:
                self.display_status()
                print("Options disponibles:")
                print("1. Démarrer API")
                print("2. Démarrer Dashboard Expert")
                print("3. Démarrer Dashboard Basique")
                print("4. Démarrer tous les services")
                print("5. Arrêter tous les services")
                print("6. Voir les logs")
                print("7. Exécuter tests")
                print("8. Quitter")
                
                choice = input("\nVotre choix (1-8): ").strip()
                
                if choice == "1":
                    self.start_service('api')
                elif choice == "2":
                    self.start_service('dashboard')
                elif choice == "3":
                    self.start_service('basic_dashboard')
                elif choice == "4":
                    for service in ['api', 'dashboard']:
                        self.start_service(service)
                elif choice == "5":
                    self.stop_all_services()
                elif choice == "6":
                    self.show_logs()
                elif choice == "7":
                    self.run_tests()
                elif choice == "8":
                    self.stop_all_services()
                    break
                else:
                    print("❌ Choix invalide")
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                self.stop_all_services()
                break
    
    def show_logs(self):
        """Affiche les logs du projet."""
        log_dir = self.project_root / 'logs'
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                print(f"\n📄 Logs récents ({latest_log.name}):")
                print("-" * 50)
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines[-20:]:  # 20 dernières lignes
                            print(line.rstrip())
                except Exception as e:
                    print(f"❌ Erreur lecture logs: {e}")
            else:
                print("ℹ️  Aucun fichier de log trouvé")
        else:
            print("ℹ️  Répertoire logs inexistant")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def run_tests(self):
        """Exécute les tests du projet."""
        print("\n🧪 Exécution des tests...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Erreurs:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Erreur exécution tests: {e}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def run(self, auto_start: bool = False):
        """Point d'entrée principal."""
        # Configuration des signaux
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🚀 Démarrage du gestionnaire de services Supply Chain AI")
        
        if auto_start:
            # Démarrage automatique des services principaux
            self.start_service('api')
            time.sleep(2)
            self.start_service('dashboard')
            
            # Surveillance en arrière-plan
            monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
            monitor_thread.start()
            
            # Affichage de l'état
            time.sleep(3)
            self.display_status()
            
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_all_services()
        else:
            # Menu interactif
            self.run_interactive_menu()

def main():
    """Point d'entrée du script."""
    manager = ServiceManager()
    
    # Mode automatique si argument --auto
    auto_start = len(sys.argv) > 1 and sys.argv[1] == '--auto'
    
    try:
        manager.run(auto_start=auto_start)
    except Exception as e:
        logger.error("❌ Erreur fatale: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
