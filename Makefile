# Makefile pour le projet Supply Chain Satisfaction Client
# Facilite les tâches de développement courantes

.PHONY: help setup install test lint format clean run-api run-dashboard run-all docker-build docker-run

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := supply-satisfaction-ai

# Aide par défaut
help:
	@echo "🚀 Commandes disponibles pour Supply Chain Satisfaction Client:"
	@echo ""
	@echo "📦 Installation & Configuration:"
	@echo "  make setup        - Configuration complète du projet"
	@echo "  make install      - Installation des dépendances"
	@echo "  make clean        - Nettoyage des fichiers temporaires"
	@echo ""
	@echo "🔍 Qualité du Code:"
	@echo "  make test         - Exécution des tests"
	@echo "  make test-cov     - Tests avec couverture de code"
	@echo "  make lint         - Vérification du code (flake8)"
	@echo "  make format       - Formatage automatique (black)"
	@echo "  make check        - Vérification complète (lint + tests)"
	@echo ""
	@echo "🚀 Démarrage des Services:"
	@echo "  make run-api      - Démarrer l'API FastAPI"
	@echo "  make run-dashboard - Démarrer le dashboard expert"
	@echo "  make run-basic    - Démarrer le dashboard basique"
	@echo "  make run-all      - Démarrer tous les services"
	@echo "  make dev          - Mode développement interactif"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build - Construire l'image Docker"
	@echo "  make docker-run   - Exécuter via Docker"
	@echo "  make docker-stop  - Arrêter les conteneurs"
	@echo ""
	@echo "📊 Data Pipeline:"
	@echo "  make collect      - Collecter les avis"
	@echo "  make clean-data   - Nettoyer les données"
	@echo "  make analyze      - Analyser sentiment & motifs"
	@echo "  make pipeline     - Pipeline complet"

# Configuration du projet
setup:
	@echo "🔧 Configuration du projet..."
	$(PYTHON) setup_project.py
	@echo "✅ Configuration terminée!"

# Installation des dépendances
install:
	@echo "📦 Installation des dépendances..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dépendances installées!"

# Nettoyage
clean:
	@echo "🧹 Nettoyage en cours..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Nettoyage terminé!"

# Tests
test:
	@echo "🧪 Exécution des tests..."
	$(PYTHON) -m pytest tests/ -v

test-cov:
	@echo "🧪 Tests avec couverture de code..."
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Vérification du code
lint:
	@echo "🔍 Vérification du code..."
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

# Formatage automatique
format:
	@echo "🎨 Formatage du code..."
	$(PYTHON) -m black src/ tests/ --line-length=100
	$(PYTHON) -m isort src/ tests/

# Vérification complète
check: lint test
	@echo "✅ Vérification complète terminée!"

# Démarrage des services
run-api:
	@echo "🚀 Démarrage de l'API FastAPI..."
	$(PYTHON) -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	@echo "🎨 Démarrage du dashboard expert..."
	$(PYTHON) -m streamlit run src/dashboard_expert.py --server.port 8501

run-basic:
	@echo "🎨 Démarrage du dashboard basique..."
	$(PYTHON) -m streamlit run src/dashboard.py --server.port 8502

run-all:
	@echo "🚀 Démarrage de tous les services..."
	$(PYTHON) start_services.py --auto

dev:
	@echo "👨‍💻 Mode développement interactif..."
	$(PYTHON) start_services.py

# Docker
docker-build:
	@echo "🐳 Construction de l'image Docker..."
	docker build -t $(PROJECT_NAME) .

docker-run:
	@echo "🐳 Exécution via Docker..."
	docker run -p 8000:8000 -p 8501:8501 $(PROJECT_NAME)

docker-stop:
	@echo "🛑 Arrêt des conteneurs Docker..."
	docker stop $$(docker ps -q --filter ancestor=$(PROJECT_NAME)) 2>/dev/null || true

# Pipeline de données
collect:
	@echo "📥 Collecte des avis..."
	$(PYTHON) src/collect_trustpilot.py

clean-data:
	@echo "🧼 Nettoyage des données..."
	$(PYTHON) src/clean_reviews.py

analyze:
	@echo "🧠 Analyse de sentiment..."
	$(PYTHON) src/sentiment_motifs.py

pipeline: collect clean-data analyze
	@echo "✅ Pipeline de données terminé!"

# Installation complète pour nouveau développeur
onboard: setup install test
	@echo "🎉 Configuration terminée pour nouveau développeur!"
	@echo "📖 Consultez README_EXPERT.md pour plus d'informations"
	@echo "🚀 Démarrage rapide: make dev"

# Déploiement
deploy-staging:
	@echo "🚀 Déploiement en staging..."
	# Ajoutez ici vos commandes de déploiement

deploy-prod:
	@echo "🚀 Déploiement en production..."
	# Ajoutez ici vos commandes de déploiement

# Monitoring
logs:
	@echo "📄 Affichage des logs..."
	tail -f logs/*.log 2>/dev/null || echo "Aucun fichier de log trouvé"

# Base de données (si utilisée)
db-migrate:
	@echo "📊 Migration de la base de données..."
	# Ajoutez vos commandes de migration

db-seed:
	@echo "🌱 Initialisation des données..."
	# Ajoutez vos commandes d'initialisation
