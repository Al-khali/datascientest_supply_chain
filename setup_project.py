#!/usr/bin/env python3
"""
Script d'installation et de configuration automatisée du projet
Analyse de Satisfaction Client Supply Chain - Niveau Expert

Auteur: khalid
Date: 03/06/2025
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Gestionnaire de configuration automatisée du projet."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_dirs = [
            'data', 'logs', 'models', 'outputs', 
            'notebooks', 'src', 'tests', '.github/workflows'
        ]
        
    def check_python_version(self):
        """Vérifie la version Python."""
        if sys.version_info < (3, 10):
            logger.error("Python 3.10+ requis. Version actuelle: %s", sys.version)
            sys.exit(1)
        logger.info("✅ Python %s.%s détecté", sys.version_info.major, sys.version_info.minor)
    
    def create_directories(self):
        """Crée la structure de répertoires nécessaire."""
        for dir_name in self.required_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info("✅ Répertoire créé/vérifié: %s", dir_name)
    
    def install_requirements(self):
        """Installe les dépendances Python."""
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
                ], check=True)
                logger.info("✅ Dépendances installées avec succès")
            except subprocess.CalledProcessError:
                logger.error("❌ Erreur lors de l'installation des dépendances")
                sys.exit(1)
        else:
            logger.warning("⚠️  Fichier requirements.txt introuvable")
    
    def download_nltk_data(self):
        """Télécharge les données NLTK nécessaires."""
        try:
            import nltk
            datasets = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet']
            for dataset in datasets:
                try:
                    nltk.download(dataset, quiet=True)
                    logger.info("✅ Dataset NLTK téléchargé: %s", dataset)
                except Exception as e:
                    logger.warning("⚠️  Erreur téléchargement %s: %s", dataset, e)
        except ImportError:
            logger.warning("⚠️  NLTK non installé, passage de cette étape")
    
    def create_env_file(self):
        """Crée le fichier .env avec les variables d'environnement."""
        env_content = """# Configuration Projet Supply Chain Satisfaction Client
PROJECT_NAME=supply-satisfaction-ai
ENVIRONMENT=development

# Base de données
DATABASE_URL=postgresql://user:password@localhost:5432/supply_db
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Dashboard
DASHBOARD_PORT=8501

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/application.log

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# Cache
CACHE_EXPIRE_HOURS=24

# External APIs
TRUSTPILOT_API_KEY=your-trustpilot-api-key
GOOGLE_PLACES_API_KEY=your-google-places-api-key

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
ENABLE_CORS=true
ALLOWED_HOSTS=localhost,127.0.0.1,*.supply-ai.com
"""
        env_file = self.project_root / '.env'
        if not env_file.exists():
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            logger.info("✅ Fichier .env créé")
        else:
            logger.info("✅ Fichier .env existant conservé")
    
    def create_gitignore(self):
        """Crée/met à jour le fichier .gitignore."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/*.csv
data/*.json
data/*.parquet
!data/.gitkeep

# Logs
logs/*.log
!logs/.gitkeep

# Models
models/*.pkl
models/*.h5
models/*.joblib
!models/.gitkeep

# Environment variables
.env
.env.local
.env.production

# Cache
.cache/
*.cache
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml

# Coverage
.coverage
htmlcov/
"""
        gitignore_file = self.project_root / '.gitignore'
        with open(gitignore_file, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        logger.info("✅ Fichier .gitignore créé/mis à jour")
    
    def create_placeholder_files(self):
        """Crée des fichiers .gitkeep pour préserver la structure."""
        placeholder_dirs = ['data', 'logs', 'models', 'outputs']
        for dir_name in placeholder_dirs:
            placeholder_file = self.project_root / dir_name / '.gitkeep'
            placeholder_file.touch()
        logger.info("✅ Fichiers .gitkeep créés")
    
    def validate_installation(self):
        """Valide l'installation en important les modules principaux."""
        modules_to_test = [
            'pandas', 'numpy', 'scikit-learn', 'fastapi', 
            'streamlit', 'plotly', 'transformers'
        ]
        
        failed_imports = []
        for module in modules_to_test:
            try:
                __import__(module)
                logger.info("✅ Module validé: %s", module)
            except ImportError:
                failed_imports.append(module)
                logger.error("❌ Module manquant: %s", module)
        
        if failed_imports:
            logger.error("❌ Modules manquants: %s", ', '.join(failed_imports))
            logger.error("Exécutez: pip install -r requirements.txt")
            return False
        
        logger.info("✅ Tous les modules sont correctement installés")
        return True
    
    def run_setup(self):
        """Exécute la configuration complète du projet."""
        logger.info("🚀 Début de la configuration du projet Supply Chain AI")
        
        # Étapes de configuration
        self.check_python_version()
        self.create_directories()
        self.create_env_file()
        self.create_gitignore()
        self.create_placeholder_files()
        self.install_requirements()
        self.download_nltk_data()
        
        # Validation
        if self.validate_installation():
            logger.info("🎉 Configuration du projet terminée avec succès!")
            logger.info("📖 Consultez README_EXPERT.md pour les instructions d'utilisation")
            logger.info("🚀 Démarrage rapide:")
            logger.info("   - Dashboard: streamlit run src/dashboard_expert.py")
            logger.info("   - API: uvicorn src.api:app --reload")
            logger.info("   - Tests: pytest tests/ -v")
        else:
            logger.error("❌ Configuration incomplète. Vérifiez les erreurs ci-dessus.")
            sys.exit(1)

if __name__ == "__main__":
    setup = ProjectSetup()
    setup.run_setup()
