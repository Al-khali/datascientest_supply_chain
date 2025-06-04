"""
Configuration Management
========================

Gestion centralisée de la configuration avec validation et chargement sécurisé.
Support des environnements multiples et validation des paramètres.

Auteur: khalid
Date: 04/06/2025
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json
from enum import Enum


class Environment(str, Enum):
    """Environnements supportés."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Configuration base de données."""
    host: str = "localhost"
    port: int = 5432
    database: str = "supply_chain_db"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @property
    def url(self) -> str:
        """URL de connexion PostgreSQL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Configuration Redis."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    decode_responses: bool = True
    
    @property
    def url(self) -> str:
        """URL de connexion Redis."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class MLConfig:
    """Configuration Machine Learning."""
    models_path: str = "./models"
    cache_dir: str = "./cache/models"
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Modèles par défaut
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    ner_model: str = "fr_core_news_md"
    
    # Seuils de confiance
    sentiment_confidence_threshold: float = 0.7
    category_confidence_threshold: float = 0.6
    
    # Performance
    enable_model_caching: bool = True
    async_processing: bool = True
    max_concurrent_requests: int = 10


@dataclass
class APIConfig:
    """Configuration API."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # Sécurité
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # CORS
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Configuration monitoring."""
    enable_prometheus: bool = True
    prometheus_port: int = 8001
    
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = "logs/app.log"
    
    # Métriques business
    track_business_metrics: bool = True
    metrics_retention_days: int = 30
    
    # Alerting
    enable_alerting: bool = True
    alert_webhook_url: Optional[str] = None
    critical_sentiment_threshold: float = -0.7
    high_volume_threshold: int = 1000


@dataclass
class ExternalAPIConfig:
    """Configuration APIs externes."""
    # Trustpilot
    trustpilot_api_key: Optional[str] = None
    trustpilot_rate_limit: int = 100
    
    # Google Reviews
    google_api_key: Optional[str] = None
    google_places_api_key: Optional[str] = None
    
    # Social Media
    twitter_bearer_token: Optional[str] = None
    facebook_access_token: Optional[str] = None
    
    # Rate limiting global
    default_rate_limit: int = 60
    request_timeout: int = 30


@dataclass
class AppConfig:
    """Configuration principale de l'application."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Configurations spécialisées
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    external_apis: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    
    # Paths
    data_path: str = "./data"
    logs_path: str = "./logs"
    temp_path: str = "./tmp"
    
    def __post_init__(self):
        """Validation et initialisation post-création."""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """Valide la configuration."""
        if self.environment == Environment.PRODUCTION:
            if self.api.secret_key == "change-me-in-production":
                raise ValueError("Secret key must be changed in production")
            
            if not self.database.password:
                raise ValueError("Database password required in production")
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires."""
        for path in [self.data_path, self.logs_path, self.temp_path, self.ml.models_path]:
            Path(path).mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Gestionnaire de configuration centralisé."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Pattern Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(
        self, 
        config_path: Optional[str] = None,
        env_override: bool = True
    ) -> AppConfig:
        """Charge la configuration depuis un fichier ou les variables d'environnement.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            env_override: Si True, les variables d'environnement surchargent le fichier
            
        Returns:
            Configuration chargée et validée
        """
        if self._config is not None:
            return self._config
        
        # Configuration par défaut
        config = AppConfig()
        
        # Chargement depuis fichier
        if config_path and Path(config_path).exists():
            config = self._load_from_file(config_path)
        
        # Surcharge avec variables d'environnement
        if env_override:
            config = self._load_from_env(config)
        
        self._config = config
        return config
    
    def _load_from_file(self, config_path: str) -> AppConfig:
        """Charge la configuration depuis un fichier YAML ou JSON."""
        config_file = Path(config_path)
        
        if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        return self._dict_to_config(data)
    
    def _load_from_env(self, config: AppConfig) -> AppConfig:
        """Charge les variables d'environnement dans la configuration."""
        
        # Environment
        env_str = os.getenv('ENVIRONMENT', config.environment.value)
        config.environment = Environment(env_str)
        config.debug = os.getenv('DEBUG', str(config.debug)).lower() == 'true'
        
        # Database
        config.database.host = os.getenv('DB_HOST', config.database.host)
        config.database.port = int(os.getenv('DB_PORT', str(config.database.port)))
        config.database.database = os.getenv('DB_NAME', config.database.database)
        config.database.username = os.getenv('DB_USER', config.database.username)
        config.database.password = os.getenv('DB_PASSWORD', config.database.password)
        
        # Redis
        config.redis.host = os.getenv('REDIS_HOST', config.redis.host)
        config.redis.port = int(os.getenv('REDIS_PORT', str(config.redis.port)))
        config.redis.password = os.getenv('REDIS_PASSWORD', config.redis.password)
        
        # API
        config.api.host = os.getenv('API_HOST', config.api.host)
        config.api.port = int(os.getenv('API_PORT', str(config.api.port)))
        config.api.secret_key = os.getenv('SECRET_KEY', config.api.secret_key)
        
        # External APIs
        config.external_apis.trustpilot_api_key = os.getenv('TRUSTPILOT_API_KEY')
        config.external_apis.google_api_key = os.getenv('GOOGLE_API_KEY')
        config.external_apis.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        return config
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """Convertit un dictionnaire en configuration."""
        # Implémentation simplifiée - à étendre selon les besoins
        config = AppConfig()
        
        if 'environment' in data:
            config.environment = Environment(data['environment'])
        
        if 'debug' in data:
            config.debug = data['debug']
        
        # Database
        if 'database' in data:
            db_data = data['database']
            config.database = DatabaseConfig(**db_data)
        
        # Redis
        if 'redis' in data:
            redis_data = data['redis']
            config.redis = RedisConfig(**redis_data)
        
        # ML
        if 'ml' in data:
            ml_data = data['ml']
            config.ml = MLConfig(**ml_data)
        
        # API
        if 'api' in data:
            api_data = data['api']
            config.api = APIConfig(**api_data)
        
        return config
    
    def get_config(self) -> AppConfig:
        """Récupère la configuration courante."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self, config_path: Optional[str] = None) -> AppConfig:
        """Recharge la configuration."""
        self._config = None
        return self.load_config(config_path)


# Instance globale
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Fonction utilitaire pour récupérer la configuration."""
    return config_manager.get_config()
