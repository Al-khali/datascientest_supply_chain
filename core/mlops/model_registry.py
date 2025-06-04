"""
Model Registry - Gestion des versions et métadonnées des modèles
=============================================================

Registry enterprise pour le versioning et la gouvernance des modèles ML.

Auteur: khalid
Date: 04/06/2025
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
from pathlib import Path
import logging


def utc_now() -> datetime:
    """Fonction helper pour obtenir l'heure UTC actuelle."""
    return datetime.now(timezone.utc)


class ModelStage(str, Enum):
    """Stages du modèle dans le registry."""
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelType(str, Enum):
    """Types de modèles ML."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CATEGORY_CLASSIFICATION = "category_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"


@dataclass
class ModelMetadata:
    """Métadonnées d'un modèle ML."""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    stage: ModelStage
    description: str
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    
    # Métriques de performance
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées MLflow
    mlflow_run_id: Optional[str] = None
    mlflow_model_uri: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    
    # Métadonnées de déploiement
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées de qualité
    data_drift_threshold: float = 0.1
    performance_threshold: float = 0.8
    last_validation_date: Optional[datetime] = None
    validation_status: str = "pending"


@dataclass
class ModelExperiment:
    """Expérimentation ML avec tracking."""
    experiment_id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=utc_now)
    tags: Dict[str, str] = field(default_factory=dict)
    artifact_location: Optional[str] = None
    lifecycle_stage: str = "active"


class IModelRegistry(ABC):
    """Interface pour le registry des modèles."""
    
    @abstractmethod
    async def register_model(
        self, 
        metadata: ModelMetadata,
        model_artifact: Optional[Path] = None
    ) -> str:
        """Enregistre un nouveau modèle."""
        pass
    
    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Récupère un modèle par ID."""
        pass
    
    @abstractmethod
    async def get_model_by_name_version(
        self, 
        name: str, 
        version: str
    ) -> Optional[ModelMetadata]:
        """Récupère un modèle par nom et version."""
        pass
    
    @abstractmethod
    async def list_models(
        self, 
        model_type: Optional[ModelType] = None,
        stage: Optional[ModelStage] = None
    ) -> List[ModelMetadata]:
        """Liste les modèles avec filtres optionnels."""
        pass
    
    @abstractmethod
    async def update_model_stage(
        self, 
        model_id: str, 
        new_stage: ModelStage
    ) -> bool:
        """Met à jour le stage d'un modèle."""
        pass
    
    @abstractmethod
    async def archive_model(self, model_id: str) -> bool:
        """Archive un modèle."""
        pass
    
    @abstractmethod
    async def create_experiment(
        self, 
        experiment: ModelExperiment
    ) -> str:
        """Crée une nouvelle expérimentation."""
        pass


class MLflowModelRegistry(IModelRegistry):
    """Implémentation MLflow du registry des modèles."""
    
    def __init__(
        self, 
        tracking_uri: str = "file:///tmp/mlruns",
        registry_uri: Optional[str] = None
    ):
        """Initialise le registry MLflow."""
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        
        # Configuration MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def register_model(
        self, 
        metadata: ModelMetadata,
        model_artifact: Optional[Path] = None
    ) -> str:
        """Enregistre un nouveau modèle dans MLflow."""
        try:
            # Créer une expérimentation si nécessaire
            experiment_name = f"{metadata.model_type.value}_experiments"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    tags={
                        "model_type": metadata.model_type.value,
                        "created_by": "mlops_system"
                    }
                )
            else:
                experiment_id = experiment.experiment_id
            
            # Démarrer un run MLflow
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log des métriques
                for metric_name, metric_value in metadata.metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log des hyperparamètres
                for param_name, param_value in metadata.hyperparameters.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log des tags
                mlflow.set_tags({
                    "model_type": metadata.model_type.value,
                    "stage": metadata.stage.value,
                    "version": metadata.version,
                    "description": metadata.description
                })
                
                # Log de l'artefact si fourni
                if model_artifact and model_artifact.exists():
                    mlflow.log_artifact(str(model_artifact))
                
                # Enregistrer le modèle
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=metadata.name
                )
                
                # Mettre à jour les métadonnées
                metadata.mlflow_run_id = run.info.run_id
                metadata.mlflow_model_uri = model_uri
                metadata.mlflow_experiment_id = experiment_id
                
                self.logger.info(f"Modèle enregistré: {metadata.name} v{metadata.version}")
                return registered_model.version
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement du modèle: {e}")
            raise
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Récupère un modèle par ID."""
        try:
            # Pour MLflow, utiliser le nom comme ID temporairement
            model_versions = self.client.search_model_versions(
                filter_string=f"name='{model_id}'"
            )
            
            if not model_versions:
                return None
            
            # Prendre la version la plus récente
            latest_version = max(
                model_versions, 
                key=lambda v: int(v.version)
            )
            
            return self._mlflow_version_to_metadata(latest_version)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du modèle {model_id}: {e}")
            return None
    
    async def get_model_by_name_version(
        self, 
        name: str, 
        version: str
    ) -> Optional[ModelMetadata]:
        """Récupère un modèle par nom et version."""
        try:
            model_version = self.client.get_model_version(
                name=name,
                version=version
            )
            
            return self._mlflow_version_to_metadata(model_version)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du modèle {name} v{version}: {e}")
            return None
    
    async def list_models(
        self, 
        model_type: Optional[ModelType] = None,
        stage: Optional[ModelStage] = None
    ) -> List[ModelMetadata]:
        """Liste les modèles avec filtres optionnels."""
        try:
            models = []
            
            # Construire le filtre
            filters = []
            if stage:
                filters.append(f"current_stage='{stage.value}'")
            
            filter_string = " AND ".join(filters) if filters else None
            
            model_versions = self.client.search_model_versions(
                filter_string=filter_string
            )
            
            for version in model_versions:
                metadata = self._mlflow_version_to_metadata(version)
                
                # Filtrer par type de modèle si spécifié
                if model_type and metadata.model_type != model_type:
                    continue
                
                models.append(metadata)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la liste des modèles: {e}")
            return []
    
    async def update_model_stage(
        self, 
        model_id: str, 
        new_stage: ModelStage
    ) -> bool:
        """Met à jour le stage d'un modèle."""
        try:
            # Récupérer la dernière version
            model_versions = self.client.search_model_versions(
                filter_string=f"name='{model_id}'"
            )
            
            if not model_versions:
                return False
            
            latest_version = max(
                model_versions, 
                key=lambda v: int(v.version)
            )
            
            # Mettre à jour le stage
            self.client.transition_model_version_stage(
                name=model_id,
                version=latest_version.version,
                stage=new_stage.value
            )
            
            self.logger.info(f"Stage mis à jour pour {model_id}: {new_stage.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du stage: {e}")
            return False
    
    async def archive_model(self, model_id: str) -> bool:
        """Archive un modèle."""
        return await self.update_model_stage(model_id, ModelStage.ARCHIVED)
    
    async def create_experiment(
        self, 
        experiment: ModelExperiment
    ) -> str:
        """Crée une nouvelle expérimentation."""
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment.name,
                artifact_location=experiment.artifact_location,
                tags=experiment.tags
            )
            
            self.logger.info(f"Expérimentation créée: {experiment.name}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de l'expérimentation: {e}")
            raise
    
    def _mlflow_version_to_metadata(self, version) -> ModelMetadata:
        """Convertit une version MLflow en métadonnées."""
        # Récupérer les tags et métriques du run
        run_id = version.run_id
        run = self.client.get_run(run_id)
        
        return ModelMetadata(
            model_id=version.name,
            name=version.name,
            version=version.version,
            model_type=ModelType(run.data.tags.get("model_type", "sentiment_analysis")),
            stage=ModelStage(version.current_stage),
            description=run.data.tags.get("description", ""),
            created_at=datetime.fromtimestamp(version.creation_timestamp / 1000, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(version.last_updated_timestamp / 1000, tz=timezone.utc),
            metrics=run.data.metrics,
            hyperparameters=run.data.params,
            mlflow_run_id=run_id,
            mlflow_model_uri=f"models:/{version.name}/{version.version}",
            mlflow_experiment_id=run.info.experiment_id
        )

    def _run_to_experiment(self, run: Run) -> ModelExperiment:
        """Convertit un run MLflow en expérimentation."""
        return ModelExperiment(
            experiment_id=run.info.experiment_id,
            name=run.data.tags.get("experiment_name", f"experiment_{run.info.experiment_id}"),
            description=run.data.tags.get("experiment_description", ""),
            tags=run.data.tags,
            artifact_location=run.info.artifact_uri,
            created_at=datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc),
            updated_at=datetime.fromtimestamp((run.info.end_time or run.info.start_time) / 1000, tz=timezone.utc)
        )


# Factory function pour créer une instance du registry
def create_model_registry(
    tracking_uri: str = "file:///tmp/mlruns",
    registry_uri: Optional[str] = None
) -> IModelRegistry:
    """Factory pour créer une instance du Model Registry."""
    return MLflowModelRegistry(tracking_uri=tracking_uri, registry_uri=registry_uri)


__all__ = [
    'ModelMetadata',
    'ModelStage', 
    'ModelType',
    'ModelExperiment',
    'IModelRegistry',
    'MLflowModelRegistry',
    'create_model_registry'
]
