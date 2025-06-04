"""
MLOps Module - Intégration Machine Learning Operations
====================================================

Module enterprise pour la gestion du cycle de vie des modèles ML.
Inclut versioning, tracking, déploiement et monitoring.

Auteur: khalid
Date: 04/06/2025
"""

# Model Registry
from .model_registry import (
    ModelMetadata,
    ModelStage,
    ModelType,
    ModelExperiment,
    IModelRegistry,
    MLflowModelRegistry,
    create_model_registry
)

# Monitoring
from .monitoring import (
    DriftType,
    AlertSeverity,
    DriftAlert,
    StatisticalDriftDetector,
    ModelMonitor,
    create_drift_detector,
    create_model_monitor
)

# Deployment
from .deployment import (
    DeploymentTarget,
    DeploymentStrategy,
    DeploymentStatus,
    ModelDeployment,
    IModelDeployer,
    MLflowModelDeployer,
    ModelDeploymentManager,
    create_deployment_manager
)

__all__ = [
    # Model Registry
    'ModelMetadata',
    'ModelStage',
    'ModelType',
    'ModelExperiment', 
    'IModelRegistry',
    'MLflowModelRegistry',
    'create_model_registry',
    
    # Monitoring
    'DriftType',
    'AlertSeverity',
    'DriftAlert',
    'StatisticalDriftDetector',
    'ModelMonitor',
    'create_drift_detector',
    'create_model_monitor',
    
    # Deployment
    'DeploymentTarget',
    'DeploymentStrategy',
    'DeploymentStatus',
    'ModelDeployment',
    'IModelDeployer',
    'MLflowModelDeployer',
    'ModelDeploymentManager',
    'create_deployment_manager'
]