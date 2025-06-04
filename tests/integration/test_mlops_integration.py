"""
Tests d'intégration MLOps - Validation complète de l'infrastructure
==================================================================

Tests enterprise pour valider l'intégration complète des composants MLOps:
- Model Registry avec MLflow
- Monitoring et détection de dérive
- Déploiement automatisé
- Workflow de bout en bout

Auteur: khalid
Date: 04/06/2025
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import mlflow
from mlflow.tracking import MlflowClient

# Imports des modules MLOps
from core.mlops.model_registry import (
    MLflowModelRegistry, 
    ModelMetadata, 
    ModelType, 
    ModelStage, 
    ModelExperiment
)
from core.mlops.monitoring import (
    ModelMonitor, 
    StatisticalDriftDetector,
    DriftType,
    AlertSeverity
)
from core.mlops.deployment import (
    ModelDeploymentManager, 
    MLflowModelDeployer,
    DeploymentConfig,
    DeploymentTarget,
    DeploymentStrategy,
    DeploymentStatus
)


class TestMLOpsIntegration:
    """Suite de tests d'intégration pour l'infrastructure MLOps."""
    
    @pytest.fixture
    def temp_mlflow_dir(self):
        """Crée un répertoire temporaire pour MLflow."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mlflow_registry(self, temp_mlflow_dir):
        """Registry MLflow pour les tests."""
        tracking_uri = f"file://{temp_mlflow_dir}/mlruns"
        return MLflowModelRegistry(tracking_uri=tracking_uri)
    
    @pytest.fixture
    def sample_model_metadata(self):
        """Métadonnées de modèle exemple pour les tests."""
        return ModelMetadata(
            model_id="sentiment_analyzer_v1",
            name="sentiment_analyzer_v1",
            version="1.0.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            stage=ModelStage.DEVELOPMENT,
            description="Modèle d'analyse de sentiment BERT fine-tuné",
            metrics={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.87,
                "f1_score": 0.84
            },
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "model_name": "bert-base-uncased"
            }
        )
    
    @pytest.fixture
    def drift_detector(self):
        """Détecteur de dérive statistique."""
        return StatisticalDriftDetector()
    
    @pytest.fixture
    def model_monitor(self, drift_detector):
        """Monitor de modèle avec détecteur."""
        return ModelMonitor(
            drift_detector=drift_detector,
            alert_thresholds={
                "data_drift": 0.05,
                "prediction_drift": 0.05,
                "performance_drift": 0.1
            }
        )
    
    @pytest.fixture
    def deployment_config(self):
        """Configuration de déploiement exemple."""
        return DeploymentConfig(
            target=DeploymentTarget.LOCAL,
            strategy=DeploymentStrategy.BLUE_GREEN,
            port=8080,
            health_check_path="/health",
            environment_variables={"MODEL_ENV": "test"},
            resource_limits={"memory": "1Gi", "cpu": "500m"}
        )

    @pytest.mark.asyncio
    async def test_model_registry_full_lifecycle(self, mlflow_registry, sample_model_metadata):
        """Test du cycle de vie complet d'un modèle dans le registry."""
        
        # 1. Enregistrement du modèle
        version = await mlflow_registry.register_model(sample_model_metadata)
        assert version is not None
        assert version != ""
        
        # 2. Récupération du modèle
        retrieved_model = await mlflow_registry.get_model(sample_model_metadata.model_id)
        assert retrieved_model is not None
        assert retrieved_model.name == sample_model_metadata.name
        assert retrieved_model.model_type == sample_model_metadata.model_type
        assert retrieved_model.metrics["accuracy"] == 0.85
        
        # 3. Transition vers Staging
        success = await mlflow_registry.update_model_stage(
            sample_model_metadata.model_id, 
            ModelStage.STAGING
        )
        assert success is True
        
        # 4. Vérification du changement de stage
        updated_model = await mlflow_registry.get_model(sample_model_metadata.model_id)
        assert updated_model.stage == ModelStage.STAGING
        
        # 5. Liste des modèles par stage
        staging_models = await mlflow_registry.list_models(stage=ModelStage.STAGING)
        assert len(staging_models) >= 1
        assert any(m.model_id == sample_model_metadata.model_id for m in staging_models)
        
        # 6. Archivage du modèle
        archived = await mlflow_registry.archive_model(sample_model_metadata.model_id)
        assert archived is True

    @pytest.mark.asyncio
    async def test_drift_detection_integration(self, model_monitor):
        """Test d'intégration de la détection de dérive."""
        
        # Données de référence (distribution normale)
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.uniform(0, 10, 1000)
        })
        
        # Données actuelles avec dérive (décalage de moyenne)
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(2, 1, 1000),  # Dérive détectable
            'feature_2': np.random.normal(5, 2, 1000),  # Pas de dérive
            'feature_3': np.random.uniform(0, 10, 1000)  # Pas de dérive
        })
        
        # Prédictions simulées
        reference_predictions = np.random.beta(2, 5, 1000)  # Distribution asymétrique
        current_predictions = np.random.beta(5, 2, 1000)   # Distribution différente
        
        # Labels simulés pour les métriques de performance
        current_labels = np.random.binomial(1, 0.7, 1000)
        
        # Exécution du monitoring
        report = await model_monitor.monitor_model(
            model_id="test_model_drift",
            reference_data=reference_data,
            current_data=current_data,
            current_predictions=current_predictions,
            current_labels=current_labels
        )
        
        # Vérifications
        assert report.model_id == "test_model_drift"
        assert len(report.drift_detections) > 0
        
        # Vérifier qu'une dérive a été détectée sur feature_1
        feature_1_drift = next(
            (d for d in report.drift_detections if d.feature_name == 'feature_1'),
            None
        )
        assert feature_1_drift is not None
        assert feature_1_drift.is_drift_detected is True
        
        # Vérifier les alertes générées
        assert len(report.alerts) > 0
        data_drift_alerts = [a for a in report.alerts if a.drift_type == DriftType.DATA_DRIFT]
        assert len(data_drift_alerts) > 0
        
        # Vérifier les métriques de performance
        assert len(report.performance_metrics) > 0
        perf_metrics = report.performance_metrics[0]
        assert 0 <= perf_metrics.accuracy <= 1
        assert 0 <= perf_metrics.precision <= 1
        assert 0 <= perf_metrics.recall <= 1
        assert 0 <= perf_metrics.f1_score <= 1

    @pytest.mark.asyncio
    async def test_deployment_integration(self, deployment_config):
        """Test d'intégration du système de déploiement."""
        
        # Mock MLflow client pour éviter les appels réels
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock des méthodes MLflow nécessaires
            mock_client.get_model_version.return_value = Mock(
                name="test_model",
                version="1",
                source="dummy_source",
                run_id="dummy_run_id"
            )
            
            # Créer le déployeur
            deployer = MLflowModelDeployer(
                tracking_uri="dummy_uri",
                registry_uri="dummy_registry_uri"
            )
            
            # Test de déploiement local (simulation)
            deployment_info = await deployer.deploy_model(
                model_id="test_model",
                model_version="1",
                config=deployment_config
            )
            
            # Vérifications
            assert deployment_info is not None
            assert deployment_info.model_id == "test_model"
            assert deployment_info.model_version == "1"
            assert deployment_info.config.target == DeploymentTarget.LOCAL
            assert deployment_info.status == DeploymentStatus.RUNNING
            
            # Test de récupération du statut
            status = await deployer.get_deployment_status(deployment_info.deployment_id)
            assert status is not None
            assert status.deployment_id == deployment_info.deployment_id

    @pytest.mark.asyncio
    async def test_end_to_end_mlops_workflow(
        self, 
        mlflow_registry, 
        model_monitor, 
        sample_model_metadata,
        deployment_config
    ):
        """Test du workflow MLOps de bout en bout."""
        
        # Étape 1: Enregistrement du modèle
        version = await mlflow_registry.register_model(sample_model_metadata)
        assert version is not None
        
        # Étape 2: Validation et promotion en staging
        success = await mlflow_registry.update_model_stage(
            sample_model_metadata.model_id,
            ModelStage.STAGING
        )
        assert success is True
        
        # Étape 3: Monitoring du modèle en staging
        reference_data = pd.DataFrame({
            'sentiment_score': np.random.normal(0.5, 0.2, 500),
            'confidence': np.random.beta(2, 2, 500),
            'text_length': np.random.poisson(100, 500)
        })
        
        current_data = pd.DataFrame({
            'sentiment_score': np.random.normal(0.6, 0.2, 500),  # Légère dérive
            'confidence': np.random.beta(2, 2, 500),
            'text_length': np.random.poisson(105, 500)  # Légère dérive
        })
        
        monitoring_report = await model_monitor.monitor_model(
            model_id=sample_model_metadata.model_id,
            reference_data=reference_data,
            current_data=current_data
        )
        
        assert monitoring_report.model_id == sample_model_metadata.model_id
        
        # Étape 4: Décision de promotion basée sur le monitoring
        high_severity_alerts = [
            a for a in monitoring_report.alerts 
            if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        ]
        
        if len(high_severity_alerts) == 0:
            # Promotion en production si pas d'alertes critiques
            production_success = await mlflow_registry.update_model_stage(
                sample_model_metadata.model_id,
                ModelStage.PRODUCTION
            )
            assert production_success is True
            
            # Vérification du modèle en production
            prod_model = await mlflow_registry.get_model(sample_model_metadata.model_id)
            assert prod_model.stage == ModelStage.PRODUCTION
        
        # Étape 5: Déploiement (simulé)
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            mock_client.get_model_version.return_value = Mock(
                name=sample_model_metadata.name,
                version=version,
                source="dummy_source",
                run_id="dummy_run_id"
            )
            
            deployer = MLflowModelDeployer(
                tracking_uri="dummy_uri",
                registry_uri="dummy_registry_uri"
            )
            
            deployment_info = await deployer.deploy_model(
                model_id=sample_model_metadata.model_id,
                model_version=version,
                config=deployment_config
            )
            
            assert deployment_info.status == DeploymentStatus.RUNNING

    @pytest.mark.asyncio
    async def test_alerting_and_notification_system(self, model_monitor):
        """Test du système d'alertes et de notifications."""
        
        # Créer une situation avec dérive critique
        reference_data = pd.DataFrame({
            'critical_feature': np.random.normal(0, 1, 1000)
        })
        
        # Dérive très importante (différence de 5 écarts-types)
        current_data = pd.DataFrame({
            'critical_feature': np.random.normal(5, 1, 1000)
        })
        
        # Monitoring
        report = await model_monitor.monitor_model(
            model_id="critical_drift_test",
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Vérifier qu'une dérive critique a été détectée
        critical_alerts = [
            a for a in report.alerts 
            if a.severity == AlertSeverity.CRITICAL
        ]
        assert len(critical_alerts) > 0
        
        # Test de récupération des alertes
        all_alerts = await model_monitor.get_alerts()
        assert len(all_alerts) > 0
        
        # Test de filtrage des alertes par sévérité
        critical_alerts_filtered = await model_monitor.get_alerts(
            severity=AlertSeverity.CRITICAL
        )
        assert len(critical_alerts_filtered) > 0
        
        # Test de filtrage des alertes par modèle
        model_alerts = await model_monitor.get_alerts(
            model_id="critical_drift_test"
        )
        assert len(model_alerts) > 0
        assert all(a.model_id == "critical_drift_test" for a in model_alerts)

    @pytest.mark.asyncio
    async def test_model_performance_degradation_detection(self, model_monitor):
        """Test de détection de dégradation des performances."""
        
        # Données de référence
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(0, 1, 500)
        })
        
        # Données actuelles (similaires, pas de dérive de données)
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(0, 1, 500)
        })
        
        # Prédictions avec performance dégradée
        # Labels réels (ground truth)
        current_labels = np.random.binomial(1, 0.7, 500)
        
        # Prédictions dégradées (faible corrélation avec les labels)
        current_predictions = np.random.uniform(0, 1, 500)
        
        # Monitoring
        report = await model_monitor.monitor_model(
            model_id="performance_degradation_test",
            reference_data=reference_data,
            current_data=current_data,
            current_predictions=current_predictions,
            current_labels=current_labels
        )
        
        # Vérifier que les métriques de performance ont été calculées
        assert len(report.performance_metrics) > 0
        perf_metrics = report.performance_metrics[0]
        
        # Avec des prédictions aléatoires, la performance devrait être faible
        assert perf_metrics.accuracy < 0.8  # Seuil de performance acceptable
        
        # Vérifier le résumé du rapport
        assert "monitoring_status" in report.summary
        assert report.summary["total_alerts"] >= 0

    @pytest.mark.asyncio
    async def test_model_registry_experiment_management(self, mlflow_registry):
        """Test de gestion des expérimentations dans le registry."""
        
        # Créer une expérimentation
        experiment = ModelExperiment(
            experiment_id="exp_001",
            name="sentiment_analysis_experiments",
            description="Expérimentations pour l'analyse de sentiment",
            tags={
                "team": "data_science",
                "project": "customer_satisfaction",
                "priority": "high"
            }
        )
        
        # Enregistrer l'expérimentation
        experiment_id = await mlflow_registry.create_experiment(experiment)
        assert experiment_id is not None
        assert experiment_id != ""
        
        # Vérifier que l'expérimentation peut être utilisée pour les modèles
        model_metadata = ModelMetadata(
            model_id="exp_model_v1",
            name="exp_model_v1",
            version="1.0.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            stage=ModelStage.DEVELOPMENT,
            description="Modèle créé dans l'expérimentation",
            metrics={"accuracy": 0.88}
        )
        
        # Le modèle devrait être enregistré dans l'expérimentation créée
        version = await mlflow_registry.register_model(model_metadata)
        assert version is not None

    def test_deployment_config_validation(self, deployment_config):
        """Test de validation de la configuration de déploiement."""
        
        # Configuration valide
        assert deployment_config.target == DeploymentTarget.LOCAL
        assert deployment_config.strategy == DeploymentStrategy.BLUE_GREEN
        assert deployment_config.port == 8080
        assert deployment_config.health_check_path == "/health"
        assert "MODEL_ENV" in deployment_config.environment_variables
        assert "memory" in deployment_config.resource_limits
        assert "cpu" in deployment_config.resource_limits
        
        # Test de sérialisation (important pour les APIs)
        config_dict = deployment_config.__dict__
        assert "target" in config_dict
        assert "strategy" in config_dict
        assert "port" in config_dict


class TestMLOpsPerformance:
    """Tests de performance pour l'infrastructure MLOps."""
    
    @pytest.mark.asyncio
    async def test_bulk_model_registration_performance(self, mlflow_registry):
        """Test de performance pour l'enregistrement en masse de modèles."""
        
        models_count = 5  # Réduit pour les tests
        models = []
        
        for i in range(models_count):
            model = ModelMetadata(
                model_id=f"bulk_model_{i}",
                name=f"bulk_model_{i}",
                version="1.0.0",
                model_type=ModelType.SENTIMENT_ANALYSIS,
                stage=ModelStage.DEVELOPMENT,
                description=f"Modèle en lot numéro {i}",
                metrics={"accuracy": 0.8 + (i * 0.01)}
            )
            models.append(model)
        
        # Enregistrement séquentiel avec mesure du temps
        start_time = datetime.now()
        
        for model in models:
            version = await mlflow_registry.register_model(model)
            assert version is not None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Vérifier que l'enregistrement est raisonnablement rapide
        # (moins de 2 secondes par modèle en moyenne)
        assert duration < (models_count * 2)
        
        # Vérifier que tous les modèles ont été enregistrés
        all_models = await mlflow_registry.list_models()
        bulk_models = [m for m in all_models if m.model_id.startswith("bulk_model_")]
        assert len(bulk_models) == models_count

    @pytest.mark.asyncio
    async def test_monitoring_large_dataset_performance(self, model_monitor):
        """Test de performance du monitoring sur de gros volumes de données."""
        
        # Datasets plus grands pour tester la performance
        large_size = 5000
        
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, large_size),
            'feature_2': np.random.exponential(2, large_size),
            'feature_3': np.random.gamma(2, 2, large_size)
        })
        
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(0.1, 1, large_size),
            'feature_2': np.random.exponential(2.1, large_size),
            'feature_3': np.random.gamma(2.1, 2, large_size)
        })
        
        # Mesure du temps de monitoring
        start_time = datetime.now()
        
        report = await model_monitor.monitor_model(
            model_id="large_dataset_test",
            reference_data=reference_data,
            current_data=current_data
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Le monitoring devrait être terminé en moins de 10 secondes
        assert duration < 10
        
        # Vérifier que le rapport est complet
        assert report.model_id == "large_dataset_test"
        assert len(report.drift_detections) == 3  # Une par feature
        assert "monitoring_status" in report.summary


if __name__ == "__main__":
    # Pour exécuter les tests directement
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
