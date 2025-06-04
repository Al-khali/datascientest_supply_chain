"""
Model Monitoring - Surveillance et détection de dérive des modèles
=================================================================

Système enterprise de monitoring pour les modèles en production.
Inclut la détection de dérive des données et des performances.

Auteur: khalid
Date: 04/06/2025
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import logging
import asyncio


def utc_now() -> datetime:
    """Fonction helper pour obtenir l'heure UTC actuelle."""
    return datetime.now(timezone.utc)


class DriftType(str, Enum):
    """Types de dérive détectables."""
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    CONCEPT_DRIFT = "concept_drift"


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftDetectionMethod(str, Enum):
    """Méthodes de détection de dérive."""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    CHI_SQUARE = "chi_square"
    POPULATION_STABILITY_INDEX = "psi"
    JENSEN_SHANNON = "jensen_shannon"


@dataclass
class DriftAlert:
    """Alerte de dérive détectée."""
    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: AlertSeverity
    detected_at: datetime = field(default_factory=utc_now)
    
    # Détails de la dérive
    feature_name: Optional[str] = None
    drift_score: float = 0.0
    threshold: float = 0.0
    p_value: Optional[float] = None
    
    # Métadonnées
    description: str = ""
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statut
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MonitoringAlert:
    """Alerte générale de monitoring."""
    alert_id: str
    model_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class IDriftDetector(ABC):
    """Interface pour la détection de dérive."""
    
    @abstractmethod
    async def detect_data_drift_ks(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """Détecte la dérive de données avec test KS."""
        pass
    
    @abstractmethod
    async def detect_data_drift_chi2(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """Détecte la dérive de données avec test Chi-carré."""
        pass
    
    @abstractmethod
    async def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """Détecte la dérive des prédictions."""
        pass


class IModelMonitor(ABC):
    """Interface pour le monitoring des modèles."""
    
    @abstractmethod
    async def monitor_data_drift(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> List[DriftAlert]:
        """Surveille la dérive des données."""
        pass
    
    @abstractmethod
    async def monitor_prediction_drift(
        self,
        model_id: str,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> List[DriftAlert]:
        """Surveille la dérive des prédictions."""
        pass
    
    @abstractmethod
    async def monitor_performance(
        self,
        model_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_metrics: Dict[str, float]
    ) -> List[DriftAlert]:
        """Surveille les performances du modèle."""
        pass
    
    @abstractmethod
    async def get_alerts(
        self,
        model_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        drift_type: Optional[DriftType] = None
    ) -> List[MonitoringAlert]:
        """Récupère les alertes avec filtres optionnels."""
        pass


class StatisticalDriftDetector(IDriftDetector):
    """Détecteur de dérive basé sur des tests statistiques."""
    
    def __init__(self):
        """Initialise le détecteur."""
        self.logger = logging.getLogger(__name__)
    
    async def detect_data_drift_ks(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """Détecte la dérive avec le test de Kolmogorov-Smirnov."""
        try:
            # Test KS pour données continues
            statistic, p_value = stats.ks_2samp(reference_data, current_data)
            drift_detected = p_value < threshold
            
            self.logger.debug(f"KS Test - Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
            return drift_detected, p_value
            
        except Exception as e:
            self.logger.error(f"Erreur lors du test KS: {e}")
            return False, 1.0
    
    async def detect_data_drift_chi2(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """Détecte la dérive avec le test Chi-carré."""
        try:
            # Créer des histogrammes pour les données
            combined_data = np.concatenate([reference_data, current_data])
            bins = np.histogram_bin_edges(combined_data, bins='auto')
            
            ref_hist, _ = np.histogram(reference_data, bins=bins)
            cur_hist, _ = np.histogram(current_data, bins=bins)
            
            # Éviter les divisions par zéro
            ref_hist = np.maximum(ref_hist, 1)
            cur_hist = np.maximum(cur_hist, 1)
            
            # Test Chi-carré
            statistic, p_value = stats.chisquare(cur_hist, ref_hist)
            drift_detected = p_value < threshold
            
            self.logger.debug(f"Chi2 Test - Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
            return drift_detected, p_value
            
        except Exception as e:
            self.logger.error(f"Erreur lors du test Chi-carré: {e}")
            return False, 1.0
    
    async def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """Détecte la dérive des prédictions."""
        # Utiliser KS test pour les prédictions continues
        return await self.detect_data_drift_ks(
            reference_predictions, 
            current_predictions, 
            threshold
        )


class ModelMonitor(IModelMonitor):
    """Implémentation complète du monitoring des modèles."""
    
    def __init__(
        self,
        drift_detector: Optional[IDriftDetector] = None,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialise le monitor."""
        self.drift_detector = drift_detector or StatisticalDriftDetector()
        self.alert_thresholds = alert_thresholds or {
            "data_drift": 0.05,
            "prediction_drift": 0.05,
            "performance_drift": 0.05
        }
        self.alerts_storage: List[MonitoringAlert] = []
        self.logger = logging.getLogger(__name__)
    
    async def monitor_data_drift(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> List[DriftAlert]:
        """Surveille la dérive des données."""
        alerts = []
        
        try:
            # Vérifier chaque colonne numérique
            numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column not in current_data.columns:
                    continue
                
                ref_values = reference_data[column].dropna().values
                cur_values = current_data[column].dropna().values
                
                if len(ref_values) == 0 or len(cur_values) == 0:
                    continue
                
                # Test KS
                drift_detected, p_value = await self.drift_detector.detect_data_drift_ks(
                    ref_values, cur_values, self.alert_thresholds["data_drift"]
                )
                
                if drift_detected:
                    alert = DriftAlert(
                        alert_id=f"drift_{model_id}_{column}_{datetime.now().timestamp()}",
                        model_id=model_id,
                        drift_type=DriftType.DATA_DRIFT,
                        severity=self._calculate_severity(p_value),
                        feature_name=column,
                        drift_score=1 - p_value,
                        threshold=self.alert_thresholds["data_drift"],
                        p_value=p_value,
                        description=f"Dérive détectée sur la feature '{column}'",
                        recommendation=f"Investiguer les changements de distribution pour '{column}'"
                    )
                    alerts.append(alert)
                    
                    self.logger.warning(f"Dérive détectée - {model_id}:{column} (p={p_value:.4f})")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Erreur lors du monitoring de dérive: {e}")
            return alerts
    
    async def monitor_prediction_drift(
        self,
        model_id: str,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> List[DriftAlert]:
        """Surveille la dérive des prédictions."""
        alerts = []
        
        try:
            drift_detected, p_value = await self.drift_detector.detect_prediction_drift(
                reference_predictions, 
                current_predictions, 
                self.alert_thresholds["prediction_drift"]
            )
            
            if drift_detected:
                alert = DriftAlert(
                    alert_id=f"pred_drift_{model_id}_{datetime.now().timestamp()}",
                    model_id=model_id,
                    drift_type=DriftType.PREDICTION_DRIFT,
                    severity=self._calculate_severity(p_value),
                    drift_score=1 - p_value,
                    threshold=self.alert_thresholds["prediction_drift"],
                    p_value=p_value,
                    description="Dérive détectée dans les prédictions du modèle",
                    recommendation="Vérifier la qualité des données d'entrée et retrainer si nécessaire"
                )
                alerts.append(alert)
                
                self.logger.warning(f"Dérive des prédictions - {model_id} (p={p_value:.4f})")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Erreur lors du monitoring des prédictions: {e}")
            return alerts
    
    async def monitor_performance(
        self,
        model_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_metrics: Dict[str, float]
    ) -> List[DriftAlert]:
        """Surveille les performances du modèle."""
        alerts = []
        
        try:
            # Calculer les métriques actuelles
            current_metrics = await self._calculate_metrics(y_true, y_pred)
            
            # Comparer avec les métriques de référence
            for metric_name, baseline_value in baseline_metrics.items():
                if metric_name not in current_metrics:
                    continue
                
                current_value = current_metrics[metric_name]
                performance_drop = baseline_value - current_value
                relative_drop = performance_drop / baseline_value if baseline_value > 0 else 0
                
                if relative_drop > self.alert_thresholds["performance_drift"]:
                    severity = AlertSeverity.CRITICAL if relative_drop > 0.2 else AlertSeverity.HIGH
                    
                    alert = DriftAlert(
                        alert_id=f"perf_drift_{model_id}_{metric_name}_{datetime.now().timestamp()}",
                        model_id=model_id,
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        severity=severity,
                        feature_name=metric_name,
                        drift_score=relative_drop,
                        threshold=self.alert_thresholds["performance_drift"],
                        description=f"Dégradation de performance: {metric_name} ({current_value:.3f} vs {baseline_value:.3f})",
                        recommendation="Retrainer le modèle ou investiguer la qualité des données",
                        metadata={
                            "current_value": current_value,
                            "baseline_value": baseline_value,
                            "relative_drop": relative_drop
                        }
                    )
                    alerts.append(alert)
                    
                    self.logger.warning(f"Dégradation performance - {model_id}:{metric_name} ({relative_drop:.2%})")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Erreur lors du monitoring des performances: {e}")
            return alerts
    
    async def get_alerts(
        self,
        model_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        drift_type: Optional[DriftType] = None
    ) -> List[MonitoringAlert]:
        """Récupère les alertes avec filtres optionnels."""
        filtered_alerts = self.alerts_storage.copy()
        
        if model_id:
            filtered_alerts = [a for a in filtered_alerts if a.model_id == model_id]
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        return filtered_alerts
    
    def _calculate_severity(self, p_value: float) -> AlertSeverity:
        """Calcule la sévérité basée sur la p-value."""
        if p_value < 0.001:
            return AlertSeverity.CRITICAL
        elif p_value < 0.01:
            return AlertSeverity.HIGH
        elif p_value < 0.05:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    async def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        try:
            # Déterminer si c'est un problème de classification ou régression
            is_classification = len(np.unique(y_true)) <= 10  # Heuristique simple
            
            metrics = {}
            
            if is_classification:
                # Métriques de classification
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                # Métriques de régression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["mae"] = mean_absolute_error(y_true, y_pred)
                metrics["r2_score"] = r2_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur calcul métriques: {e}")
            return {}


# Factory functions
def create_drift_detector() -> IDriftDetector:
    """Factory pour créer un détecteur de dérive."""
    return StatisticalDriftDetector()


def create_model_monitor(
    drift_detector: Optional[IDriftDetector] = None,
    alert_thresholds: Optional[Dict[str, float]] = None
) -> IModelMonitor:
    """Factory pour créer un monitor de modèle."""
    return ModelMonitor(drift_detector=drift_detector, alert_thresholds=alert_thresholds)


__all__ = [
    'DriftType',
    'AlertSeverity',
    'DriftDetectionMethod',
    'DriftAlert',
    'MonitoringAlert',
    'IDriftDetector',
    'IModelMonitor',
    'StatisticalDriftDetector',
    'ModelMonitor',
    'create_drift_detector',
    'create_model_monitor'
]
