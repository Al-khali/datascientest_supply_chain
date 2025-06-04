"""
Migration Script - Integration Legacy
=====================================

Script de migration pour intégrer l'architecture Clean avec le code existant.
Permet une transition progressive vers l'architecture enterprise.

Auteur: khalid
Date: 04/06/2025
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import json

# Import du code existant
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.sentiment_motifs import SentimentAnalyzer  # Code existant
except ImportError:
    print("Legacy code not found, creating adapter...")
    SentimentAnalyzer = None

# Architecture Clean
from core.services.service_configuration import initialize_enterprise_services, get_service_configuration
from core.application.services import get_review_service, get_analytics_service
from core.domain.models.review import ReviewId, SentimentLabel, SupplyChainCategory, CriticalityLevel
from core.services.dependency_injection import inject
from core.domain.interfaces.repositories import ReviewRepository

logger = logging.getLogger(__name__)


class LegacyIntegrationService:
    """Service d'intégration pour migrer le code legacy vers l'architecture Clean."""
    
    def __init__(self):
        """Initialise le service de migration."""
        self.legacy_analyzer = None
        self.enterprise_service = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialise les services legacy et enterprise."""
        if self._initialized:
            return
        
        try:
            # Initialisation de l'architecture enterprise
            await initialize_enterprise_services()
            self.enterprise_service = await get_review_service()
            
            # Initialisation du code legacy si disponible
            if SentimentAnalyzer:
                self.legacy_analyzer = SentimentAnalyzer()
            
            self._initialized = True
            logger.info("Legacy integration service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize integration service: {e}")
            raise
    
    async def migrate_data_from_legacy(self, data_path: str) -> Dict[str, Any]:
        """Migre les données du système legacy vers l'architecture Clean.
        
        Args:
            data_path: Chemin vers les données legacy
            
        Returns:
            Rapport de migration
        """
        if not self._initialized:
            await self.initialize()
        
        migration_report = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_records": 0,
            "migrated_records": 0,
            "failed_records": 0,
            "errors": []
        }
        
        try:
            # Lecture des données legacy
            data_file = Path(data_path)
            
            if data_file.suffix == '.csv':
                df = pd.read_csv(data_file)
            elif data_file.suffix == '.json':
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            else:
                raise ValueError(f"Unsupported file format: {data_file.suffix}")
            
            migration_report["total_records"] = len(df)
            
            # Migration par batch
            batch_size = 50
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                batch_results = await self._migrate_batch(batch)
                
                migration_report["migrated_records"] += batch_results["migrated"]
                migration_report["failed_records"] += batch_results["failed"]
                migration_report["errors"].extend(batch_results["errors"])
                
                logger.info(f"Migrated batch {i // batch_size + 1}: {batch_results['migrated']}/{len(batch)} records")
            
            migration_report["completed_at"] = datetime.now(timezone.utc).isoformat()
            migration_report["success_rate"] = migration_report["migrated_records"] / migration_report["total_records"] if migration_report["total_records"] > 0 else 0
            
            logger.info(f"Migration completed: {migration_report['migrated_records']}/{migration_report['total_records']} records migrated")
            
            return migration_report
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            migration_report["errors"].append(str(e))
            migration_report["completed_at"] = datetime.now(timezone.utc).isoformat()
            return migration_report
    
    async def _migrate_batch(self, batch_df: pd.DataFrame) -> Dict[str, Any]:
        """Migre un batch de données."""
        batch_results = {
            "migrated": 0,
            "failed": 0,
            "errors": []
        }
        
        for _, row in batch_df.iterrows():
            try:
                # Extraction des données legacy
                text = self._extract_text(row)
                metadata = self._extract_metadata(row)
                
                if not text:
                    continue
                
                # Traitement avec l'architecture enterprise
                review = await self.enterprise_service.process_review(text, metadata)
                
                batch_results["migrated"] += 1
                
            except Exception as e:
                batch_results["failed"] += 1
                batch_results["errors"].append(f"Row {row.name}: {str(e)}")
                logger.warning(f"Failed to migrate row {row.name}: {e}")
        
        return batch_results
    
    def _extract_text(self, row: pd.Series) -> Optional[str]:
        """Extrait le texte de l'avis depuis les données legacy."""
        # Tentative de détection automatique des colonnes
        text_columns = ['text', 'review', 'comment', 'content', 'message', 'description']
        
        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()
        
        # Si aucune colonne standard, prendre la première colonne string non-vide
        for col in row.index:
            if isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
        
        return None
    
    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extrait les métadonnées depuis les données legacy."""
        metadata = {}
        
        # Colonnes communes à extraire
        metadata_columns = [
            'source', 'channel', 'user_id', 'product_id', 
            'timestamp', 'date', 'rating', 'score'
        ]
        
        for col in metadata_columns:
            if col in row.index and pd.notna(row[col]):
                metadata[col] = row[col]
        
        # Ajout d'informations de migration
        metadata['migrated_from_legacy'] = True
        metadata['migration_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return metadata
    
    async def compare_legacy_vs_enterprise(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Compare les résultats entre l'ancien et le nouveau système.
        
        Args:
            sample_texts: Textes de test
            
        Returns:
            Rapport de comparaison
        """
        if not self._initialized:
            await self.initialize()
        
        comparison_report = {
            "sample_size": len(sample_texts),
            "comparisons": [],
            "accuracy_metrics": {},
            "performance_metrics": {}
        }
        
        if not self.legacy_analyzer:
            logger.warning("Legacy analyzer not available for comparison")
            return comparison_report
        
        for i, text in enumerate(sample_texts):
            try:
                # Analyse legacy
                start_time = datetime.now()
                legacy_result = self._analyze_with_legacy(text)
                legacy_time = (datetime.now() - start_time).total_seconds()
                
                # Analyse enterprise
                start_time = datetime.now()
                enterprise_review = await self.enterprise_service.process_review(text)
                enterprise_time = (datetime.now() - start_time).total_seconds()
                
                # Comparaison
                comparison = {
                    "text_id": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "legacy": {
                        "sentiment": legacy_result.get("sentiment", "unknown"),
                        "score": legacy_result.get("score", 0),
                        "processing_time": legacy_time
                    },
                    "enterprise": {
                        "sentiment": enterprise_review.sentiment_label.value,
                        "score": float(enterprise_review.sentiment_score.value),
                        "category": enterprise_review.category.value,
                        "criticality": enterprise_review.criticality_level.value,
                        "confidence": float(enterprise_review.confidence_score.value),
                        "processing_time": enterprise_time
                    }
                }
                
                comparison_report["comparisons"].append(comparison)
                
            except Exception as e:
                logger.error(f"Comparison failed for text {i}: {e}")
        
        # Calcul des métriques
        if comparison_report["comparisons"]:
            comparison_report["accuracy_metrics"] = self._calculate_accuracy_metrics(
                comparison_report["comparisons"]
            )
            comparison_report["performance_metrics"] = self._calculate_performance_metrics(
                comparison_report["comparisons"]
            )
        
        return comparison_report
    
    def _analyze_with_legacy(self, text: str) -> Dict[str, Any]:
        """Analyse avec le système legacy."""
        try:
            if hasattr(self.legacy_analyzer, 'analyze_sentiment'):
                result = self.legacy_analyzer.analyze_sentiment(text)
                return {
                    "sentiment": result.get("label", "neutral"),
                    "score": result.get("score", 0)
                }
            else:
                # Simulation basique si la méthode n'existe pas
                return {"sentiment": "neutral", "score": 0}
        except Exception as e:
            logger.error(f"Legacy analysis failed: {e}")
            return {"sentiment": "error", "score": 0}
    
    def _calculate_accuracy_metrics(self, comparisons: List[Dict]) -> Dict[str, float]:
        """Calcule les métriques de précision."""
        if not comparisons:
            return {}
        
        # Mapping des sentiments pour comparaison
        sentiment_mapping = {
            "positive": "positive",
            "negative": "negative", 
            "neutral": "neutral",
            "pos": "positive",
            "neg": "negative",
            "neu": "neutral"
        }
        
        agreements = 0
        total = 0
        
        for comp in comparisons:
            legacy_sentiment = sentiment_mapping.get(
                comp["legacy"]["sentiment"].lower(), 
                comp["legacy"]["sentiment"].lower()
            )
            enterprise_sentiment = comp["enterprise"]["sentiment"].lower()
            
            if legacy_sentiment == enterprise_sentiment:
                agreements += 1
            total += 1
        
        return {
            "sentiment_agreement_rate": agreements / total if total > 0 else 0,
            "total_comparisons": total
        }
    
    def _calculate_performance_metrics(self, comparisons: List[Dict]) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        if not comparisons:
            return {}
        
        legacy_times = [comp["legacy"]["processing_time"] for comp in comparisons]
        enterprise_times = [comp["enterprise"]["processing_time"] for comp in comparisons]
        
        return {
            "average_legacy_time": sum(legacy_times) / len(legacy_times),
            "average_enterprise_time": sum(enterprise_times) / len(enterprise_times),
            "speed_improvement_ratio": (
                sum(legacy_times) / sum(enterprise_times) 
                if sum(enterprise_times) > 0 else 1
            )
        }
    
    async def validate_migration(self) -> Dict[str, Any]:
        """Valide la migration en vérifiant l'intégrité des données."""
        if not self._initialized:
            await self.initialize()
        
        validation_report = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "overall_status": "unknown"
        }
        
        try:
            # Vérification des services
            config = await get_service_configuration()
            health = await config.health_check()
            validation_report["checks"]["services_health"] = health["overall_status"]
            
            # Vérification des données
            review_repo = inject(ReviewRepository)
            
            # Comptage des avis
            total_reviews = await review_repo.count_by_criteria()
            validation_report["checks"]["total_reviews"] = total_reviews
            
            # Vérification de la distribution des sentiments
            positive_count = await review_repo.count_by_criteria(
                sentiment_label=SentimentLabel.POSITIVE
            )
            negative_count = await review_repo.count_by_criteria(
                sentiment_label=SentimentLabel.NEGATIVE
            )
            neutral_count = await review_repo.count_by_criteria(
                sentiment_label=SentimentLabel.NEUTRAL
            )
            
            validation_report["checks"]["sentiment_distribution"] = {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "total": positive_count + negative_count + neutral_count
            }
            
            # Vérification des catégories
            category_counts = {}
            for category in SupplyChainCategory:
                count = await review_repo.count_by_criteria(category=category)
                category_counts[category.value] = count
            
            validation_report["checks"]["category_distribution"] = category_counts
            
            # Détermination du statut global
            if (health["overall_status"] == "healthy" and 
                total_reviews > 0 and 
                (positive_count + negative_count + neutral_count) == total_reviews):
                validation_report["overall_status"] = "valid"
            else:
                validation_report["overall_status"] = "issues_detected"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_report["checks"]["validation_error"] = str(e)
            validation_report["overall_status"] = "failed"
        
        return validation_report


async def main():
    """Point d'entrée principal pour la migration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Enterprise Migration Tool")
    print("============================")
    
    # Initialisation du service de migration
    migration_service = LegacyIntegrationService()
    await migration_service.initialize()
    
    # Menu interactif
    while True:
        print("\nOptions disponibles:")
        print("1. Migrer des données depuis un fichier")
        print("2. Comparer legacy vs enterprise")
        print("3. Valider la migration")
        print("4. Quitter")
        
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == "1":
            data_path = input("Chemin vers le fichier de données: ").strip()
            if Path(data_path).exists():
                print(f"\n🔄 Migration en cours depuis {data_path}...")
                report = await migration_service.migrate_data_from_legacy(data_path)
                
                print(f"\n📊 Rapport de migration:")
                print(f"   - Records totaux: {report['total_records']}")
                print(f"   - Records migrés: {report['migrated_records']}")
                print(f"   - Records échoués: {report['failed_records']}")
                print(f"   - Taux de succès: {report['success_rate']:.2%}")
                
                if report['errors']:
                    print(f"   - Erreurs: {len(report['errors'])} (voir logs)")
            else:
                print(f"❌ Fichier non trouvé: {data_path}")
        
        elif choice == "2":
            sample_texts = [
                "Excellent produit, livraison rapide !",
                "Service client décevant, problème non résolu.",
                "Produit correct, rien d'exceptionnel."
            ]
            
            print(f"\n🔍 Comparaison sur {len(sample_texts)} exemples...")
            report = await migration_service.compare_legacy_vs_enterprise(sample_texts)
            
            print(f"\n📊 Rapport de comparaison:")
            if report['accuracy_metrics']:
                print(f"   - Accord sentiment: {report['accuracy_metrics']['sentiment_agreement_rate']:.2%}")
            if report['performance_metrics']:
                print(f"   - Temps legacy: {report['performance_metrics']['average_legacy_time']:.3f}s")
                print(f"   - Temps enterprise: {report['performance_metrics']['average_enterprise_time']:.3f}s")
        
        elif choice == "3":
            print(f"\n🔍 Validation de la migration...")
            report = await migration_service.validate_migration()
            
            print(f"\n📊 Rapport de validation:")
            print(f"   - Statut global: {report['overall_status']}")
            print(f"   - Santé des services: {report['checks'].get('services_health', 'unknown')}")
            print(f"   - Total avis: {report['checks'].get('total_reviews', 0)}")
            
            if 'sentiment_distribution' in report['checks']:
                dist = report['checks']['sentiment_distribution']
                print(f"   - Distribution sentiment: +{dist['positive']} ={dist['neutral']} -{dist['negative']}")
        
        elif choice == "4":
            print("\n👋 Migration terminée !")
            break
        
        else:
            print("❌ Choix invalide, veuillez réessayer.")


if __name__ == "__main__":
    asyncio.run(main())
