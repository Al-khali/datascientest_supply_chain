# Référence projet :
Le projet doit suivre le cahier des charges Sephora (voir cahier_des_charges_sample.md) comme fil conducteur, en cohérence avec les exigences, livrables et KPIs définis, ainsi que tous les paramètres précédemment vus.

# TODO - Projet Satisfaction client

## Cadrage & Discovery
- [ ] Lire la documentation métier
- [ ] Réaliser des entretiens/questionnaires
- [ ] Lister les besoins/difficultés/attentes
- [ ] Définir 3-4 KPIs pertinents
- [ ] Cartographier les ressources (humaines, techniques, données)
- [ ] Rédiger le guide d’entretien et l’expérience map

## Veille technologique & SWOT
- [ ] Rechercher des solutions concurrentes et outils IA
- [ ] Lire des articles/papiers scientifiques
- [ ] Identifier les contraintes réglementaires (RGPD, etc.)
- [ ] Réaliser une analyse SWOT
- [ ] Rédiger le benchmark et la veille

## Conception du MVP
- [ ] Lister/prioriser les fonctionnalités du MVP
- [ ] Définir le schéma de la base de données
- [ ] Prendre en compte contraintes RSE, réglementaires, accessibilité
- [ ] Rédiger la spécification du MVP

## Challenger le MVP
- [ ] Évaluer la qualité et la complétude des données
- [ ] Lister les risques éthiques, techniques, business
- [ ] Définir une stratégie de pricing (si besoin)
- [ ] Prévoir les besoins de formation/recrutement
- [ ] Rédiger la cartographie des risques

## Développement technique
- [ ] Web scraping des avis/notes/feedbacks
- [ ] Organisation de la donnée (base, pipeline ETL)
- [ ] Implémenter le modèle ML (classification/sentiment analysis)
- [ ] Créer le dashboard de restitution
- [ ] Créer l’API (FastAPI/Flask)
- [ ] Dockeriser le projet
- [ ] Automatiser le déploiement (CI/CD)
- [ ] Mettre en place le monitoring
- [x] Développement du pipeline de collecte réelle d’avis clients (Trustpilot/API) avec stockage structuré (CSV).
- [x] Développement du pipeline de nettoyage, anonymisation et structuration avancée des données collectées (clean_reviews.py)
- [x] Implémentation de l’analyse de sentiment et extraction des motifs d’insatisfaction (NLP)
- [x] Développement du dashboard interactif (Streamlit/Dash) avec filtres et exports
- [x] Création de l’API REST pour intégration SI
- [x] Dockerisation, CI/CD, monitoring

## Tests & Lancement
- [ ] Réaliser les tests unitaires, intégration, utilisateurs
- [ ] Préparer les supports de formation et de communication
- [ ] Lancer la solution

## Soutenance & Amélioration continue
- [ ] Préparer la soutenance (slides, démo)
- [ ] Recueillir les feedbacks
- [ ] Planifier les évolutions

# Suivi des étapes projet (aligné mentor Dan)
- [ ] Étape 0 : Cadrage (présentation équipe, planification)
- [ ] Étape 1 : Rapport sources de données + exemples collectés
- [ ] Étape 2 : Cahier des charges, veille technologique, SWOT
- [ ] Étape 3 : KPIs & Roadmap détaillée
- [ ] Étape 4 : Organisation, ingestion, bases SQL/NoSQL, schéma
- [ ] Étape 5 : Algorithme ML/DL, MLFlow, mesure dérive
- [ ] Étape 6 : API, Dockerisation, microservices
- [ ] Étape 7 : Déploiement, monitoring, features prod

> À chaque étape, le fichier TODO sera mis à jour pour refléter l’avancement, dans une démarche professionnelle d’expert data engineer/data scientist.
