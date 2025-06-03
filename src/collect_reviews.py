"""
Script de base pour collecter des avis clients (exemple fictif).
A adapter pour chaque source (Trustpilot, site distributeur, etc.)
"""
import requests
import pandas as pd

# Exemple : récupération d'avis fictifs (à remplacer par du vrai scraping ou une API)
def get_sample_reviews():
    data = [
        {"auteur": "Alice", "commentaire": "Livraison rapide, produit conforme.", "note": 5},
        {"auteur": "Bob", "commentaire": "Produit abîmé à la réception.", "note": 2},
        {"auteur": "Chloé", "commentaire": "Service client réactif.", "note": 4},
    ]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = get_sample_reviews()
    print(df)
    df.to_csv("../data/avis_sample.csv", index=False)
