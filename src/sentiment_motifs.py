"""
Script d'analyse de sentiment et extraction des motifs d'insatisfaction sur les avis clients supply chain.
- Chargement des avis nettoyés (CSV)
- Analyse de sentiment (TextBlob)
- Extraction de motifs d'insatisfaction (mots-clés)
- Export d'un fichier enrichi pour le dashboard
"""
import pandas as pd
from textblob import TextBlob
import re

INPUT_PATH = '../data/avis_clean.csv'
OUTPUT_PATH = '../data/avis_sentiment.csv'

# Liste simple de motifs d'insatisfaction (à enrichir selon le contexte)
MOTIFS = {
    'livraison': ['retard', 'livraison', 'colis', 'temps', 'attente', 'non reçu'],
    'produit': ['abîmé', 'cassé', 'défectueux', 'mauvais', 'non conforme'],
    'service client': ['service', 'réponse', 'contact', 'support', 'injoignable'],
    'remboursement': ['remboursement', 'remboursé', 'retour', 'échange'],
}

def detect_motif(comment):
    comment = comment.lower()
    for motif, keywords in MOTIFS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', comment):
                return motif
    return 'autre'

def main():
    df = pd.read_csv(INPUT_PATH)
    # Analyse de sentiment
    df['sentiment'] = df['commentaire'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    # Extraction du motif principal
    df['motif'] = df['commentaire'].apply(detect_motif)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fichier enrichi exporté : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
