"""
Script de nettoyage, anonymisation et structuration avancée des avis clients supply chain.
- Chargement des avis collectés (CSV)
- Nettoyage des textes (suppression HTML, ponctuation, etc.)
- Anonymisation RGPD (hash ou suppression des auteurs)
- Structuration des champs (dates, notes, typage)
- Export d'un fichier propre prêt pour l'analyse NLP
"""
import pandas as pd
import re
import hashlib
from datetime import datetime

INPUT_PATH = '../data/avis_trustpilot.csv'
OUTPUT_PATH = '../data/avis_clean.csv'

# Nettoyage du texte
def clean_text(text):
    if pd.isnull(text):
        return ''
    text = re.sub(r'<.*?>', '', text)  # Suppression HTML
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Suppression caractères spéciaux
    text = re.sub(r'\s+', ' ', text)  # Espaces multiples
    return text.strip()

# Anonymisation RGPD (hash de l'auteur)
def anonymize_author(author):
    if pd.isnull(author):
        return ''
    return hashlib.sha256(author.encode()).hexdigest()

# Structuration de la date
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str).strftime('%Y-%m-%d')
    except Exception:
        return ''

def main():
    df = pd.read_csv(INPUT_PATH)
    if 'auteur' in df.columns:
        df['auteur'] = df['auteur'].apply(anonymize_author)
    if 'commentaire' in df.columns:
        df['commentaire'] = df['commentaire'].apply(clean_text)
    if 'date' in df.columns:
        df['date'] = df['date'].apply(parse_date)
    # Typage des notes
    if 'note' in df.columns:
        df['note'] = pd.to_numeric(df['note'], errors='coerce')
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fichier nettoyé et anonymisé exporté : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
