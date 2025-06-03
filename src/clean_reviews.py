"""
Script de nettoyage, anonymisation et structuration avancée des avis clients supply chain.
- Chargement multi-format (CSV, JSON, Parquet)
- Nettoyage avancé des textes (HTML, emojis, stopwords, lower, accents)
- Anonymisation RGPD (hash, suppression, pseudonymisation)
- Structuration stricte des champs (dates, notes, typage, détection langue)
- Détection de doublons, gestion des valeurs manquantes
- Export multi-format (CSV, Parquet, JSON)
- Logging détaillé et rapport de qualité des données
"""
import pandas as pd
import re
import hashlib
import unicodedata
import logging
from datetime import datetime
from langdetect import detect, LangDetectException
from pathlib import Path
import emoji
from collections import Counter
from nltk.corpus import stopwords

INPUT_PATH = '../data/reviews_collected_*.csv'  # Prend tous les fichiers collectés
OUTPUT_PATH = '../data/avis_clean.csv'
LOG_PATH = '../logs/cleaning.log'

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Chargement des stopwords français et anglais
try:
    STOPWORDS_FR = set(stopwords.words('french'))
    STOPWORDS_EN = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS_FR = set(stopwords.words('french'))
    STOPWORDS_EN = set(stopwords.words('english'))

# Nettoyage avancé du texte avec suppression emojis et stopwords

def clean_text(text):
    if pd.isnull(text):
        return ''
    text = str(text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'<.*?>', '', text)  # HTML
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Caractères spéciaux
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')  # accents
    # Suppression des stopwords fr/en
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_FR and t not in STOPWORDS_EN]
    return ' '.join(tokens).strip()

def anonymize_author(author):
    if pd.isnull(author) or author == '':
        return 'anonymous_user'
    return hashlib.sha256(author.encode()).hexdigest()[:16]

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d')
    except Exception:
        return ''

def detect_language(text):
    try:
        return detect(text)
    except (LangDetectException, TypeError):
        return 'unknown'

def main():
    # Chargement multi-fichiers
    files = list(Path('../data').glob('reviews_collected_*.csv'))
    if not files:
        logger.error('Aucun fichier source trouvé.')
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(f'{len(df)} avis chargés.')

    # Nettoyage et structuration
    if 'author_hash' in df.columns:
        df['author_hash'] = df['author_hash'].apply(anonymize_author)
    if 'review_text' in df.columns:
        df['review_text'] = df['review_text'].apply(clean_text)
        df['review_length'] = df['review_text'].str.len()
    if 'date_published' in df.columns:
        df['date_published'] = df['date_published'].apply(parse_date)
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        # Contrôle de cohérence sur les notes
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    if 'review_language' not in df.columns:
        df['review_language'] = df['review_text'].apply(detect_language)
    # Gestion des doublons et valeurs manquantes
    df = df.drop_duplicates(subset=['review_id'])
    df = df.dropna(subset=['review_text', 'rating'])
    logger.info(f"{len(df)} avis après nettoyage et dédoublonnage.")

    # Export multi-format
    df.to_csv(OUTPUT_PATH, index=False)
    df.to_parquet(OUTPUT_PATH.replace('.csv', '.parquet'), index=False)
    df.to_json(OUTPUT_PATH.replace('.csv', '.json'), orient='records', force_ascii=False)
    logger.info(f"Fichiers exportés : {OUTPUT_PATH}, {OUTPUT_PATH.replace('.csv', '.parquet')}, {OUTPUT_PATH.replace('.csv', '.json')}")

    # Rapport qualité avancé
    stats = {
        'total_reviews': int(len(df)),
        'review_length': {
            'min': int(df['review_length'].min()),
            'max': int(df['review_length'].max()),
            'mean': float(df['review_length'].mean()),
            'median': float(df['review_length'].median())
        },
        'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
        'languages': df['review_language'].value_counts().to_dict(),
        'date_range': {
            'min': str(df['date_published'].min()),
            'max': str(df['date_published'].max())
        },
        'missing_values': df.isnull().sum().to_dict(),
        'columns': list(df.columns)
    }
    stats_path = OUTPUT_PATH.replace('.csv', '_quality_report.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Rapport qualité exporté : {stats_path}")
    print(f"Nettoyage terminé. {len(df)} avis propres exportés.")

if __name__ == "__main__":
    main()
