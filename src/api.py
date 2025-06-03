"""
API REST FastAPI pour exposer les résultats d'analyse de satisfaction client supply chain.
Permet de requêter les avis enrichis, les KPIs et les motifs d'insatisfaction.
"""
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
from typing import List, Optional

app = FastAPI(title="API Satisfaction Client Supply Chain")
DATA_PATH = '../data/avis_sentiment.csv'

def load_data():
    return pd.read_csv(DATA_PATH)

@app.get("/avis/")
def get_avis(motif: Optional[str] = None, min_note: float = 0, max_note: float = 5):
    df = load_data()
    if motif:
        df = df[df['motif'] == motif]
    df = df[(df['note'] >= min_note) & (df['note'] <= max_note)]
    return df.to_dict(orient="records")

@app.get("/kpi/")
def get_kpi():
    df = load_data()
    return {
        "note_moyenne": round(df['note'].mean(), 2),
        "pourcentage_negatif": round((df['sentiment'] < 0).mean() * 100, 1),
        "total_avis": int(len(df))
    }

@app.get("/motifs/")
def get_motifs():
    df = load_data()
    return df['motif'].value_counts().to_dict()
