"""
Dashboard interactif Streamlit pour l’analyse de la satisfaction client supply chain.
Affichage des KPIs, filtres, visualisations dynamiques et export des données.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

DATA_PATH = '../data/avis_sentiment.csv'

st.set_page_config(page_title="Dashboard Satisfaction Client Supply Chain", layout="wide")
st.title("Dashboard Satisfaction Client – Supply Chain Sephora")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Filtres
motif = st.sidebar.multiselect("Filtrer par motif d’insatisfaction", options=df['motif'].unique(), default=list(df['motif'].unique()))
note_min, note_max = st.sidebar.slider("Note minimale et maximale", float(df['note'].min()), float(df['note'].max()), (float(df['note'].min()), float(df['note'].max())))
df_filtre = df[(df['motif'].isin(motif)) & (df['note'] >= note_min) & (df['note'] <= note_max)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Note moyenne", f"{df_filtre['note'].mean():.2f}")
col2.metric("% Avis négatifs", f"{(df_filtre['sentiment']<0).mean()*100:.1f}%")
col3.metric("Avis filtrés", len(df_filtre))

# Visualisation : Répartition des motifs
fig1 = px.histogram(df_filtre, x='motif', color='motif', title="Répartition des motifs d’insatisfaction")
st.plotly_chart(fig1, use_container_width=True)

# Visualisation : Distribution des notes
fig2 = px.histogram(df_filtre, x='note', nbins=10, title="Distribution des notes")
st.plotly_chart(fig2, use_container_width=True)

# Export CSV
st.download_button("Exporter les données filtrées (CSV)", df_filtre.to_csv(index=False), file_name="avis_filtrés.csv")
