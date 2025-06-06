import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Supply Chain Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    """Load and merge multiple data sources with error handling."""
    data_files = [
        Path("data/trustpilot_reviews.json"),
        Path("data/google_reviews.json"),
        Path("data/amazon_reviews.json")
    ]
    
    dfs = []
    for file_path in data_files:
        if file_path.exists():
            try:
                df = pd.read_json(file_path)
                dfs.append(df)
            except Exception as e:
                st.error(f"Error loading {file_path.name}: {str(e)}")
        else:
            st.warning(f"File not found: {file_path.name}")
    
    if not dfs:
        st.error("No data available. Using sample data.")
        return generate_sample_data()
    
    return pd.concat(dfs, ignore_index=True)

def generate_sample_data():
    """Generate realistic sample data for demonstration."""
    # Sample implementation would be here
    return pd.DataFrame({
        'review_id': [f'R{i}' for i in range(100)],
        'rating': np.random.randint(1, 6, 100),
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'category': np.random.choice(['logistics', 'product', 'service'], 100)
    })

def main():
    st.title("📊 Executive Dashboard - Customer Satisfaction")
    
    # Load data with progress indicator
    with st.spinner('Chargement des données...'):
        df = load_data()
    
    # Calculate KPIs
    avg_rating = df['rating'].mean()
    negative_reviews = df[df['rating'] < 3].shape[0]
    critical_issues = df[df['sentiment_score'] < -0.7].shape[0]
    
    # Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Note Moyenne", f"{avg_rating:.2f}/5")
    col2.metric("Avis Négatifs", negative_reviews)
    col3.metric("Problèmes Critiques", critical_issues, delta="-5% vs mois dernier")
    
    st.divider()
    
    # Sentiment distribution
    st.subheader("Distribution des Sentiments")
    fig = px.histogram(df, x='sentiment_score', nbins=20, 
                      title="Analyse des Sentiments Clients")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
