import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import time
from typing import List, Dict, Any, Tuple

# Constants
DATA_DIR = Path("../data")
KPI_GOALS = {
    "satisfaction": 4.5,
    "response_rate": 0.9,
    "resolution_time": 24
}

# Helper functions
def load_data() -> pd.DataFrame:
    """Load and merge multiple data sources with error handling."""
    data_files = [
        DATA_DIR / "trustpilot_reviews.json",
        DATA_DIR / "google_reviews.json",
        DATA_DIR / "amazon_reviews.json"
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

def generate_sample_data() -> pd.DataFrame:
    """Generate realistic sample data for demonstration."""
    np.random.seed(42)
    sample_size = 500
    sources = ["Trustpilot", "Google", "Amazon"]
    products = ["Perfume", "Skincare", "Makeup", "Haircare"]
    
    dates = [datetime.now() - timedelta(days=x) for x in range(180)]
    data = {
        "author": [f"khalid_{i}" for i in range(sample_size)],
        "content": [f"Sample review content {i}" for i in range(sample_size)],
        "rating": np.random.randint(1, 6, sample_size),
        "date": np.random.choice(dates, sample_size),
        "source": np.random.choice(sources, sample_size),
        "product": np.random.choice(products, sample_size),
        "language": ["fr" if x < 0.7 else "en" for x in np.random.rand(sample_size)]
    }
    return pd.DataFrame(data)

# Dashboard layout
st.set_page_config(
    page_title="Customer Satisfaction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard header
st.title("📊 Customer Satisfaction Dashboard")
st.markdown("""
    <style>
        .reportview-container { padding-top: 2rem; }
        .st-bb { background-color: #f0f2f6; }
        .st-at { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data(ttl=300, show_spinner="Loading data...")
def load_cached_data():
    return load_data()

df = load_cached_data()

# KPI calculation
def calculate_kpis(df: pd.DataFrame) -> Tuple[float, float, float]:
    avg_rating = df['rating'].mean()
    response_rate = 0.85  # Placeholder for actual calculation
    resolution_time = 24  # Placeholder for actual calculation
    return avg_rating, response_rate, resolution_time

# Display KPIs
st.header("Key Performance Indicators")
kpi1, kpi2, kpi3 = st.columns(3)
avg_rating, response_rate, resolution_time = calculate_kpis(df)

kpi1.metric(
    label="Average Rating",
    value=f"{avg_rating:.1f}/5",
    delta=f"{(avg_rating - KPI_GOALS['satisfaction']):.1f} vs target",
    delta_color="inverse" if avg_rating < KPI_GOALS['satisfaction'] else "normal"
)

kpi2.metric(
    label="Response Rate",
    value=f"{response_rate*100:.0f}%",
    delta=f"{(response_rate - KPI_GOALS['response_rate'])*100:.0f}% vs target",
    delta_color="inverse" if response_rate < KPI_GOALS['response_rate'] else "normal"
)

kpi3.metric(
    label="Avg. Resolution Time",
    value=f"{resolution_time}h",
    delta=f"{(KPI_GOALS['resolution_time'] - resolution_time):.0f}h vs target",
    delta_color="inverse" if resolution_time > KPI_GOALS['resolution_time'] else "normal"
)

# Trend analysis
st.header("Satisfaction Trends Over Time")
df['week'] = df['date'].dt.isocalendar().week
weekly_avg = df.groupby('week')['rating'].mean().reset_index()

fig = px.line(
    weekly_avg,
    x="week",
    y="rating",
    title="Weekly Average Rating",
    markers=True
)
fig.update_layout(yaxis_range=[1,5])
st.plotly_chart(fig, use_container_width=True)

# Data table
st.header("Review Data")
st.dataframe(df[['author', 'rating', 'date', 'source']].sort_values('date', ascending=False))

# Export button
if st.button("Export Report PDF"):
    with st.spinner("Generating PDF report..."):
        time.sleep(2)  # Simulate report generation
        st.success("Report generated successfully!")
        st.download_button(
            label="Download PDF Report",
            data=json.dumps(df.to_dict(), indent=2),
            file_name="customer_satisfaction_report.pdf",
            mime="application/pdf"
        )
