# Dockerfile pour projet Analyse Satisfaction Client Supply Chain
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8501 8000

CMD ["sh", "-c", "streamlit run src/dashboard.py & uvicorn src.api:app --host 0.0.0.0 --port 8000"]
