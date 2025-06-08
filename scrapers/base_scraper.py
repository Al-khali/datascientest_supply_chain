import abc
import json
import os
from datetime import datetime
from typing import Dict, Any

class BaseScraper(abc.ABC):
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.data_dir = os.path.join(os.getcwd(), "data", "raw", source_name)
        os.makedirs(self.data_dir, exist_ok=True)
        
    @abc.abstractmethod
    def fetch_reviews(self, *args, **kwargs) -> list:
        """Fetch reviews from the source API or website"""
        pass
        
    def save_to_json(self, data: list, filename: str):
        """Save data to JSON file"""
        filepath = os.path.join(self.data_dir, f"{filename}.json")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(data)} items to {filepath}")
        return filepath
        
    def store_to_db(self, reviews: list):
        """Store reviews to databases (PostgreSQL, Elasticsearch)"""
        # Store in PostgreSQL
        self.store_to_postgres(reviews)
        
        # Index in Elasticsearch
        self.index_in_elasticsearch(reviews)
        
    def store_to_postgres(self, reviews: list):
        """Store reviews in PostgreSQL database"""
        import psycopg2
        from psycopg2 import sql
        
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        
        for review in reviews:
            query = sql.SQL("""
                INSERT INTO reviews (source, source_id, author, rating, content, date, product)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id) DO NOTHING;
            """)
            cur.execute(query, (
                review["source"],
                review["source_id"],
                review["author"],
                review["rating"],
                review["content"],
                review["date"],
                review["product"]
            ))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"Stored {len(reviews)} reviews in PostgreSQL")
        
    def index_in_elasticsearch(self, reviews: list):
        """Index reviews in Elasticsearch"""
        from elasticsearch import Elasticsearch
        
        es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
        
        for review in reviews:
            doc = {
                "source": review["source"],
                "author": review["author"],
                "rating": review["rating"],
                "content": review["content"],
                "date": review["date"],
                "product": review["product"],
                "sentiment_score": review.get("sentiment_score", None)
            }
            es.index(index="reviews", id=review["source_id"], document=doc)
        
        print(f"Indexed {len(reviews)} reviews in Elasticsearch")
