import requests
from datetime import datetime
from scrapers.base_scraper import BaseScraper

class GoogleScraper(BaseScraper):
    def __init__(self, api_key: str):
        super().__init__("google")
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place/details/json"
        
    def fetch_reviews(self, place_id: str, limit: int = 100) -> list:
        """Fetch reviews from Google Places API"""
        reviews = []
        params = {
            "place_id": place_id,
            "fields": "review",
            "key": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if "result" in data and "reviews" in data["result"]:
                reviews = data["result"]["reviews"][:limit]
        except Exception as e:
            print(f"Error fetching Google reviews: {e}")
            
        return reviews
        
    def parse_review(self, review_data: dict) -> dict:
        """Parse Google review data into our schema"""
        return {
            "source": "google",
            "source_id": str(review_data.get("time")),
            "author": review_data.get("author_name"),
            "rating": review_data.get("rating"),
            "content": review_data.get("text"),
            "date": datetime.fromtimestamp(review_data.get("time")).isoformat(),
            "product": "Sephora"
        }
