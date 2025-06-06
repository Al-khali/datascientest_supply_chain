import requests
from scrapers.base_scraper import BaseScraper

class TrustpilotScraper(BaseScraper):
    def __init__(self, api_key: str):
        super().__init__("trustpilot")
        self.api_key = api_key
        self.base_url = "https://api.trustpilot.com/v1/business-units"
        self.headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        
    def fetch_reviews(self, business_unit_id: str, limit: int = 100) -> list:
        """Fetch reviews from Trustpilot API"""
        reviews = []
        url = f"{self.base_url}/{business_unit_id}/reviews"
        params = {
            "perPage": min(limit, 100),
            "orderBy": "createdat.desc"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            reviews = data.get("reviews", [])
        except Exception as e:
            print(f"Error fetching Trustpilot reviews: {e}")
            
        return reviews
        
    def parse_review(self, review_data: dict) -> dict:
        """Parse Trustpilot review data into our schema"""
        return {
            "source": "trustpilot",
            "source_id": review_data.get("id"),
            "author": review_data.get("consumer", {}).get("displayName"),
            "rating": review_data.get("stars"),
            "content": review_data.get("text"),
            "date": review_data.get("createdAt"),
            "product": review_data.get("brand", {}).get("name")
        }
