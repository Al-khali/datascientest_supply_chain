import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from scrapers.base_scraper import BaseScraper
import time
import random

class TrustpilotScraper(BaseScraper):
    def __init__(self):
        super().__init__("trustpilot")
        self.base_url = "https://www.trustpilot.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def fetch_reviews(self, business_name: str, pages: int = 3) -> list:
        """Scrape reviews from Trustpilot without API"""
        reviews = []
        url = f"{self.base_url}/review/{business_name}"
        
        for page in range(1, pages + 1):
            page_url = f"{url}?page={page}"
            try:
                response = requests.get(page_url, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find review elements
                review_elements = soup.select('div[data-service-review]')
                for element in review_elements:
                    review_data = self.parse_review(element)
                    if review_data:
                        reviews.append(review_data)
                
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error scraping Trustpilot page {page}: {e}")
                continue
                
        return reviews
        
    def parse_review(self, element) -> dict:
        """Parse HTML review element into structured data"""
        try:
            # Extract review ID
            review_id = element.get('data-service-review') or ""
            
            # Extract author
            author_element = element.select_one('span.typography_heading-xxs__QKBS8')
            author = author_element.text.strip() if author_element else "Anonymous"
            
            # Extract rating
            rating_element = element.select_one('div.star-rating_starRating__4rrcf img')
            rating = int(rating_element.get('alt', '0')[0]) if rating_element else 0
            
            # Extract content
            content_element = element.select_one('p.typography_body-l__KUYFJ')
            content = content_element.text.strip() if content_element else ""
            
            # Extract date
            date_element = element.select_one('time')
            date_str = date_element.get('datetime') if date_element else ""
            date = datetime.fromisoformat(date_str) if date_str else datetime.now()
            
            return {
                "source": "trustpilot",
                "source_id": review_id,
                "author": author,
                "rating": rating,
                "content": content,
                "date": date.isoformat(),
                "product": "Sephora"
            }
        except Exception as e:
            print(f"Error parsing review: {e}")
            return None
