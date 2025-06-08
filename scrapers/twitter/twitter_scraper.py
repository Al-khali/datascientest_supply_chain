import requests
from bs4 import BeautifulSoup
from datetime import datetime
from scrapers.base_scraper import BaseScraper
import time
import random

class TwitterScraper(BaseScraper):
    def __init__(self):
        super().__init__("twitter")
        self.base_url = "https://twitter.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def fetch_reviews(self, search_query: str, pages: int = 5) -> list:
        """Scrape tweets without API"""
        reviews = []
        url = f"{self.base_url}/search?q={search_query}&src=typed_query&f=live"
        
        for page in range(1, pages + 1):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find tweet elements
                tweet_elements = soup.select('article[data-testid="tweet"]')
                for element in tweet_elements:
                    review_data = self.parse_tweet(element)
                    if review_data:
                        reviews.append(review_data)
                
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"Error scraping Twitter page {page}: {e}")
                continue
                
        return reviews
        
    def parse_tweet(self, element) -> dict:
        """Parse HTML tweet element into structured data"""
        try:
            # Extract tweet ID
            tweet_link = element.select_one('a[href*="/status/"]')
            tweet_id = tweet_link.get('href').split('/')[-1] if tweet_link else ""
            
            # Extract author
            author_element = element.select_one('div[data-testid="User-Name"]')
            author = author_element.text.split('·')[0].strip() if author_element else "Anonymous"
            
            # Extract content
            content_element = element.select_one('div[data-testid="tweetText"]')
            content = content_element.text.strip() if content_element else ""
            
            # Extract date
            date_element = element.select_one('time')
            date_str = date_element.get('datetime') if date_element else ""
            date = datetime.fromisoformat(date_str) if date_str else datetime.now()
            
            return {
                "source": "twitter",
                "source_id": tweet_id,
                "author": author,
                "content": content,
                "date": date.isoformat(),
                "product": "Sephora"
            }
        except Exception as e:
            print(f"Error parsing tweet: {e}")
            return None
