import os
import requests
from datetime import datetime
from scrapers.base_scraper import BaseScraper

class TwitterScraper(BaseScraper):
    def __init__(self, bearer_token: str):
        super().__init__("twitter")
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2/tweets/search/recent"
        
    def fetch_tweets(self, query: str, limit: int = 100) -> list:
        """Fetch tweets using Twitter API v2"""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "SephoraCSAT/1.0"
        }
        params = {
            "query": f"{query} lang:en -is:retweet",
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,public_metrics",
            "expansions": "author_id"
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"Error fetching Twitter data: {e}")
            return []
            
    def parse_tweet(self, tweet: dict, user_map: dict) -> dict:
        """Parse tweet data into our review schema"""
        user = user_map.get(tweet.get("author_id"), {})
        return {
            "source": "twitter",
            "source_id": tweet.get("id"),
            "author": user.get("username", "unknown"),
            "rating": None,  # Twitter doesn't have ratings
            "content": tweet.get("text"),
            "date": tweet.get("created_at"),
            "product": "Sephora",
            "metrics": tweet.get("public_metrics", {})
        }
        
    def fetch_reviews(self, product: str, limit: int = 100) -> list:
        """Fetch and parse tweets for a product"""
        tweets = self.fetch_tweets(f"#{product} OR @{product}", limit)
        if not tweets:
            return []
            
        # Get user information
        user_ids = [tweet.get("author_id") for tweet in tweets]
        users = self.fetch_users(user_ids)
        user_map = {user["id"]: user for user in users}
        
        return [self.parse_tweet(tweet, user_map) for tweet in tweets]
        
    def fetch_users(self, user_ids: list) -> list:
        """Fetch user information for given IDs"""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "SephoraCSAT/1.0"
        }
        params = {
            "ids": ",".join(user_ids),
            "user.fields": "username"
        }
        
        try:
            response = requests.get(
                "https://api.twitter.com/2/users",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return []
