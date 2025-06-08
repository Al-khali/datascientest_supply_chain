import os
import time
from dotenv import load_dotenv
from scrapers.trustpilot.trustpilot_scraper import TrustpilotScraper
from scrapers.google.google_scraper import GoogleScraper
from scrapers.twitter.twitter_scraper import TwitterScraper

# Load environment variables
load_dotenv()

def main():
    print("Starting scrapers...")
    
    # Initialize scrapers
    trustpilot = TrustpilotScraper()
    google = GoogleScraper(api_key=os.getenv("GOOGLE_API_KEY"))
    twitter = TwitterScraper()
    
    # Scrape Sephora reviews from Trustpilot
    print("Scraping Trustpilot...")
    trustpilot_reviews = trustpilot.fetch_reviews(business_name="sephora", pages=3)
    print(f"Fetched {len(trustpilot_reviews)} reviews from Trustpilot")
    
    # Scrape Sephora reviews from Google
    print("Scraping Google...")
    google_reviews = google.fetch_reviews(place_id="ChIJq6qqqqqqqRkRwA2Z2Z2Z2Z2Y", limit=100)
    print(f"Fetched {len(google_reviews)} reviews from Google")
    
    # Scrape Sephora mentions from Twitter
    print("Scraping Twitter...")
    twitter_mentions = twitter.fetch_reviews("Sephora", pages=5)
    print(f"Fetched {len(twitter_mentions)} mentions from Twitter")
    
    # Combine all reviews
    all_reviews = trustpilot_reviews + google_reviews + twitter_mentions
    
    # Save to JSON
    trustpilot.save_to_json(trustpilot_reviews, "sephora_trustpilot")
    google.save_to_json(google_reviews, "sephora_google")
    twitter.save_to_json(twitter_mentions, "sephora_twitter")
    
    print(f"Total {len(all_reviews)} items stored in JSON files")
    print("Scraping completed successfully")

if __name__ == "__main__":
    main()
