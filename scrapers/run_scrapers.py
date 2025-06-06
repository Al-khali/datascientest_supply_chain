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
    
    # Initialize scrapers with API keys
    trustpilot = TrustpilotScraper(api_key=os.getenv("TRUSTPILOT_API_KEY"))
    google = GoogleScraper(api_key=os.getenv("GOOGLE_API_KEY"))
    twitter = TwitterScraper(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))
    
    # Scrape Sephora reviews from Trustpilot
    sephora_business_id = "5a4b4b5e0000ff00059b6f5f"  # Sephora's Trustpilot business ID
    trustpilot_raw = trustpilot.fetch_reviews(business_unit_id=sephora_business_id, limit=100)
    trustpilot_parsed = [trustpilot.parse_review(review) for review in trustpilot_raw]
    
    # Scrape Sephora reviews from Google
    sephora_place_id = "ChIJq6qqqqqqqRkRwA2Z2Z2Z2Z2Y"  # Sephora's Google Place ID
    google_raw = google.fetch_reviews(place_id=sephora_place_id, limit=100)
    google_parsed = [google.parse_review(review) for review in google_raw]
    
    # Scrape Sephora mentions from Twitter
    twitter_raw = twitter.fetch_reviews("Sephora", limit=100)
    
    # Combine all reviews
    all_reviews = trustpilot_parsed + google_parsed + twitter_raw
    
    # Save to JSON
    trustpilot.save_to_json(trustpilot_parsed, "sephora_trustpilot")
    google.save_to_json(google_parsed, "sephora_google")
    twitter.save_to_json(twitter_raw, "sephora_twitter")
    
    # Store to databases
    trustpilot.store_to_db(trustpilot_parsed)
    google.store_to_db(google_parsed)
    twitter.store_to_db(twitter_raw)
    
    print(f"Scraped {len(trustpilot_parsed)} reviews from Trustpilot")
    print(f"Scraped {len(google_parsed)} reviews from Google")
    print(f"Scraped {len(twitter_raw)} mentions from Twitter")
    print(f"Total {len(all_reviews)} items stored in databases")
    
    print("Scraping completed successfully")

if __name__ == "__main__":
    main()
