CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    source_id VARCHAR(100) NOT NULL UNIQUE,
    author VARCHAR(100),
    rating INTEGER,
    content TEXT,
    date TIMESTAMP,
    product VARCHAR(100),
    sentiment_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_source ON reviews(source);
CREATE INDEX IF NOT EXISTS idx_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_sentiment ON reviews(sentiment_score);
