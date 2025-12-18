-- PostgreSQL Initialization Script

CREATE DATABASE "sentiment-analysis";

CREATE TABLE IF NOT EXISTS sentiment_analysis_predictions (
    -- Primary key
    id_str VARCHAR(50) PRIMARY KEY,

    -- Tweet metadata
    conversation_id_str VARCHAR(50),
    created_at TIMESTAMP,
    tweet_url TEXT,
    lang VARCHAR(10),

    -- Tweet content
    full_text TEXT NOT NULL,
    clean_text TEXT NOT NULL,

    -- User information
    user_id_str VARCHAR(50),
    username VARCHAR(100),
    location TEXT,

    -- Tweet engagement metrics
    favorite_count INTEGER DEFAULT 0,
    quote_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    retweet_count INTEGER DEFAULT 0,

    -- Tweet attributes
    image_url TEXT,
    in_reply_to_screen_name VARCHAR(100),

    -- Sentiment prediction results
    pred_label VARCHAR(20) NOT NULL,  -- predicted label (positive, negative, neutral)
    hf_label VARCHAR(20) NOT NULL,    -- huggingface label (LABEL_0, LABEL_1, LABEL_2)
    pred_score FLOAT NOT NULL,        -- confidence score (0-1)

    -- Original label (if available from training data)
    original_label VARCHAR(20),

    -- Prediction metadata
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_run_id VARCHAR(100),

    -- Indexes for common queries
    CONSTRAINT valid_score CHECK (pred_score >= 0 AND pred_score <= 1)
);

-- Create indexes for better query performance
CREATE INDEX idx_created_at ON sentiment_analysis_predictions(created_at);
CREATE INDEX idx_pred_label ON sentiment_analysis_predictions(pred_label);
CREATE INDEX idx_username ON sentiment_analysis_predictions(username);
CREATE INDEX idx_predicted_at ON sentiment_analysis_predictions(predicted_at);

DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
    RAISE NOTICE 'Created database: sentiment-analysis';
    RAISE NOTICE 'Created table: sentiment_analysis_predictions with indexes';
END $$;
