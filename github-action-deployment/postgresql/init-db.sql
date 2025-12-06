-- PostgreSQL Initialization Script

CREATE DATABASE "sentiment-analysis";

DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
    RAISE NOTICE 'Created database: sentiment-analysis';
END $$;
