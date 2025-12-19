-- ============================================
-- Grafana SQL Queries for MLflow Model Metrics
-- ============================================
-- Database: mlflow (PostgreSQL)
-- Use these queries in Grafana dashboard panels
-- ============================================

-- Query 1: Latest Model Run Metrics
-- ============================================
-- Shows the most recent training run with all metrics
-- Use for: Single Stat or Table panel
-- ============================================
SELECT
    r.run_uuid as run_id,
    r.name as run_name,
    r.status,
    r.start_time,
    r.end_time,
    MAX(CASE WHEN m.key = 'eval_accuracy' THEN m.value END) as eval_accuracy,
    MAX(CASE WHEN m.key = 'eval_f1_macro' THEN m.value END) as eval_f1_macro,
    MAX(CASE WHEN m.key = 'eval_loss' THEN m.value END) as eval_loss,
    MAX(CASE WHEN p.key = 'model_accepted' THEN p.value END) as model_accepted,
    MAX(CASE WHEN t.key = 'model_status' THEN t.value END) as model_status,
    MAX(CASE WHEN p.key = 'epochs' THEN p.value END) as epochs,
    MAX(CASE WHEN p.key = 'learning_rate' THEN p.value END) as learning_rate,
    MAX(CASE WHEN p.key = 'is_incremental' THEN p.value END) as is_incremental
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN metrics m ON r.run_uuid = m.run_uuid
LEFT JOIN params p ON r.run_uuid = p.run_uuid
LEFT JOIN tags t ON r.run_uuid = t.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
GROUP BY r.run_uuid, r.name, r.status, r.start_time, r.end_time
ORDER BY r.start_time DESC
LIMIT 1;

-- Query 2: F1 Macro Score Over Time
-- ============================================
-- Shows F1 score trend across all training runs
-- Use for: Time Series / Line Chart panel
-- ============================================
SELECT
    r.start_time as time,
    m.value as f1_macro_score,
    r.name as run_name,
    CASE
        WHEN m.value >= 0.80 THEN 'Accepted'
        ELSE 'Rejected'
    END as status
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND m.key = 'eval_f1_macro'
    AND r.deleted_time IS NULL
    AND r.start_time >= NOW() - INTERVAL '30 days'  -- Last 30 days
ORDER BY r.start_time ASC;

-- Query 3: Model Acceptance Rate
-- ============================================
-- Shows percentage of accepted vs rejected models
-- Use for: Pie Chart or Gauge panel
-- ============================================
SELECT
    CASE
        WHEN p.value = 'True' THEN 'Accepted'
        ELSE 'Rejected'
    END as status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN params p ON r.run_uuid = p.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND p.key = 'model_accepted'
    AND r.deleted_time IS NULL
    AND r.start_time >= NOW() - INTERVAL '30 days'
GROUP BY p.value;

-- Query 4: Average Training Metrics (Last 7 Days)
-- ============================================
-- Shows average performance metrics
-- Use for: Stat panel with sparklines
-- ============================================
SELECT
    ROUND(AVG(CASE WHEN m.key = 'eval_accuracy' THEN m.value END)::numeric, 4) as avg_accuracy,
    ROUND(AVG(CASE WHEN m.key = 'eval_f1_macro' THEN m.value END)::numeric, 4) as avg_f1_macro,
    ROUND(AVG(CASE WHEN m.key = 'eval_loss' THEN m.value END)::numeric, 4) as avg_loss,
    COUNT(DISTINCT r.run_uuid) as total_runs
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
    AND r.start_time >= NOW() - INTERVAL '7 days';

-- Query 5: Model Performance Comparison
-- ============================================
-- Compares last 10 training runs
-- Use for: Table panel with sorting
-- ============================================
SELECT
    r.start_time,
    r.name as run_name,
    ROUND(MAX(CASE WHEN m.key = 'eval_f1_macro' THEN m.value END)::numeric, 4) as f1_macro,
    ROUND(MAX(CASE WHEN m.key = 'eval_accuracy' THEN m.value END)::numeric, 4) as accuracy,
    ROUND(MAX(CASE WHEN m.key = 'eval_loss' THEN m.value END)::numeric, 4) as loss,
    MAX(CASE WHEN p.key = 'epochs' THEN p.value END) as epochs,
    MAX(CASE WHEN p.key = 'learning_rate' THEN p.value END) as lr,
    MAX(CASE WHEN p.key = 'model_accepted' THEN p.value END) as accepted,
    MAX(CASE WHEN p.key = 'is_incremental' THEN p.value END) as incremental,
    ROUND(EXTRACT(EPOCH FROM (r.end_time - r.start_time)) / 60, 2) as duration_min
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN metrics m ON r.run_uuid = m.run_uuid
LEFT JOIN params p ON r.run_uuid = p.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
GROUP BY r.run_uuid, r.name, r.start_time, r.end_time
ORDER BY r.start_time DESC
LIMIT 10;

-- Query 6: Training Duration Trend
-- ============================================
-- Shows how long each training takes
-- Use for: Bar Chart or Time Series
-- ============================================
SELECT
    r.start_time as time,
    r.name as run_name,
    ROUND(EXTRACT(EPOCH FROM (r.end_time - r.start_time)) / 60, 2) as duration_minutes,
    MAX(CASE WHEN p.key = 'is_incremental' THEN p.value END) as is_incremental
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN params p ON r.run_uuid = p.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
    AND r.end_time IS NOT NULL
    AND r.start_time >= NOW() - INTERVAL '30 days'
GROUP BY r.run_uuid, r.name, r.start_time, r.end_time
ORDER BY r.start_time ASC;

-- Query 7: Latest Run Status (for Alert)
-- ============================================
-- Returns 1 if latest run was accepted, 0 if rejected
-- Use for: Alert rule or Stat panel with threshold
-- ============================================
SELECT
    r.start_time as time,
    CASE
        WHEN p.value = 'True' THEN 1
        ELSE 0
    END as is_accepted,
    m.value as f1_score
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN params p ON r.run_uuid = p.run_uuid AND p.key = 'model_accepted'
LEFT JOIN metrics m ON r.run_uuid = m.run_uuid AND m.key = 'eval_f1_macro'
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
ORDER BY r.start_time DESC
LIMIT 1;

-- Query 8: Incremental vs Fresh Training Performance
-- ============================================
-- Compares performance between training types
-- Use for: Bar Chart (grouped)
-- ============================================
SELECT
    MAX(CASE WHEN p.key = 'is_incremental' THEN p.value END) as training_type,
    ROUND(AVG(CASE WHEN m.key = 'eval_f1_macro' THEN m.value END)::numeric, 4) as avg_f1_macro,
    ROUND(AVG(CASE WHEN m.key = 'eval_accuracy' THEN m.value END)::numeric, 4) as avg_accuracy,
    COUNT(DISTINCT r.run_uuid) as run_count
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
JOIN params p ON r.run_uuid = p.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
    AND r.start_time >= NOW() - INTERVAL '30 days'
GROUP BY p.key, p.value
HAVING p.key = 'is_incremental';

-- Query 9: Training Metrics Detail (for Logs Panel)
-- ============================================
-- Shows all metrics for debugging
-- Use for: Logs panel or detailed table
-- ============================================
SELECT
    r.start_time as time,
    r.name as run_name,
    m.key as metric_name,
    ROUND(m.value::numeric, 4) as metric_value,
    m.step,
    m.timestamp
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
    AND r.start_time >= NOW() - INTERVAL '7 days'
ORDER BY r.start_time DESC, m.key ASC;

-- Query 10: Model Parameters Distribution
-- ============================================
-- Shows what hyperparameters are being used
-- Use for: Table or text panel
-- ============================================
SELECT
    p.key as parameter,
    p.value,
    COUNT(*) as usage_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY p.key), 2) as percentage
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN params p ON r.run_uuid = p.run_uuid
WHERE e.name = 'mbg-sentiment-analysis'
    AND r.deleted_time IS NULL
    AND r.start_time >= NOW() - INTERVAL '30 days'
    AND p.key IN ('epochs', 'learning_rate', 'batch_size', 'freeze_layers')
GROUP BY p.key, p.value
ORDER BY p.key, usage_count DESC;

-- ============================================
-- SENTIMENT ANALYSIS PREDICTIONS QUERIES
-- ============================================
-- Database: sentiment-analysis (PostgreSQL)
-- Table: sentiment_analysis_predictions
-- ============================================

-- Query 11: Sentiment Distribution (Last 7 Days)
-- ============================================
-- Shows percentage of Positive, Negative, Neutral predictions
-- Use for: Pie Chart or Donut Chart panel
-- ============================================
SELECT
    INITCAP(pred_label) as sentiment,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) as percentage
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY pred_label
ORDER BY COUNT(*) DESC;

-- Query 11a: Positive Sentiment Percentage (Last 7 Days)
-- ============================================
-- Shows only positive sentiment percentage
-- Use for: Stat panel
-- ============================================
SELECT
    ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM sentiment_analysis_predictions WHERE predicted_at >= NOW() - INTERVAL '7 days'), 0), 2) as percentage
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
    AND pred_label = 'positive';

-- Query 11b: Negative Sentiment Percentage (Last 7 Days)
-- ============================================
-- Shows only negative sentiment percentage
-- Use for: Stat panel
-- ============================================
SELECT
    ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM sentiment_analysis_predictions WHERE predicted_at >= NOW() - INTERVAL '7 days'), 0), 2) as percentage
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
    AND pred_label = 'negative';

-- Query 11c: Neutral Sentiment Percentage (Last 7 Days)
-- ============================================
-- Shows only neutral sentiment percentage
-- Use for: Stat panel
-- ============================================
SELECT
    ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM sentiment_analysis_predictions WHERE predicted_at >= NOW() - INTERVAL '7 days'), 0), 2) as percentage
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
    AND pred_label = 'neutral';

-- Query 12: Sentiment Trend Over Time (Last 30 Days)
-- ============================================
-- Shows sentiment distribution over time
-- Use for: Time Series / Stacked Area Chart
-- ============================================
SELECT
    DATE_TRUNC('day', predicted_at) as time,
    INITCAP(pred_label) as sentiment,
    COUNT(*) as count
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', predicted_at), pred_label
ORDER BY time ASC, pred_label;

-- Query 13: Average Prediction Confidence by Sentiment
-- ============================================
-- Shows how confident the model is for each sentiment
-- Use for: Stat panel or Bar Chart
-- ============================================
SELECT
    INITCAP(pred_label) as sentiment,
    ROUND(AVG(pred_score)::numeric, 4) as avg_confidence,
    ROUND(MIN(pred_score)::numeric, 4) as min_confidence,
    ROUND(MAX(pred_score)::numeric, 4) as max_confidence,
    COUNT(*) as total_predictions
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY pred_label
ORDER BY avg_confidence DESC;

-- Query 14: Tweet Volume and Engagement by Sentiment
-- ============================================
-- Shows total engagement metrics grouped by sentiment
-- Use for: Table panel
-- ============================================
SELECT
    INITCAP(pred_label) as sentiment,
    COUNT(*) as tweet_count,
    SUM(favorite_count) as total_favorites,
    SUM(retweet_count) as total_retweets,
    SUM(reply_count) as total_replies,
    SUM(quote_count) as total_quotes,
    ROUND(AVG(favorite_count)::numeric, 2) as avg_favorites,
    ROUND(AVG(retweet_count)::numeric, 2) as avg_retweets
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY pred_label
ORDER BY tweet_count DESC;

-- Query 15: Top Users by Tweet Count (Last 7 Days)
-- ============================================
-- Shows most active users with sentiment breakdown
-- Use for: Table panel
-- ============================================
SELECT
    username,
    COUNT(*) as total_tweets,
    SUM(CASE WHEN pred_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
    SUM(CASE WHEN pred_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
    SUM(CASE WHEN pred_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
    ROUND(AVG(pred_score)::numeric, 4) as avg_confidence
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
    AND username IS NOT NULL
GROUP BY username
ORDER BY total_tweets DESC
LIMIT 10;

-- Query 16: Hourly Sentiment Distribution
-- ============================================
-- Shows sentiment patterns throughout the day
-- Use for: Heatmap or Time Series
-- ============================================
SELECT
    DATE_TRUNC('hour', predicted_at) as time,
    INITCAP(pred_label) as sentiment,
    COUNT(*) as count
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', predicted_at), pred_label
ORDER BY time ASC;

-- Query 17: Latest Predictions (Real-time Feed)
-- ============================================
-- Shows most recent predictions with details
-- Use for: Table panel or Logs panel
-- ============================================
SELECT
    predicted_at as time,
    username,
    INITCAP(pred_label) as sentiment,
    ROUND(pred_score::numeric, 4) as confidence,
    LEFT(clean_text, 100) as tweet_preview,
    favorite_count,
    retweet_count,
    model_run_id
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '24 hours'
ORDER BY predicted_at DESC
LIMIT 50;

-- Query 18: Sentiment by Language (Last 7 Days)
-- ============================================
-- Shows sentiment distribution across different languages
-- Use for: Bar Chart (grouped)
-- ============================================
SELECT
    lang as language,
    INITCAP(pred_label) as sentiment,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY lang), 2) as percentage
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
    AND lang IS NOT NULL
GROUP BY lang, pred_label
ORDER BY lang, count DESC;

-- Query 19: Model Performance - Prediction Confidence Distribution
-- ============================================
-- Shows distribution of confidence scores
-- Use for: Histogram
-- ============================================
SELECT
    CASE
        WHEN pred_score >= 0.9 THEN '0.9-1.0 (Very High)'
        WHEN pred_score >= 0.8 THEN '0.8-0.9 (High)'
        WHEN pred_score >= 0.7 THEN '0.7-0.8 (Medium)'
        WHEN pred_score >= 0.6 THEN '0.6-0.7 (Low)'
        ELSE '0.0-0.6 (Very Low)'
    END as confidence_range,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY confidence_range
ORDER BY confidence_range DESC;

-- Query 20: Daily Summary Statistics
-- ============================================
-- Shows daily aggregated metrics
-- Use for: Table or Time Series
-- ============================================
SELECT
    DATE_TRUNC('day', predicted_at) as date,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT username) as unique_users,
    ROUND(AVG(pred_score)::numeric, 4) as avg_confidence,
    SUM(CASE WHEN pred_label = 'positive' THEN 1 ELSE 0 END) as positive,
    SUM(CASE WHEN pred_label = 'negative' THEN 1 ELSE 0 END) as negative,
    SUM(CASE WHEN pred_label = 'neutral' THEN 1 ELSE 0 END) as neutral,
    SUM(favorite_count + retweet_count + reply_count + quote_count) as total_engagement
FROM sentiment_analysis_predictions
WHERE predicted_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', predicted_at)
ORDER BY date DESC;
