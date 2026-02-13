-- ─────────────────────────────────────────────────────────
-- mart_market_segments.sql
-- Mart: Neighborhood-level market segmentation table
-- ─────────────────────────────────────────────────────────
-- Aggregated market metrics by neighborhood for clustering
-- analysis and market segmentation models. Provides the
-- input data for K-means/DBSCAN segmentation of NYC
-- sub-markets.
-- ─────────────────────────────────────────────────────────

WITH property_features AS (
    SELECT * FROM {{ ref('int_property_features') }}
),

neighborhood_metrics AS (
    SELECT
        neighborhood,
        borough_name,

        -- Price metrics
        AVG(sale_price) AS avg_sale_price,
        APPROX_QUANTILES(sale_price, 100)[OFFSET(50)] AS median_sale_price,
        STDDEV(sale_price) AS std_sale_price,
        AVG(price_per_sqft) AS avg_ppsf,
        APPROX_QUANTILES(price_per_sqft, 100)[OFFSET(50)] AS median_ppsf,

        -- Volume
        COUNT(*) AS total_transactions,
        COUNT(DISTINCT sale_year) AS years_with_sales,

        -- Property characteristics (averages)
        AVG(gross_sqft) AS avg_gross_sqft,
        AVG(building_age) AS avg_building_age,
        AVG(num_floors) AS avg_num_floors,
        AVG(floor_area_ratio) AS avg_far,
        AVG(commercial_ratio) AS avg_commercial_ratio,

        -- Property mix
        COUNT(CASE WHEN height_category = 'low_rise' THEN 1 END) AS low_rise_count,
        COUNT(CASE WHEN height_category = 'mid_rise' THEN 1 END) AS mid_rise_count,
        COUNT(CASE WHEN height_category = 'high_rise' THEN 1 END) AS high_rise_count,
        COUNT(CASE WHEN height_category IN ('tower', 'supertall') THEN 1 END) AS tower_count,

        -- Location
        AVG(dist_to_cbd_km) AS avg_dist_to_cbd_km,
        AVG(latitude) AS centroid_lat,
        AVG(longitude) AS centroid_lon,

        -- Development activity
        AVG(zip_permit_count) AS avg_permit_activity,
        AVG(zip_new_building_permits) AS avg_new_building_permits,

        -- Price trends (year-over-year)
        AVG(CASE WHEN sale_year = EXTRACT(YEAR FROM CURRENT_DATE()) THEN price_per_sqft END) AS current_year_ppsf,
        AVG(CASE WHEN sale_year = EXTRACT(YEAR FROM CURRENT_DATE()) - 1 THEN price_per_sqft END) AS prior_year_ppsf

    FROM property_features
    GROUP BY neighborhood, borough_name
    HAVING COUNT(*) >= 10  -- Minimum transaction threshold for statistical significance
),

final AS (
    SELECT
        *,

        -- YoY price change
        SAFE_DIVIDE(
            current_year_ppsf - prior_year_ppsf,
            prior_year_ppsf
        ) AS yoy_ppsf_change,

        -- Price volatility (coefficient of variation)
        SAFE_DIVIDE(std_sale_price, avg_sale_price) AS price_cv,

        -- Density score
        SAFE_DIVIDE(avg_gross_sqft * total_transactions, 1000000) AS transaction_volume_score

    FROM neighborhood_metrics
)

SELECT * FROM final
ORDER BY avg_sale_price DESC
