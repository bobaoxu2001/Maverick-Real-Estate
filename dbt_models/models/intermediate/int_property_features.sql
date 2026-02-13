-- ─────────────────────────────────────────────────────────
-- int_property_features.sql
-- Intermediate model: Property features enriched with
-- PLUTO characteristics and market context
-- ─────────────────────────────────────────────────────────
-- Joins sales transactions with property attributes and
-- neighborhood-level market indicators. This forms the
-- primary feature table for ML models.
-- ─────────────────────────────────────────────────────────

WITH sales AS (
    SELECT * FROM {{ ref('stg_sales') }}
),

pluto AS (
    SELECT * FROM {{ ref('stg_pluto') }}
),

permits AS (
    SELECT * FROM {{ ref('stg_permits') }}
),

-- Neighborhood-level aggregated sales statistics
neighborhood_stats AS (
    SELECT
        neighborhood,
        AVG(sale_price) AS neighborhood_avg_price,
        APPROX_QUANTILES(sale_price, 100)[OFFSET(50)] AS neighborhood_median_price,
        COUNT(*) AS neighborhood_sales_volume,
        STDDEV(sale_price) AS neighborhood_price_std,
        AVG(price_per_sqft) AS neighborhood_avg_ppsf
    FROM sales
    GROUP BY neighborhood
),

-- Zip-code-level permit activity (development signal)
permit_activity AS (
    SELECT
        zip_code,
        COUNT(*) AS permit_count,
        SUM(estimated_job_cost) AS total_permit_value,
        COUNT(CASE WHEN job_type = 'NB' THEN 1 END) AS new_building_permits,
        COUNT(CASE WHEN job_type = 'DM' THEN 1 END) AS demolition_permits
    FROM permits
    GROUP BY zip_code
),

-- Join all features together
enriched AS (
    SELECT
        s.*,

        -- PLUTO property characteristics
        p.lot_area_sqft,
        p.building_area_sqft,
        p.commercial_area_sqft,
        p.office_area_sqft,
        p.retail_area_sqft,
        p.num_floors,
        p.assessed_total,
        p.assessed_land,
        p.floor_area_ratio,
        p.building_age,
        p.zoning_district,
        p.building_class,
        p.owner_name,
        p.latitude,
        p.longitude,

        -- Geospatial: Distance to Midtown CBD (Grand Central)
        ST_DISTANCE(
            ST_GEOGPOINT(p.longitude, p.latitude),
            ST_GEOGPOINT(-73.9772, 40.7527)
        ) / 1000 AS dist_to_cbd_km,

        -- Neighborhood context
        ns.neighborhood_avg_price,
        ns.neighborhood_median_price,
        ns.neighborhood_sales_volume,
        ns.neighborhood_price_std,
        ns.neighborhood_avg_ppsf,

        -- Price relative to neighborhood
        SAFE_DIVIDE(s.sale_price, ns.neighborhood_avg_price) AS price_vs_neighborhood,

        -- Development activity
        COALESCE(pa.permit_count, 0) AS zip_permit_count,
        COALESCE(pa.total_permit_value, 0) AS zip_total_permit_value,
        COALESCE(pa.new_building_permits, 0) AS zip_new_building_permits,

        -- Derived features
        CASE
            WHEN p.num_floors <= 5 THEN 'low_rise'
            WHEN p.num_floors <= 15 THEN 'mid_rise'
            WHEN p.num_floors <= 30 THEN 'high_rise'
            WHEN p.num_floors <= 60 THEN 'tower'
            ELSE 'supertall'
        END AS height_category,

        SAFE_DIVIDE(p.commercial_area_sqft, p.building_area_sqft) AS commercial_ratio

    FROM sales s
    LEFT JOIN pluto p ON s.bbl = p.bbl
    LEFT JOIN neighborhood_stats ns ON s.neighborhood = ns.neighborhood
    LEFT JOIN permit_activity pa ON s.zip_code = pa.zip_code
)

SELECT * FROM enriched
