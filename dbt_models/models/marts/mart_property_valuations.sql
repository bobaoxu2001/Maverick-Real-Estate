-- ─────────────────────────────────────────────────────────
-- mart_property_valuations.sql
-- Mart: Final analytical table for property valuation models
-- ─────────────────────────────────────────────────────────
-- This mart serves as the primary input for the hedonic
-- regression and property valuation ML models. It includes
-- all property features, market context, and derived metrics.
-- ─────────────────────────────────────────────────────────

WITH property_features AS (
    SELECT * FROM {{ ref('int_property_features') }}
),

final AS (
    SELECT
        -- Identifiers
        bbl,
        borough_name,
        neighborhood,
        address,
        zip_code,

        -- Target variable
        sale_price,
        price_per_sqft,
        sale_date,
        sale_year,
        sale_quarter,

        -- Property characteristics (hedonic regression features)
        building_class_category,
        building_class_at_sale,
        gross_sqft,
        land_sqft,
        lot_area_sqft,
        building_area_sqft,
        commercial_area_sqft,
        office_area_sqft,
        retail_area_sqft,
        num_floors,
        total_units,
        commercial_units,
        year_built,
        building_age,
        floor_area_ratio,
        commercial_ratio,
        height_category,
        zoning_district,

        -- Valuation context
        assessed_total,
        assessed_land,
        SAFE_DIVIDE(sale_price, assessed_total) AS sale_to_assessed_ratio,

        -- Geospatial
        latitude,
        longitude,
        dist_to_cbd_km,

        -- Market context
        neighborhood_avg_price,
        neighborhood_median_price,
        neighborhood_sales_volume,
        neighborhood_avg_ppsf,
        price_vs_neighborhood,

        -- Development activity
        zip_permit_count,
        zip_total_permit_value,
        zip_new_building_permits,

        -- Log transforms for modeling
        LN(sale_price + 1) AS log_sale_price,
        LN(GREATEST(gross_sqft, 1)) AS log_gross_sqft,
        LN(GREATEST(assessed_total, 1)) AS log_assessed_total,

        -- Ownership
        owner_name

    FROM property_features
    WHERE
        -- Final quality filters
        sale_price >= {{ var('min_sale_price') }}
        AND gross_sqft > 0
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
)

SELECT * FROM final
