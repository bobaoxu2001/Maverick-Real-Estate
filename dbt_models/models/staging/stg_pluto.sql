-- ─────────────────────────────────────────────────────────
-- stg_pluto.sql
-- Staging model for PLUTO (Primary Land Use Tax Lot Output)
-- Source: NYC Open Data → BigQuery raw layer
-- ─────────────────────────────────────────────────────────
-- This model cleans and standardizes the raw PLUTO data,
-- filtering to commercial properties with valid coordinates
-- and assessed values.
-- ─────────────────────────────────────────────────────────

WITH source AS (
    SELECT *
    FROM {{ source('nyc_opendata', 'pluto_raw') }}
),

cleaned AS (
    SELECT
        -- Property Identifiers
        CAST(bbl AS STRING) AS bbl,
        CAST(borocode AS INT64) AS borough_code,
        CASE CAST(borocode AS INT64)
            WHEN 1 THEN 'Manhattan'
            WHEN 2 THEN 'Bronx'
            WHEN 3 THEN 'Brooklyn'
            WHEN 4 THEN 'Queens'
            WHEN 5 THEN 'Staten Island'
        END AS borough_name,
        block,
        lot,
        TRIM(address) AS address,
        TRIM(zipcode) AS zip_code,

        -- Land Use & Classification
        TRIM(landuse) AS land_use_code,
        TRIM(bldgclass) AS building_class,
        TRIM(zonedist1) AS zoning_district,
        TRIM(overlay1) AS zoning_overlay,
        TRIM(ownername) AS owner_name,

        -- Property Dimensions
        SAFE_CAST(lotarea AS FLOAT64) AS lot_area_sqft,
        SAFE_CAST(bldgarea AS FLOAT64) AS building_area_sqft,
        SAFE_CAST(comarea AS FLOAT64) AS commercial_area_sqft,
        SAFE_CAST(resarea AS FLOAT64) AS residential_area_sqft,
        SAFE_CAST(officearea AS FLOAT64) AS office_area_sqft,
        SAFE_CAST(retailarea AS FLOAT64) AS retail_area_sqft,
        SAFE_CAST(numfloors AS FLOAT64) AS num_floors,
        SAFE_CAST(unitstotal AS INT64) AS units_total,
        SAFE_CAST(unitsres AS INT64) AS units_residential,

        -- Temporal
        SAFE_CAST(yearbuilt AS INT64) AS year_built,
        SAFE_CAST(yearalter1 AS INT64) AS year_altered_1,
        SAFE_CAST(yearalter2 AS INT64) AS year_altered_2,

        -- Valuation
        SAFE_CAST(assesstot AS FLOAT64) AS assessed_total,
        SAFE_CAST(assessland AS FLOAT64) AS assessed_land,

        -- Geospatial
        SAFE_CAST(latitude AS FLOAT64) AS latitude,
        SAFE_CAST(longitude AS FLOAT64) AS longitude,

        -- Derived: Floor Area Ratio
        SAFE_DIVIDE(
            SAFE_CAST(bldgarea AS FLOAT64),
            NULLIF(SAFE_CAST(lotarea AS FLOAT64), 0)
        ) AS floor_area_ratio,

        -- Derived: Building Age
        EXTRACT(YEAR FROM CURRENT_DATE()) - SAFE_CAST(yearbuilt AS INT64) AS building_age

    FROM source
    WHERE
        -- Valid coordinates within NYC bounds
        SAFE_CAST(latitude AS FLOAT64) BETWEEN 40.4 AND 41.0
        AND SAFE_CAST(longitude AS FLOAT64) BETWEEN -74.3 AND -73.6
        -- Positive assessed value
        AND SAFE_CAST(assesstot AS FLOAT64) > 0
        -- Commercial land use codes
        AND TRIM(landuse) IN {{ var('commercial_land_use_codes') }}
)

SELECT *
FROM cleaned
QUALIFY ROW_NUMBER() OVER (PARTITION BY bbl ORDER BY assessed_total DESC) = 1
