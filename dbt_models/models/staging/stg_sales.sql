-- ─────────────────────────────────────────────────────────
-- stg_sales.sql
-- Staging model for DOF Rolling Sales
-- Source: NYC Open Data → BigQuery raw layer
-- ─────────────────────────────────────────────────────────
-- Cleans property sale transactions, filtering non-arms-length
-- transfers and standardizing data types.
-- ─────────────────────────────────────────────────────────

WITH source AS (
    SELECT *
    FROM {{ source('nyc_opendata', 'rolling_sales_raw') }}
),

cleaned AS (
    SELECT
        -- Identifiers
        CONCAT(
            CAST(borough AS STRING), '-',
            LPAD(CAST(block AS STRING), 5, '0'), '-',
            LPAD(CAST(lot AS STRING), 4, '0')
        ) AS bbl,
        CAST(borough AS INT64) AS borough_code,
        CASE CAST(borough AS INT64)
            WHEN 1 THEN 'Manhattan'
            WHEN 2 THEN 'Bronx'
            WHEN 3 THEN 'Brooklyn'
            WHEN 4 THEN 'Queens'
            WHEN 5 THEN 'Staten Island'
        END AS borough_name,
        TRIM(neighborhood) AS neighborhood,
        TRIM(address) AS address,
        TRIM(zip_code) AS zip_code,

        -- Property Classification
        TRIM(building_class_at_time_of_sale) AS building_class_at_sale,
        TRIM(building_class_at_present) AS building_class_current,
        TRIM(building_class_category) AS building_class_category,
        TRIM(tax_class_at_time_of_sale) AS tax_class_at_sale,

        -- Transaction
        SAFE_CAST(
            REPLACE(REPLACE(sale_price, ',', ''), '$', '') AS FLOAT64
        ) AS sale_price,
        PARSE_DATE('%m/%d/%Y', sale_date) AS sale_date,
        EXTRACT(YEAR FROM PARSE_DATE('%m/%d/%Y', sale_date)) AS sale_year,
        EXTRACT(QUARTER FROM PARSE_DATE('%m/%d/%Y', sale_date)) AS sale_quarter,
        EXTRACT(MONTH FROM PARSE_DATE('%m/%d/%Y', sale_date)) AS sale_month,

        -- Dimensions
        SAFE_CAST(REPLACE(gross_square_feet, ',', '') AS FLOAT64) AS gross_sqft,
        SAFE_CAST(REPLACE(land_square_feet, ',', '') AS FLOAT64) AS land_sqft,
        SAFE_CAST(residential_units AS INT64) AS residential_units,
        SAFE_CAST(commercial_units AS INT64) AS commercial_units,
        SAFE_CAST(total_units AS INT64) AS total_units,
        SAFE_CAST(year_built AS INT64) AS year_built,

        -- Derived: Price per Square Foot
        SAFE_DIVIDE(
            SAFE_CAST(REPLACE(REPLACE(sale_price, ',', ''), '$', '') AS FLOAT64),
            NULLIF(SAFE_CAST(REPLACE(gross_square_feet, ',', '') AS FLOAT64), 0)
        ) AS price_per_sqft

    FROM source
    WHERE
        -- Filter out non-arms-length transactions (< $100k indicates transfer)
        SAFE_CAST(
            REPLACE(REPLACE(sale_price, ',', ''), '$', '') AS FLOAT64
        ) >= {{ var('min_sale_price') }}
        -- Filter extreme price per sqft (likely data errors)
        AND SAFE_DIVIDE(
            SAFE_CAST(REPLACE(REPLACE(sale_price, ',', ''), '$', '') AS FLOAT64),
            NULLIF(SAFE_CAST(REPLACE(gross_square_feet, ',', '') AS FLOAT64), 0)
        ) BETWEEN 50 AND 50000
)

SELECT * FROM cleaned
