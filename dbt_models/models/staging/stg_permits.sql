-- ─────────────────────────────────────────────────────────
-- stg_permits.sql
-- Staging model for DOB Building Permits
-- Source: NYC Open Data → BigQuery raw layer
-- ─────────────────────────────────────────────────────────
-- Building permit activity serves as a leading indicator for
-- neighborhood development and property value appreciation.
-- ─────────────────────────────────────────────────────────

WITH source AS (
    SELECT *
    FROM {{ source('nyc_opendata', 'dob_permits_raw') }}
),

cleaned AS (
    SELECT
        job_doc_number AS permit_id,
        TRIM(borough) AS borough,
        TRIM(block) AS block,
        TRIM(lot) AS lot,
        TRIM(zip_code) AS zip_code,

        -- Permit Details
        TRIM(job_type) AS job_type,
        CASE TRIM(job_type)
            WHEN 'NB' THEN 'New Building'
            WHEN 'A1' THEN 'Major Alteration'
            WHEN 'A2' THEN 'Minor Alteration'
            WHEN 'A3' THEN 'Minor Alteration (no plans)'
            WHEN 'DM' THEN 'Demolition'
            ELSE 'Other'
        END AS job_type_description,

        TRIM(work_type) AS work_type,
        TRIM(permit_status) AS permit_status,

        -- Dates
        SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%S.%E*S', filing_date) AS filing_date,
        SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%S.%E*S', issuance_date) AS issuance_date,
        EXTRACT(YEAR FROM SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%S.%E*S', filing_date)) AS filing_year,

        -- Cost
        SAFE_CAST(REPLACE(estimated_job_cost, ',', '') AS FLOAT64) AS estimated_job_cost,
        SAFE_CAST(REPLACE(initial_cost, ',', '') AS FLOAT64) AS initial_cost,

        -- Owner
        TRIM(owner_s_first_name) AS owner_first_name,
        TRIM(owner_s_last_name) AS owner_last_name,
        TRIM(owner_s_business_name) AS owner_business_name

    FROM source
    WHERE
        filing_date IS NOT NULL
        AND SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%S.%E*S', filing_date)
            >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {{ var('lookback_years') }} YEAR)
)

SELECT * FROM cleaned
