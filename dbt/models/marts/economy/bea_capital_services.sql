{{ config(
    materialized='table',
    indexes=[{'columns': ['year'], 'unique': true}]
) }}

-- Final BEA capital-services mart: stg_bea_nipa plus year-over-year growth
-- rates. No intermediate layer yet — BEA is the only source today, so there
-- is nothing to share this logic with; revisit once a second source needs
-- the same (year x economy-state) grain (see int_economy_state_yearly in
-- docs/3b_dbt.md).

select
    *,

    -- YoY growth, real series. Single window function — within the
    -- "≤1 window function stays in dbt" threshold from docs/3b_dbt.md.
    y_real / nullif(lag(y_real) over (order by year), 0) - 1 as y_real_growth,
    cap_it_real / nullif(lag(cap_it_real) over (order by year), 0)
    - 1 as cap_it_real_growth,
    cap_nonit_real / nullif(lag(cap_nonit_real) over (order by year), 0)
    - 1 as cap_nonit_real_growth

from {{ ref('stg_bea_nipa') }}
