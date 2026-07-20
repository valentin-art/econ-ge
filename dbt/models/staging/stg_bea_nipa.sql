{{ config(materialized='view') }}

-- One-to-one with silver.bea_nipa. Column names already match project
-- conventions (BeaSilverRow is the schema's single source of truth), so no
-- renaming here — just a light derived ratio and a transform timestamp.

select
    year,

    cap_it_nom,
    cap_nonit_nom,
    cap_it_real,
    cap_nonit_real,
    cap_it_nom_idx,
    cap_nonit_nom_idx,
    cap_it_real_idx,
    cap_nonit_real_idx,

    rent_it_nom,
    rent_nonit_nom,
    rent_it_real,
    rent_nonit_real,

    share_it,
    share_nonit,

    delta_it,
    delta_nonit,

    inv_it_nom,
    inv_nonit_nom,
    inv_it_real,
    inv_nonit_real,

    r_t,

    y_nom,
    y_real,
    y_nom_idx,
    y_real_idx,
    p_output,
    pi_output,

    -- Capital deepening: real capital per unit of real output.
    loaded_at,

    (cap_it_real + cap_nonit_real) / nullif(y_real, 0) as capital_deepening,
    current_timestamp as transformed_at

from {{ source('silver', 'bea_nipa') }}
