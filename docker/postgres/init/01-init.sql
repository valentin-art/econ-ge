CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS dbt_artifacts;
CREATE SCHEMA IF NOT EXISTS experiments;

CREATE TABLE IF NOT EXISTS silver.bea_nipa (
    id SERIAL PRIMARY KEY,
    year INT NOT NULL,
    cap_it_nom REAL,
    cap_nonit_nom REAL,
    cap_it_real REAL,
    cap_nonit_real REAL,
    cap_it_nom_idx REAL,
    cap_nonit_nom_idx REAL,
    cap_it_real_idx REAL,
    cap_nonit_real_idx REAL,
    rent_it_nom REAL,
    rent_nonit_nom REAL,
    rent_it_real REAL,
    rent_nonit_real REAL,
    share_it REAL,
    share_nonit REAL,
    delta_it REAL,
    delta_nonit REAL,
    inv_it_nom REAL,
    inv_nonit_nom REAL,
    inv_it_real REAL,
    inv_nonit_real REAL,
    r_t REAL,
    y_nom REAL,
    y_real REAL,
    y_nom_idx REAL,
    y_real_idx REAL,
    p_output REAL,
    pi_output REAL,
);
