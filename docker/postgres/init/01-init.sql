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
    capital_it REAL,
    capital_non_it REAL,
    p_it REAL,
    p_non_it REAL,
    p_it_real REAL,
    p_non_it_real REAL,
    delta_it REAL,
    delta_non_it REAL,
    r_t REAL,
    r_t_real REAL,
    y_real REAL,
    y_real_idx REAL,
    y_nominal REAL,
    p_output REAL
);
