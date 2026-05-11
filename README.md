# General equilibrium model

## Purpose:
To implement full modeling pipeline for a research project. The central part is a macroeconomic model that aims to replicate and analyse evolution in occupational choice of individual workers under technological progress observed in last 60 years.

## Model
Classical macroeconomic model featured by occupational choice of heterogenous workers that is calibrated using a comination of multiple statistical, ML, and optimization techniques.

## Data
Macroeconomic sources:
 - BEA: helps to measure evolution of technological progress, capital and labor prices
 - CPI: inflation

Microeconoic resources:
 - US CENSUS: cross-sectional data containing millions of individual oservations over multiple decades
 - CPS (March extract): cross-sectional data with reach demographic information
 - CPS (May rotating groups): cross-sectional data with reach earnings information
 - CPS (Monthly basic): potentially, helps to make a link between March and May extracts

## Data pipeline
1. Collection
2. Extraction (data validation and storage in structured format)
3. Unification (cleaning, crosswalks, matching, and re-weighting)
4. Normalization and loading to a database as ground truth.
5. Aggregation and transformation for straingtforward analytics in data marts.
6. Transormed data is used as input to the model.
7. Outcomes of the model are stored for further analytics.

## Tools required
- Database: PostgreSQL
- Data transformation: Python/polars
- Pipeline management: dbt
- Orchestration: Dagster
- ML modeling for occupational choice: scikit-learn, PySpark
- GE model optimization: scipy, PySpark
- Other analytics: R
- Reporting: LateX, Quardo.

## Project setup
 - Passwords required
 - docker compose up -d (runs postgres, dbt, dagster and R server in containers)
 - VS code preparation (useful plugins)
 - Help files in each section to understand details of the setup
