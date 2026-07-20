from dagster import AssetSelection, define_asset_job

bea_full_refresh = define_asset_job(
    name="bea_full_refresh",
    selection=AssetSelection.groups(
        "external", "bronze_bea", "silver_bea", "postgres_load"
    ),
    description="Extract, parse, transform, and load the full BEA panel into Postgres.",
)
