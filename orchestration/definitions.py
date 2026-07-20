from dagster import Definitions, load_assets_from_modules

from orchestration.assets import bea
from orchestration.jobs import bea_full_refresh

all_assets = load_assets_from_modules([bea])

defs = Definitions(
    assets=all_assets,
    jobs=[bea_full_refresh],
)
