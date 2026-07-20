"""Dagster code location for the dbt project.

Runs inside dbt/'s own Python 3.10 venv, separate from the root project's
environment (Python 3.14) -- dbt-core pins mashumaro<3.15, which fails to
import under 3.14. Wired into the same workspace as src/orchestration via
../workspace.yaml. The load-to-postgres asset in src/orchestration/assets/bea.py
sets its key to ["silver", "bea_nipa"] to match the asset key dagster-dbt
derives from the `silver.bea_nipa` dbt source below, so the two code
locations connect into one lineage graph in the UI.
"""

import os
import sys
from pathlib import Path

from dagster import AssetExecutionContext, AssetSelection, Definitions, define_asset_job
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

# dagster-dbt shells out to `dbt` by name. When this code location is spawned
# by `dagster dev` from the root project, the parent process's PATH puts the
# root venv's `dbt` first -- wrong interpreter (Python 3.14, where dbt-core's
# mashumaro pin doesn't import). Force PATH to resolve `dbt` from this venv.
os.environ["PATH"] = (
    str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
)

dbt_project = DbtProject(project_dir=Path(__file__).parent)
dbt_project.prepare_if_dev()


@dbt_assets(manifest=dbt_project.manifest_path)
def econ_ge_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()


dbt_build = define_asset_job(
    name="dbt_build",
    selection=AssetSelection.assets(econ_ge_dbt_assets),
    description="Runs `dbt build` over the econ_ge project.",
)

defs = Definitions(
    assets=[econ_ge_dbt_assets],
    jobs=[dbt_build],
    resources={
        "dbt": DbtCliResource(
            project_dir=str(dbt_project.project_dir),
            profiles_dir=str(dbt_project.project_dir),
        )
    },
)
