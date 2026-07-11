"""Schema for the asset dimension table built by parsers.asset_dim.build_asset_dim.

Applies to both ASSET_DIM (all lines) and ASSET_DIM_NONRES (residential
excluded) — same shape, `is_residential` just narrows the row set.
"""

from typing import Literal

import pandas as pd
import patito as pt


class AssetDimRow(pt.Model):
    LineNumber: int = pt.Field(ge=0, unique=True)
    asset_name: str
    # Hulten-Wykoff geometric depreciation rate: strictly between 0 and 1.
    delta_j: float = pt.Field(gt=0.0, lt=1.0)
    bucket: Literal["IT", "non_IT"]
    is_residential: bool
    delta_source: str


def validate_asset_dim(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an asset dimension table; raises patito.exceptions.DataFrameValidationError on failure."""
    AssetDimRow.validate(df)
    return df
