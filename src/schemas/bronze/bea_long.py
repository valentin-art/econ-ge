"""Schema for the tidy long DataFrame returned by extractors.bea_api.

Columns: LineNumber, LineDescription, Year, DataValue — the shape every BEA
FixedAssets/NIPA pull is normalized to before anything downstream touches it.
"""

from typing import Optional

import pandas as pd
import patito as pt


class BeaLongRow(pt.Model):
    LineNumber: int = pt.Field(ge=0)
    LineDescription: str
    Year: int = pt.Field(ge=1900, le=2100)
    # BEA occasionally reports a blank cell (suppressed/not applicable) —
    # extractors.bea_api coerces those to NaN, so DataValue must stay nullable.
    DataValue: Optional[float] = pt.Field(default=None)


def validate_bea_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a tidy long BEA table; raises patito.exceptions.DataFrameValidationError on failure."""
    BeaLongRow.validate(df)
    return df
