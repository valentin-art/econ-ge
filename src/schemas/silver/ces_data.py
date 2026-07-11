"""Schema for the final CES data panel assembled by pipeline.run_capital_pipeline.

Index: Year. One row per year, columns are the CES production-function inputs
(capital services, rental prices, depreciation, internal return) plus the
output-value aggregate used to write the FOCs in a single numeraire.
"""

import pandas as pd
import patito as pt


class CesDataRow(pt.Model):
    # patito validates columns, not a pandas index — validate_ces_data() below
    # resets the Year index into a column before checking against this model.
    Year: int = pt.Field(ge=1900, le=2100)
    K_IT: float = pt.Field(gt=0)
    K_non_IT: float = pt.Field(gt=0)
    p_IT: float = pt.Field(gt=0)
    p_non_IT: float = pt.Field(gt=0)
    p_IT_real: float = pt.Field(gt=0)
    p_non_IT_real: float = pt.Field(gt=0)
    delta_IT: float = pt.Field(gt=0.0, lt=1.0)
    delta_non_IT: float = pt.Field(gt=0.0, lt=1.0)
    r_t: float
    r_t_real: float
    Y_real: float = pt.Field(gt=0)
    Y_real_idx: float = pt.Field(gt=0)
    Y_nom: float = pt.Field(gt=0)
    P_output: float = pt.Field(gt=0)


def validate_ces_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the final CES panel; raises patito.exceptions.DataFrameValidationError on failure."""
    CesDataRow.validate(df.reset_index())
    return df
