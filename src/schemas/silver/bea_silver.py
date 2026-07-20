"""Schema for the BEA bronze->silver panel built by pipelines.bea_silver_pipeline.

Index: Year. One row per year — nominal/real capital, rental price,
investment and output-value series for the IT / non-IT capital buckets
(Hall-Jorgenson framework). Field `description=` is the single source of
truth for the short-name -> full-name mapping written alongside the parquet
output; see docs/data/silver/bea/README.md for the derivations.
"""

import pandas as pd
import patito as pt


class BeaSilverRow(pt.Model):
    year: int = pt.Field(ge=1900, le=2100, description="Calendar year")

    cap_it_nom: float = pt.Field(
        gt=0, description="Capital IT (nominal), current USD (millions)"
    )
    cap_nonit_nom: float = pt.Field(
        gt=0, description="Capital non-IT (nominal), current USD (millions)"
    )
    cap_it_real: float = pt.Field(
        gt=0, description="Capital IT (real), real USD (millions, ref-year prices)"
    )
    cap_nonit_real: float = pt.Field(
        gt=0,
        description="Capital non-IT (real), real USD (millions, ref-year prices)",
    )
    cap_it_nom_idx: float = pt.Field(
        gt=0, description="Capital IT index (nominal), = 1.0 in ref year"
    )
    cap_nonit_nom_idx: float = pt.Field(
        gt=0, description="Capital non-IT index (nominal), = 1.0 in ref year"
    )
    cap_it_real_idx: float = pt.Field(
        gt=0, description="Capital IT index (real, Tornqvist), = 1.0 in ref year"
    )
    cap_nonit_real_idx: float = pt.Field(
        gt=0, description="Capital non-IT index (real, Tornqvist), = 1.0 in ref year"
    )

    rent_it_nom: float = pt.Field(description="Rental price IT (nominal), current USD")
    rent_nonit_nom: float = pt.Field(
        description="Rental price non-IT (nominal), current USD"
    )
    rent_it_real: float = pt.Field(
        description="Rental price IT (real), real USD (deflated by output price)"
    )
    rent_nonit_real: float = pt.Field(
        description="Rental price non-IT (real), real USD (deflated by output price)"
    )

    share_it: float = pt.Field(description="Share of output paid to IT capital")
    share_nonit: float = pt.Field(description="Share of output paid to non-IT capital")

    delta_it: float = pt.Field(
        gt=0.0, lt=1.0, description="Effective depreciation rate, IT bucket"
    )
    delta_nonit: float = pt.Field(
        gt=0.0, lt=1.0, description="Effective depreciation rate, non-IT bucket"
    )

    inv_it_nom: float = pt.Field(
        gt=0, description="Investment in IT (nominal), current USD (millions)"
    )
    inv_nonit_nom: float = pt.Field(
        gt=0, description="Investment in non-IT (nominal), current USD (millions)"
    )
    inv_it_real: float = pt.Field(
        gt=0,
        description="Investment in IT (real), real USD (millions, ref-year prices)",
    )
    inv_nonit_real: float = pt.Field(
        gt=0,
        description="Investment in non-IT (real), real USD (millions, ref-year prices)",
    )

    r_t: float = pt.Field(description="Internal rate of return")

    y_nom: float = pt.Field(
        gt=0, description="Output value aggregate (nominal), current USD (millions)"
    )
    y_real: float = pt.Field(
        gt=0,
        description="Output value aggregate (real), real USD (millions, ref-year prices)",
    )
    y_nom_idx: float = pt.Field(
        gt=0,
        description="Output value index (nominal / nominal_ref), = 1.0 in ref year",
    )
    y_real_idx: float = pt.Field(
        gt=0, description="Output value index (real / real_ref), = 1.0 in ref year"
    )
    p_output: float = pt.Field(
        gt=0, description="Output value deflator, = 1.0 in ref year"
    )
    pi_output: float = pt.Field(
        description="Output value inflation, pct change of the deflator"
    )


def validate_bea_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the BEA silver panel; raises patito.exceptions.DataFrameValidationError on failure."""
    BeaSilverRow.validate(df.reset_index())
    return df


def column_descriptions() -> dict[str, str]:
    """Short-name -> one-line description, derived from the field `description=`s above."""
    properties = BeaSilverRow.model_json_schema()["properties"]
    return {name: info["description"] for name, info in properties.items()}
