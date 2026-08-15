import pandas as pd

from src.harmonization import cps_tables
from src.harmonization.cps_transformers import (
    age_last_year,
    code_to_value,
    experience_years,
    implied_rate_topcode_flag,
    multi_key_lookup,
    real_value_flag,
    rescale,
    topcode_cap,
    topcode_multiplier,
    weighted_product,
)


def test_topcode_cap_clips_values_above_ceiling() -> None:
    series = pd.Series([85, 90, 95, 100])

    result = topcode_cap(series, 90)

    assert result.tolist() == [85, 90, 90, 90]


def test_topcode_cap_applies_uniformly_regardless_of_year() -> None:
    # Regression guard: the original aa_clean/clean7909km.do's 1992-2009
    # block omitted `age=90 if age>=90`; this function has no year branch,
    # so it must cap consistently for every caller.
    for age_this_block_year in (1965, 1992, 2005):
        series = pd.Series([89, 90, 99])
        assert topcode_cap(series, 90).tolist() == [89, 90, 90]


def test_rescale_undoes_implied_decimal_places() -> None:
    series = pd.Series([149499, 200000])

    result = rescale(series, 1 / 100)

    assert result.tolist() == [1494.99, 2000.0]


def test_weighted_product_multiplies_all_series() -> None:
    wgt = pd.Series([100.0, 200.0])
    wkswork1 = pd.Series([50, 40])
    uhrsworkly = pd.Series([2.0, 3.0])

    result = weighted_product(wgt, wkswork1, uhrsworkly)

    assert result.tolist() == [10000.0, 24000.0]


def test_code_to_value_maps_years_to_deflators() -> None:
    years = pd.Series([1982, 1988, 2008])

    result = code_to_value(years, cps_tables.GDP_PCE_DEFLATOR)

    assert result.tolist() == [
        cps_tables.GDP_PCE_DEFLATOR[1982],
        cps_tables.GDP_PCE_DEFLATOR[1988],
        cps_tables.GDP_PCE_DEFLATOR[2008],
    ]


def test_code_to_value_fills_missing_codes_with_default() -> None:
    series = pd.Series([1, 2, 99])

    result = code_to_value(series, {1: "a", 2: "b"}, default="unknown")

    assert result.tolist() == ["a", "b", "unknown"]


def test_multi_key_lookup_matches_educomp_table_for_a_known_row() -> None:
    keys = pd.DataFrame({"RACE3": [2, 1], "FEMALE": [1, 0], "EDUC": [46, 31]})

    result = multi_key_lookup(keys, cps_tables.EDUCOMP_LOOKUP)

    assert result.tolist() == [18.00, 0.32]


def test_topcode_multiplier_ge_scales_values_at_or_above_threshold() -> None:
    income = pd.Series([50000.0, 99999.0, 150000.0])

    result = topcode_multiplier(income, threshold=99999, comparison="ge")

    assert result.tolist() == [50000.0, 99999.0 * 1.5, 150000.0 * 1.5]


def test_topcode_multiplier_eq_scales_only_exact_match() -> None:
    income = pd.Series([50000.0, 60000.0])

    result = topcode_multiplier(income, threshold=50000, comparison="eq")

    assert result.tolist() == [75000.0, 60000.0]


def test_implied_rate_topcode_flag_matches_tcwkwg_formula() -> None:
    income = pd.Series([50000.0, 200000.0])
    wkswork1 = pd.Series([50.0, 50.0])
    rate_ceiling = 50000 * 1.5 / 40

    result = implied_rate_topcode_flag(income, wkswork1, rate_ceiling)

    assert result.tolist() == [False, True]


def test_real_value_flag_matches_bcwkwg_formula() -> None:
    winc_ws = pd.Series([30.0, 100.0])
    gdp = pd.Series([1.0, 1.0])
    dollar_ceiling = 40 * cps_tables.KM_1982_DEFLATOR_REF

    result = real_value_flag(winc_ws, gdp, dollar_ceiling)

    assert result.tolist() == [True, False]


def test_experience_years_matches_hand_computed_aa_clean_formula() -> None:
    # exp = max(min(age-educomp-7, age-17), 0); age=40, educomp=12 -> 21
    age = pd.Series([40, 16, 20])
    educomp = pd.Series([12, 12, 0])

    result = experience_years(age, educomp)

    assert result.tolist() == [21, 0, 3]


def test_age_last_year_matches_aa_clean_agely_formula() -> None:
    age = pd.Series([17, 71, 72, 90])

    result = age_last_year(age)

    assert result.tolist() == [16, 70, 71, 71]
