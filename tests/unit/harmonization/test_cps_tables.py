from itertools import pairwise

from src.harmonization import cps_tables


def test_cpi_deflator_covers_1963_to_2008_excluding_1962() -> None:
    assert 1962 not in cps_tables.CPI_DEFLATOR
    assert cps_tables.CPI_DEFLATOR[1961] == 215.3 / 32.5
    assert cps_tables.CPI_DEFLATOR[2008] == 1.0


def test_deflator_series_fall_monotonically_toward_the_2008_base() -> None:
    # A deflator to a fixed 2008 base must shrink as the income year gets
    # closer to 2008 - prices only rose over 1961-2008. Any rise means a
    # mistyped denominator, which is exactly how `2000: 215.3 / 215.3` (the
    # base year's own denominator, pasted a row early) went unnoticed and
    # understated every survey-year-2001 real wage by ~20%.
    for table in (cps_tables.CPI_DEFLATOR, cps_tables.GDP_PCE_DEFLATOR):
        years = sorted(table)
        rising = [(a, b) for a, b in pairwise(years) if table[a] < table[b]]
        assert rising == []


def test_gdp_pce_deflator_matches_km_1982_reference() -> None:
    assert cps_tables.GDP_PCE_DEFLATOR[1982] == cps_tables.KM_1982_DEFLATOR_REF
    assert cps_tables.KM_1982_DEFLATOR_REF == 109.031 / 53.615


def test_income_topcode_pre1988_bands_are_contiguous_1962_to_1987() -> None:
    bands = sorted(cps_tables.INCOME_TOPCODE_PRE1988)
    assert bands[0][0] == 1962
    assert bands[-1][1] == 1987
    assert cps_tables.INCOME_TOPCODE_PRE1988[(1968, 1981)] == 50000


def test_maxer_maxwg_tables_cover_1988_to_2009() -> None:
    assert set(cps_tables.MAXER_TABLE) == set(range(1988, 2010))
    assert set(cps_tables.MAXWG_TABLE) == set(range(1988, 2010))
    assert cps_tables.MAXER_TABLE[2003] == 200000
    assert cps_tables.MAXWG_TABLE[1989] == 95000


def test_educomp_lookup_has_one_entry_per_race_sex_grdatn_combo() -> None:
    # 3 race groups x 2 sexes x 17 distinct grdatn codes = 102
    assert len(cps_tables.EDUCOMP_LOOKUP) == 102
    assert cps_tables.EDUCOMP_LOOKUP[(1, 0, 31)] == cps_tables.EDUCOMP_LOOKUP[(1, 0, 0)]
    assert cps_tables.EDUCOMP_LOOKUP[(2, 1, 46)] == 18.00


def test_classwly_wage_and_selfemp_codes_are_disjoint() -> None:
    wage = set(cps_tables.CLASSWLY_WAGE_CODES)
    selfemp = set(cps_tables.CLASSWLY_SELFEMP_CODES)
    assert wage.isdisjoint(selfemp)
    assert 29 not in wage | selfemp  # unpaid family worker excluded from both
