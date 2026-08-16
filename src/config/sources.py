"""Source parameters: table names, sample window, and line-number mappings.

These are fixed methodological choices:
 - CPS data sources
 - BEA tables
 - Sample window and reference year
 - Line which encode "Net operating surplus" in BEA data
 - The smoothing window for BEA data

 Unlike `settings.py`, this is a plain constants module, not env-driven
 pydantic settings.
"""

from typing import Literal, NamedTuple

CPSSource = Literal["basic", "mw"]


# -- BEA Sample window and reference year ----------------------------------

# BEA chain-index reference year
REF_YEAR = 1990

# Sample window: FAA detail is reliable from 1975
YEAR_START = 1975
YEAR_END = 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))

# First year with valid pi_bar and Tornqvist increment (edges lose 2 years)
EFF_START = YEAR_START + 2

# Smoothing window for pi_{j,t}: 3-year centred MA is standard in literature
PI_SMOOTH_WINDOW = 3

# Hard floor on smoothed pi: no asset loses more than 40% in a smoothed year
PI_FLOOR = -0.40


# -- BEA Fixed Assets tables -----------------------------------------------
#
# Current-cost net stock, ALL private fixed assets
FA_TABLE_21 = "FAAt201"
# Chain-quantity index, net stock
FA_TABLE_24 = "FAAt204"
# Current-cost investment
FA_TABLE_25 = "FAAt205"
# Chain-quantity index, investment
FA_TABLE_26 = "FAAt206"


# -- NIPA income tables ----------------------------------------------------
#
# Scope alignment:
#   FAA 2.1 denominator - all private nonresidential:
#      + corp
#      + noncorp
#      + farm
#      + financial
#      + nonprofit
#
#   T11600 (nonfarm nonfin private):
#      + corp
#      + noncorp
#      - farm (<1% of stock)
#      - financial (capital income ≠ user-cost formula)
#     Best available match for the asset-level stock denominator.
#
#   T11400 (corporate only) is retained for scope-ratio diagnostics.

# Domestic Corporate Business GVA
NIPA_TABLE_1_14 = "T11400"
# Domestic Nonfarm Nonfinancial Private Business GVA (primary)
NIPA_TABLE_1_16 = "T11600"

# T11400 Line 8 - "Net operating surplus" (NOS)
NOS_LINE_CORP = 8
# T11600 Line 2 - "Net operating surplus" (NOS)
NOS_LINE_TOTAL = 2
# Alias used downstream
NOS_LINE = NOS_LINE_TOTAL


# -- NIPA value-added-by-sector tables (CES output aggregate) --------------
#
# Nonfarm business sector = Line 3 (a sub-line of Line 2 "Business").
#
# Scope-matches the T11600 NOS and the nonfarm-nonfinancial capital stock:
# the same sector generates the output, the capital income, and the stock.
#
# Published price index (T10304) = nominal/real × 100
# base 2017 = 100, so the implicit deflator (nominal/real) is exact.
# We renormalize to REF_YEAR.

# Table 1.3.5: Gross value added by sector, current $
VA_TABLE_NOMINAL = "T10305"
# Table 1.3.6: Real gross value added, chained $
VA_TABLE_REAL = "T10306"
# Table 1.3.4: Price indexes for GVA by sector
VA_TABLE_PRICE = "T10304"
# verified: Line 3 = "Nonfarm" (business)
VA_LINE_NONFARM = 3

# (dataset, table) pairs pulled by the BEA extract/parse pipeline
# NOTE: VA_TABLE_PRICE (T10304) is not pulled.
BEA_TABLES: list[tuple[str, str]] = [
    ("FixedAssets", FA_TABLE_21),
    ("FixedAssets", FA_TABLE_24),
    ("FixedAssets", FA_TABLE_25),
    ("FixedAssets", FA_TABLE_26),
    ("NIPA", NIPA_TABLE_1_14),
    ("NIPA", NIPA_TABLE_1_16),
    ("NIPA", VA_TABLE_NOMINAL),
    ("NIPA", VA_TABLE_REAL),
]

# -- CPS Mare-Winship (NBER) -----------------------------------------------

# Two-digit "year" suffix used in NBER's cpsmw{YY}.zip file naming, one file
# per March CPS extract.
CPS_MW_YEARS = [1964]

# SPS-files are used as dictionary (fixed-width column layout)
CPS_MW_SPS_RANGES = {
    (1964, 1988): "cpsmw64_88.sps",
    (1989, 1992): "cpsmw89_92.sps",
}


def cps_mw_sps_filename(year: int) -> str:
    """The SPS dictionary filename covering `year`, e.g. 1970 -> cpsmw64_88.sps."""
    for (start, end), filename in CPS_MW_SPS_RANGES.items():
        if start <= year <= end:
            return filename
    raise ValueError(f"No CPS Mare-Winship SPS dictionary covers year {year}")


# -- CPS Basic (NBER) ------------------------------------------------------
#
# (year, month) periods to pull for CPS Basic
CPS_BASIC_PERIODS: list[tuple[int, int]] = [(1991, 2)]

_CPS_BASIC_DICT_YEARMONS = [
    198901,
    199201,
    199401,
    199404,
    199506,
    199509,
    199801,
    200301,
    200405,
    200508,
    200701,
    200901,
    201001,
    201205,
    201301,
    201401,
    201404,
    201501,
    201701,
    202001,
    202301,
    202401,
    202405,
    202501,
    202601,
]


def cps_basic_sps_filename(year: int, month: int) -> str:
    """The SPS dictionary filename covering `year` and `month`, e.g. 1989, 3 -> cpsb198903.sps."""
    yearmon = year * 100 + month
    if yearmon < _CPS_BASIC_DICT_YEARMONS[0]:
        raise ValueError(f"No CPS Basic SPS dictionary covers {year}-{month:02d}")
    for start, end in zip(_CPS_BASIC_DICT_YEARMONS, _CPS_BASIC_DICT_YEARMONS[1:]):
        if start <= yearmon < end:
            return f"cpsb{start}.sps"
    return f"cpsb{_CPS_BASIC_DICT_YEARMONS[-1]}.sps"


# -- IPUMS ------------------------------------------------------------------

CPS_VARS = [
    # Identifiers
    "YEAR",
    "SERIAL",
    "PERNUM",
    "STATEFIP",
    "CPSID",
    "CPSIDP",
    "MISH",
    "ASECFLAG",
    # Weights
    "ASECWT",
    "ASECWTH",
    # Demography
    "AGE",
    "SEX",
    "RACE",
    "HISPAN",
    "RELATE",
    "MARST",
    # Schooling
    "EDUC",
    # Labor supply
    "POPSTAT",
    "EMPSTAT",
    "LABFORCE",
    "FULLPART",
    "AHRSWORKT",
    "UHRSWORKLY",
    "WKSWORK1",
    "WKSWORK2",
    "WHYNWLY",
    "INDLY",
    "CLASSWLY",
    "OCCLY",
    "OCC",
    # Earnings
    "INCWAGE",
    "INCBUS",
    "INCFARM",
    "SRCEARN",
]


class IPUMSExtractRequest(NamedTuple):
    collection: str
    samples: tuple[str, ...]
    variables: tuple[str, ...]
    description: str = ""
    data_quality_flags: bool = True


# Microdata extracts pulled by the IPUMS extract/parse pipeline.
IPUMS_EXTRACTS: list[IPUMSExtractRequest] = [
    IPUMSExtractRequest(
        collection="cps",
        samples=("cps2006_09s",),
        variables=("AGE", "SEX"),
        description="econ-ge CPS extract",
    ),
]
