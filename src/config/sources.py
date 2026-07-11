"""BEA source parameters: table names, sample window, and line-number mappings.

These are fixed methodological choices (which BEA table vintage, which line
encodes "Net operating surplus", the smoothing window) rather than deployment
config — so, unlike settings.py, this is a plain constants module, not
env-driven pydantic settings.
"""

# BEA chain-index reference year (verify from Table 2.4 output)
REF_YEAR = 1990

# Sample window — FAA detail is reliable from 1975
YEAR_START = 1975
YEAR_END = 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))

# First year with valid pi_bar and Tornqvist increment (edges lose 2 years)
EFF_START = YEAR_START + 2

# Smoothing window for pi_{j,t} — 3-year centred MA is standard in literature
PI_SMOOTH_WINDOW = 3

# Hard floor on smoothed pi: no asset loses more than 40% in a smoothed year
PI_FLOOR = -0.40

# ── BEA Fixed Assets tables ───────────────────────────────────────────────
# Current-cost net stock, ALL private fixed assets
FA_TABLE_21 = "FAAt201"
# Chain-quantity index, net stock
FA_TABLE_24 = "FAAt204"
# Current-cost investment
FA_TABLE_25 = "FAAt205"
# Chain-quantity index, investment
FA_TABLE_26 = "FAAt206"

# ── NIPA income tables ────────────────────────────────────────────────────
# Scope alignment:
#   FAA 2.1 denominator = all private nonresidential (corporate + noncorporate
#   + farm + financial + nonprofit).
#   T11600 (nonfarm nonfinancial private) covers corporate + noncorporate,
#   excluding farm (<1% of stock) and financial (capital income ≠ user-cost
#   formula) — best available match for the asset-level stock denominator.
#   T11400 (corporate only) is retained for scope-ratio diagnostics.

# Domestic Corporate Business GVA
NIPA_TABLE_1_14 = "T11400"
# Domestic Nonfarm Nonfinancial Private Business GVA (primary)
NIPA_TABLE_1_16 = "T11600"

# T11400 Line 8 — verified: "Net operating surplus"
NOS_LINE_CORP = 8
# T11600 Line 2 — verified: "Net operating surplus"
NOS_LINE_TOTAL = 2
# alias used downstream
NOS_LINE = NOS_LINE_TOTAL

# ── NIPA value-added-by-sector tables (CES output aggregate) ──────────────
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

# ── CPS Mare-Winship (NBER) ────────────────────────────────────────────────
# Two-digit "year" suffix used in NBER's cpsmw{YY}.zip file naming, one file
# per March CPS extract. cpsmw64.zip is the only year-file currently in use
# (see src/config/cps_source_instructions.md) — extend CPS_MW_YEARS to pull
# more.
CPS_MW_YEARS = [1964]

# SPS dictionary (fixed-width column layout) covering each year range —
# NBER ships one dictionary per multi-year span, not one per year.
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
