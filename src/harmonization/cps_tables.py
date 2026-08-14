"""Year-keyed and code-keyed constants transcribed from `aa_clean/`.

Katz-Murphy/Wasserman-lineage March CPS cleaning tables (`aa_clean/cpi.do`,
`aa_clean/gdpmar.do`, `aa_clean/cps_exper_post92.do`, and the topcode
scalars inlined in `aa_clean/clean7909km.do`/`clean7678km.do`/
`clean6275km.do`), kept as plain data so `cps_transformers.py`'s functions
stay generic instead of branching on year internally. Separate from
`src/config/sources.py` because these are cleaning-methodology constants,
not extract/source configuration.
"""

# -- CPI deflator, 2008$ (aa_clean/cpi.do) --------------------------------
# cpi[year] = 215.3 / CPI[year]. No 1962 entry in the source (gap in the
# original .do file - 1962 income isn't deflated via this table there).
CPI_DEFLATOR: dict[int, float] = {
    1961: 215.3 / 32.5,
    1963: 215.3 / 33.3,
    1964: 215.3 / 33.7,
    1965: 215.3 / 34.2,
    1966: 215.3 / 35.2,
    1967: 215.3 / 36.3,
    1968: 215.3 / 37.7,
    1969: 215.3 / 39.4,
    1970: 215.3 / 41.3,
    1971: 215.3 / 43.1,
    1972: 215.3 / 44.4,
    1973: 215.3 / 47.2,
    1974: 215.3 / 51.9,
    1975: 215.3 / 56.2,
    1976: 215.3 / 59.4,
    1977: 215.3 / 63.2,
    1978: 215.3 / 67.5,
    1979: 215.3 / 74.0,
    1980: 215.3 / 82.3,
    1981: 215.3 / 90.1,
    1982: 215.3 / 95.6,
    1983: 215.3 / 99.6,
    1984: 215.3 / 103.9,
    1985: 215.3 / 107.6,
    1986: 215.3 / 109.6,
    1987: 215.3 / 113.6,
    1988: 215.3 / 118.3,
    1989: 215.3 / 124.0,
    1990: 215.3 / 130.7,
    1991: 215.3 / 136.2,
    1992: 215.3 / 140.3,
    1993: 215.3 / 144.5,
    1994: 215.3 / 148.2,
    1995: 215.3 / 152.4,
    1996: 215.3 / 156.9,
    1997: 215.3 / 160.5,
    1998: 215.3 / 163.0,
    1999: 215.3 / 166.6,
    2000: 215.3 / 172.2,
    2001: 215.3 / 177.1,
    2002: 215.3 / 179.9,
    2003: 215.3 / 184.0,
    2004: 215.3 / 188.9,
    2005: 215.3 / 195.3,
    2006: 215.3 / 201.6,
    2007: 215.3 / 207.3,
    2008: 215.3 / 215.3,
}

# -- GDP personal-consumption-expenditure deflator, 2008$ (aa_clean/gdpmar.do) --
GDP_PCE_DEFLATOR: dict[int, float] = {
    1961: 109.031 / 18.801,
    1963: 109.031 / 19.245,
    1964: 109.031 / 19.527,
    1965: 109.031 / 19.81,
    1966: 109.031 / 20.313,
    1967: 109.031 / 20.824,
    1968: 109.031 / 21.636,
    1969: 109.031 / 22.616,
    1970: 109.031 / 23.674,
    1971: 109.031 / 24.68,
    1972: 109.031 / 25.525,
    1973: 109.031 / 26.901,
    1974: 109.031 / 29.703,
    1975: 109.031 / 32.184,
    1976: 109.031 / 33.95,
    1977: 109.031 / 36.155,
    1978: 109.031 / 38.687,
    1979: 109.031 / 42.118,
    1980: 109.031 / 46.641,
    1981: 109.031 / 50.81,
    1982: 109.031 / 53.615,
    1983: 109.031 / 55.923,
    1984: 109.031 / 58.038,
    1985: 109.031 / 59.938,
    1986: 109.031 / 61.399,
    1987: 109.031 / 63.589,
    1988: 109.031 / 66.121,
    1989: 109.031 / 68.994,
    1990: 109.031 / 72.147,
    1991: 109.031 / 74.755,
    1992: 109.031 / 76.954,
    1993: 109.031 / 78.643,
    1994: 109.031 / 80.265,
    1995: 109.031 / 82.041,
    1996: 109.031 / 83.826,
    1997: 109.031 / 85.395,
    1998: 109.031 / 86.207,
    1999: 109.031 / 87.596,
    2000: 109.031 / 89.777,
    2001: 109.031 / 91.488,
    2002: 109.031 / 92.736,
    2003: 109.031 / 94.622,
    2004: 109.031 / 97.098,
    2005: 109.031 / 100,
    2006: 109.031 / 102.746,
    2007: 109.031 / 105.502,
    2008: 109.031 / 109.031,
}

# Fixed 1982-reference deflator ratio used by every aa_clean low-wage flag
# (bcwkwg/bchrwg/bcwkwgkm/bchrwgkm), i.e. GDP_PCE_DEFLATOR[1982]. All but one
# of the 12 occurrences of this formula across aa_clean/*.do use 109.031/53.615;
# clean7678km.do's `bcwkwg` alone used 109.031/3.615 (copy-paste typo) - not
# replicated here, this constant matches the other 11 occurrences.
KM_1982_DEFLATOR_REF: float = 109.031 / 53.615
assert KM_1982_DEFLATOR_REF == GDP_PCE_DEFLATOR[1982]

# -- Pre-1988 income topcode thresholds (year-banded, exact-equality match) --
# From clean6275km.do / clean7678km.do / clean7909km.do's 1979-87 block:
# _incse/_incfrm/_incwag *= 1.5 if value == threshold for the year's band.
# Keyed by inclusive (start_year, end_year).
INCOME_TOPCODE_PRE1988: dict[tuple[int, int], float] = {
    (1962, 1964): 90000,
    (1965, 1967): 99900,
    (1968, 1981): 50000,
    (1982, 1984): 75000,
    (1985, 1987): 99999,
}

# -- 1988-2009 income topcode thresholds (per-year, >= match) ------------
# From clean7909km.do lines 335-367 (1988-91 block) and 526-570 (1992-2009
# block) - `incer1`/`incwg1` are IPUMS's split INCBUS+INCFARM / INCWAGE.
MAXER_TABLE: dict[int, float] = {
    1988: 99999,
    1989: 99999,
    1990: 99999,
    1991: 99999,
    1992: 99999,
    1993: 99999,
    1994: 99999,
    1995: 99999,
    1996: 150000,
    1997: 150000,
    1998: 150000,
    1999: 150000,
    2000: 150000,
    2001: 150000,
    2002: 150000,
    2003: 200000,
    2004: 200000,
    2005: 200000,
    2006: 200000,
    2007: 200000,
    2008: 200000,
    2009: 200000,
}

MAXWG_TABLE: dict[int, float] = {
    1988: 99999,
    1989: 95000,
    1990: 99999,
    1991: 90000,
    1992: 99999,
    1993: 99999,
    1994: 99999,
    1995: 99999,
    1996: 25000,
    1997: 25000,
    1998: 25000,
    1999: 25000,
    2000: 25000,
    2001: 25000,
    2002: 25000,
    2003: 35000,
    2004: 35000,
    2005: 35000,
    2006: 35000,
    2007: 35000,
    2008: 35000,
    2009: 35000,
}

# -- 1992+ years-of-schooling imputation (aa_clean/cps_exper_post92.do) --
# Keyed (race, female, grdatn) -> imputed years of completed schooling.
# `race` follows aa_clean's own 1/2/3 = white/black/other recode (== IPUMS
# RACE collapsed to white/black/other, matching the `white`/`black`/`other`
# dummies built earlier in every aa_clean block); `grdatn` is the IPUMS EDUC
# code. grdatn 31 and 00 map to the same imputed value in the source (both
# "less than 1st grade" / no schooling variants) and are represented as two
# separate keys here pointing at the same value.
EDUCOMP_LOOKUP: dict[tuple[int, int, int], float] = {}

_EDUCOMP_ROWS: list[tuple[int, int, tuple[int, ...], float]] = [
    # (race, female, grdatn_codes, educomp)
    (1, 0, (31, 0), 0.32),
    (1, 0, (32,), 3.19),
    (1, 0, (33, 34), 7.24),
    (1, 0, (35,), 8.97),
    (1, 0, (36,), 9.92),
    (1, 0, (37,), 10.86),
    (1, 0, (38,), 11.58),
    (1, 0, (39,), 11.99),
    (1, 0, (40,), 13.48),
    (1, 0, (41, 42), 14.23),
    (1, 0, (43,), 16.17),
    (1, 0, (44,), 17.68),
    (1, 0, (45,), 17.71),
    (1, 0, (46,), 17.83),
    (1, 1, (31, 0), 0.62),
    (1, 1, (32,), 3.15),
    (1, 1, (33, 34), 7.23),
    (1, 1, (35,), 8.99),
    (1, 1, (36,), 9.95),
    (1, 1, (37,), 10.87),
    (1, 1, (38,), 11.73),
    (1, 1, (39,), 12.00),
    (1, 1, (40,), 13.35),
    (1, 1, (41, 42), 14.22),
    (1, 1, (43,), 16.15),
    (1, 1, (44,), 17.64),
    (1, 1, (45,), 17.00),
    (1, 1, (46,), 17.76),
    (2, 0, (31, 0), 0.92),
    (2, 0, (32,), 3.28),
    (2, 0, (33, 34), 7.04),
    (2, 0, (35,), 9.02),
    (2, 0, (36,), 9.91),
    (2, 0, (37,), 10.90),
    (2, 0, (38,), 11.41),
    (2, 0, (39,), 11.98),
    (2, 0, (40,), 13.57),
    (2, 0, (41, 42), 14.33),
    (2, 0, (43,), 16.13),
    (2, 0, (44,), 17.51),
    (2, 0, (45,), 17.83),
    (2, 0, (46,), 18.00),
    (2, 1, (31, 0), 0.00),
    (2, 1, (32,), 2.90),
    (2, 1, (33, 34), 7.03),
    (2, 1, (35,), 9.05),
    (2, 1, (36,), 9.99),
    (2, 1, (37,), 10.85),
    (2, 1, (38,), 11.64),
    (2, 1, (39,), 12.00),
    (2, 1, (40,), 13.43),
    (2, 1, (41, 42), 14.33),
    (2, 1, (43,), 16.04),
    (2, 1, (44,), 17.69),
    (2, 1, (45,), 17.40),
    (2, 1, (46,), 18.00),
    (3, 0, (31, 0), 0.62),
    (3, 0, (32,), 3.24),
    (3, 0, (33, 34), 7.14),
    (3, 0, (35,), 9.00),
    (3, 0, (36,), 9.92),
    (3, 0, (37,), 10.88),
    (3, 0, (38,), 11.50),
    (3, 0, (39,), 11.99),
    (3, 0, (40,), 13.53),
    (3, 0, (41, 42), 14.28),
    (3, 0, (43,), 16.15),
    (3, 0, (44,), 17.60),
    (3, 0, (45,), 17.77),
    (3, 0, (46,), 17.92),
    (3, 1, (31, 0), 0.31),
    (3, 1, (32,), 3.03),
    (3, 1, (33, 34), 7.13),
    (3, 1, (35,), 9.02),
    (3, 1, (36,), 9.97),
    (3, 1, (37,), 10.86),
    (3, 1, (38,), 11.69),
    (3, 1, (39,), 12.00),
    (3, 1, (40,), 13.47),
    (3, 1, (41, 42), 14.28),
    (3, 1, (43,), 16.10),
    (3, 1, (44,), 17.67),
    (3, 1, (45,), 17.20),
    (3, 1, (46,), 17.88),
]
for _race, _female, _codes, _value in _EDUCOMP_ROWS:
    for _code in _codes:
        EDUCOMP_LOOKUP[(_race, _female, _code)] = _value
del _race, _female, _codes, _value, _code, _EDUCOMP_ROWS
# 3 race groups x 2 sexes x 17 distinct grdatn codes (14 rows per race/sex,
# 3 of which - (31,0), (33,34), (41,42) - each cover 2 codes, so
# 11*1 + 3*2 = 17 codes per race/sex).
assert len(EDUCOMP_LOOKUP) == 3 * 2 * 17

# -- CLASSWLY (IPUMS "class of worker, last year") code groups -----------
# Read directly off data/reference/ipums/cps/*.json's CLASSWLY Values
# (uniform across years). aa_clean's own clslyr[1-4]=wage/clslyr[5-6]=selfemp
# NBER recode isn't reusable as-is (different, pre-IPUMS coding), so these
# groups are built from CLASSWLY's actual documented codes instead: wage
# includes both the parent code (20) and its private/government children
# (22, 24-28); self-employed includes the parent (10) and its
# incorporated/unincorporated children (13, 14). Code 29 (unpaid family
# worker) is deliberately excluded from both groups, matching aa_clean's
# implicit exclusion of unpaid family workers via its clslyr scheme.
CLASSWLY_WAGE_CODES: tuple[int, ...] = (20, 22, 24, 25, 27, 28)
CLASSWLY_SELFEMP_CODES: tuple[int, ...] = (10, 13, 14)

# WKSWORK2 (IPUMS "weeks worked last year, intervalled") bridge: the
# pre-1976 case (WKSWORK1 missing) is handled by
# `src.cleaning.custom_functions.bridge_weeks_pre_1976`, which fits
# sex/race/bracket group means from WKSWORK1 at runtime (aa_clean's own
# method, clean7678km.do:132-140) rather than a fixed midpoint table - no
# constant needed here.
