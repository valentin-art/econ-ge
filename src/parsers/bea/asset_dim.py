"""Asset dimension table: BEA line numbers, Hulten-Wykoff depreciation rates, IT buckets."""

import pandas as pd

from src.schemas.silver.asset_dim import validate_asset_dim

# Each tuple:
# (LineNumber, asset_name, delta_j, bucket, is_residential)
#
# delta_j:
#   source:
#       Fraumeni (1997), Survey of Current Business, Table 1.
#   calculation:
#   delta_j = d_j / T_j  where d_j = declining-balance rate, T_j = service life.
#   Methodology:
#       Reproduced in Krusell et al. (2000) appendix and BEA Fixed Assets Methodology.
#
# IT bucket: computers + communication + software (IPP) + R&D.
#   non_IT: all other equipment + nonresidential structures + other IPP.
#
# is_residential = True → excluded from nonresidential pipeline.
ASSET_DIM_RAW: list[tuple] = [
    # ── Information processing equipment ─────────────────────────────────────
    (5, "Computers and peripheral equipment", 0.315, "IT", False),
    (6, "Communication equipment", 0.110, "IT", False),
    (7, "Medical equipment and instruments", 0.165, "non_IT", False),
    (8, "Nonmedical instruments", 0.135, "non_IT", False),
    (9, "Photocopy and related equipment", 0.180, "non_IT", False),
    (10, "Office and accounting equipment", 0.315, "non_IT", False),
    # ── Industrial equipment ──────────────────────────────────────────────────
    (12, "Fabricated metal products", 0.092, "non_IT", False),
    (13, "Engines and turbines", 0.079, "non_IT", False),
    (14, "Metalworking machinery", 0.123, "non_IT", False),
    (15, "Special industry machinery, n.e.c.", 0.103, "non_IT", False),
    (16, "General industrial, incl. materials handling", 0.107, "non_IT", False),
    (17, "Electrical transmission and distribution", 0.050, "non_IT", False),
    # ── Transportation equipment ──────────────────────────────────────────────
    (20, "Light trucks (including utility vehicles)", 0.154, "non_IT", False),
    (21, "Other trucks, buses, and truck trailers", 0.154, "non_IT", False),
    (22, "Autos", 0.192, "non_IT", False),
    (23, "Aircraft", 0.083, "non_IT", False),
    (24, "Ships and boats", 0.075, "non_IT", False),
    (25, "Railroad equipment", 0.066, "non_IT", False),
    # ── Other nonresidential equipment ────────────────────────────────────────
    (27, "Furniture and fixtures", 0.115, "non_IT", False),
    (28, "Agricultural machinery", 0.118, "non_IT", False),
    (29, "Construction machinery", 0.155, "non_IT", False),
    (30, "Mining and oilfield machinery", 0.150, "non_IT", False),
    (31, "Service industry machinery", 0.163, "non_IT", False),
    (32, "Electrical equipment, n.e.c.", 0.110, "non_IT", False),
    (33, "Other nonresidential equipment", 0.150, "non_IT", False),
    # ── Residential equipment — EXCLUDED ──────────────────────────────────────
    (34, "Residential equipment", 0.150, "non_IT", True),
    # ── Nonresidential structures ─────────────────────────────────────────────
    (38, "Office", 0.028, "non_IT", False),
    (41, "Hospitals", 0.028, "non_IT", False),
    (42, "Special care", 0.028, "non_IT", False),
    (43, "Medical buildings", 0.028, "non_IT", False),
    (44, "Multimerchandise shopping", 0.028, "non_IT", False),
    (45, "Food and beverage establishments", 0.034, "non_IT", False),
    (46, "Warehouses", 0.025, "non_IT", False),
    (47, "Other commercial", 0.028, "non_IT", False),
    (48, "Manufacturing", 0.031, "non_IT", False),
    (51, "Electric", 0.020, "non_IT", False),
    (52, "Other power", 0.020, "non_IT", False),
    (53, "Communication structures", 0.020, "IT", False),
    (55, "Petroleum and natural gas", 0.075, "non_IT", False),
    (56, "Mining", 0.075, "non_IT", False),
    (58, "Religious", 0.026, "non_IT", False),
    (59, "Educational and vocational", 0.026, "non_IT", False),
    (60, "Lodging", 0.026, "non_IT", False),
    (61, "Amusement and recreation", 0.026, "non_IT", False),
    (63, "Air transportation structures", 0.020, "non_IT", False),
    (64, "Land transportation structures", 0.020, "non_IT", False),
    (65, "Farm structures", 0.024, "non_IT", False),
    (66, "Other structures", 0.026, "non_IT", False),
    # ── Residential structures — EXCLUDED ────────────────────────────────────
    (67, "Residential structures", 0.015, "non_IT", True),
    (68, "Housing units", 0.015, "non_IT", True),
    (69, "Permanent site", 0.015, "non_IT", True),
    (70, "1 to 4 unit", 0.015, "non_IT", True),
    (71, "5-or more-unit", 0.015, "non_IT", True),
    (72, "Manufactured homes", 0.015, "non_IT", True),
    (73, "Brokers commissions", 0.015, "non_IT", True),
    (74, "Improvements", 0.015, "non_IT", True),
    (75, "Other residential", 0.015, "non_IT", True),
    # ── Software (leaves of line 78) ──────────────────────────────────────────
    (79, "Prepackaged software", 0.550, "IT", False),
    (80, "Custom software", 0.330, "IT", False),
    (81, "Own-account software", 0.330, "IT", False),
    # ── R&D (leaf lines only) ─────────────────────────────────────────────────
    (85, "Pharmaceutical and medicine manufacturing", 0.200, "IT", False),
    (86, "Chemical manufacturing excl pharma", 0.200, "IT", False),
    (87, "Semiconductor and electronic components", 0.200, "IT", False),
    (88, "Other computer and electronic products", 0.200, "IT", False),
    (89, "Motor vehicles and parts manufacturing", 0.200, "IT", False),
    (90, "Aerospace products and parts", 0.200, "IT", False),
    (91, "Other manufacturing R&D", 0.200, "IT", False),
    (93, "Scientific R&D services", 0.200, "IT", False),
    (94, "All other nonmanufacturing R&D", 0.200, "IT", False),
    (96, "Universities and colleges R&D", 0.200, "IT", False),
    (97, "Other nonprofit institutions R&D", 0.200, "IT", False),
    # ── Entertainment originals (leaves of line 98) ───────────────────────────
    (99, "Theatrical movies", 0.120, "non_IT", False),
    (100, "Long-lived television programs", 0.120, "non_IT", False),
    (101, "Books", 0.120, "non_IT", False),
    (102, "Music", 0.120, "non_IT", False),
    (103, "Other originals", 0.120, "non_IT", False),
]

_COLUMNS = ["LineNumber", "asset_name", "delta_j", "bucket", "is_residential"]


def build_asset_dim() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ASSET_DIM and ASSET_DIM_NONRES from ASSET_DIM_RAW.

    Returns
    -------
    ASSET_DIM       : full table including residential lines
    ASSET_DIM_NONRES: residential lines excluded — used for all pipeline steps
    """
    dim = pd.DataFrame(ASSET_DIM_RAW, columns=_COLUMNS)
    dim["delta_source"] = "BEA/Fraumeni(1997) Hulten-Wykoff geometric rates"
    nonres = dim[~dim["is_residential"]].copy()
    validate_asset_dim(dim)
    validate_asset_dim(nonres)
    return dim, nonres
