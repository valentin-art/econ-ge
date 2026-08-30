from src.parsers.bea.asset_dim import build_asset_dim


def test_nonres_excludes_residential_lines() -> None:
    asset_dim, asset_dim_nonres = build_asset_dim()

    assert asset_dim_nonres["is_residential"].sum() == 0
    assert len(asset_dim_nonres) < len(asset_dim)
    assert asset_dim["is_residential"].sum() > 0  # sanity: fixture actually has some


def test_no_duplicate_line_numbers() -> None:
    asset_dim, _ = build_asset_dim()
    assert asset_dim["LineNumber"].is_unique


def test_buckets_are_it_or_non_it() -> None:
    asset_dim, _ = build_asset_dim()
    assert set(asset_dim["bucket"].unique()) <= {"IT", "non_IT"}


def test_depreciation_rates_are_valid_fractions() -> None:
    asset_dim, _ = build_asset_dim()
    assert (asset_dim["delta_j"] > 0).all()
    assert (asset_dim["delta_j"] <= 1).all()
