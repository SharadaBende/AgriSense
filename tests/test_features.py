"""Leakage tests. The central idea: if features for year <= Y really depend only on
information up to Y, then CORRUPTING all data after Y must leave those features
untouched, bit for bit."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import features as F

N_DIST = 40


@pytest.fixture(scope="module")
def raw():
    crop = pd.read_csv(C.PROCESSED / "crop_yield.csv")
    rain = pd.read_csv(C.PROCESSED / "rainfall.csv")
    dist = pd.read_csv(C.PROCESSED / "districts.csv")
    keep = sorted(crop.dist_code.unique())[::8][:N_DIST]
    return (crop[crop.dist_code.isin(keep)].copy(), rain[rain.dist_code.isin(keep)].copy(),
            dist, F.district_points())


@pytest.fixture(scope="module")
def table(raw):
    return F.build_model_table(*raw)


def _same(a, b, cols):
    k = ["dist_code", "crop", "year"]
    m = a[k + cols].merge(b[k + cols], on=k, suffixes=("_a", "_b"))
    for c in cols:
        x, y = m[c + "_a"].to_numpy(dtype=float, na_value=np.nan), m[c + "_b"].to_numpy(dtype=float, na_value=np.nan)
        assert np.allclose(x, y, equal_nan=True), f"feature {c} changed when only FUTURE data changed"
    return len(m)


def test_future_yield_corruption_does_not_change_past_features(raw, table):
    crop, rain, dist, pts = raw
    Y0 = 1999
    bad = crop.copy()
    fut = bad.year > Y0
    bad.loc[fut, ["yield_kg_ha", "area_1000ha", "production_1000t"]] *= 10
    t2 = F.build_model_table(bad, rain, dist, pts)
    past_a, past_b = table[table.year <= Y0], t2[t2.year <= Y0]
    feats = [c for c in F.feature_columns(table, include_year=True)
             if c not in F.CATEGORICAL and c not in ("year", "crop")]
    n = _same(past_a, past_b, feats)
    assert n > 5000


def test_future_rainfall_corruption_does_not_change_past_features(raw, table):
    crop, rain, dist, pts = raw
    Y0 = 1995
    bad = rain.copy()
    cols = [c for c in bad.columns if c.startswith("rain_") or c.endswith("_mm")]
    bad.loc[bad.year > Y0, cols] *= 3
    t2 = F.build_model_table(crop, bad, dist, pts)
    rain_feats = [c for c in table.columns if c.endswith("_ratio") or c.startswith("spi_") or c.startswith("clim_")
                  or c.endswith("_prev") or c in ("n_deficit_months", "jul_aug_share", "monsoon_ratio_2yr")]
    _same(table[table.year <= Y0], t2[t2.year <= Y0], rain_feats)


def test_lag_features_are_the_previous_calendar_year(raw, table):
    crop = raw[0]
    ok = crop[crop.is_outlier == 0].set_index(["dist_code", "crop", "year"]).yield_kg_ha
    sample = table[table.y_lag1.notna()].sample(500, random_state=1)
    for r in sample.itertuples():
        assert ok.loc[(r.dist_code, r.crop, r.year - 1)] == pytest.approx(r.y_lag1)


def test_roll5_uses_only_previous_years(raw, table):
    crop = raw[0]
    ok = crop[crop.is_outlier == 0].set_index(["dist_code", "crop", "year"]).yield_kg_ha
    sample = table[table.y_roll5.notna()].sample(300, random_state=2)
    for r in sample.itertuples():
        vals = [ok.get((r.dist_code, r.crop, y)) for y in range(r.year - 5, r.year)]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        assert len(vals) >= 3 and np.mean(vals) == pytest.approx(r.y_roll5)


def test_no_feature_is_the_target_or_derived_from_it(table):
    feats = F.feature_columns(table)
    for forbidden in ["yield_kg_ha", "log_yield", "target_logratio", "split"]:
        assert forbidden not in feats
    num = table[feats].select_dtypes("number")
    corr = num.corrwith(table.log_yield).abs()
    assert corr.max() < 0.97, f"suspiciously high correlation: {corr.idxmax()} = {corr.max():.3f}"


def test_splits_are_ordered_and_disjoint(table):
    tr, va, te = (table[table.split == s].year for s in ["train", "valid", "test"])
    assert tr.max() <= C.TRAIN_END < va.min() and va.max() < te.min()
    assert (va.min(), va.max()) == C.VALID_YEARS and (te.min(), te.max()) == C.TEST_YEARS


def test_cv_folds_stay_inside_training_and_are_expanding():
    prev_end = 0
    for train_end, (v0, v1) in C.CV_FOLDS:
        assert train_end < v0 <= v1 <= C.TRAIN_END and train_end > prev_end
        prev_end = train_end


def test_spi_is_standard_normal_on_reference_period(table):
    lo, hi = C.REF_PERIOD
    d = table[(table.year >= lo) & (table.year <= hi)].drop_duplicates(["dist_code", "year"])
    assert abs(d.spi_monsoon.mean()) < 0.15 and 0.8 < d.spi_monsoon.std() < 1.2


def test_trend_slope_is_clipped_and_history_is_sufficient(table):
    s = table.y_trend_slope.dropna()
    assert s.between(-0.08, 0.08).all()
    assert (table.loc[table.y_trend_slope.notna(), "n_hist10"] >= 6).all()


def test_no_duplicate_rows(table):
    assert not table.duplicated(["dist_code", "crop", "year"]).any()
