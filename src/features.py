"""Phase 4: leakage-safe feature engineering.

RULE: a row for (district, crop, year T) may only use
  * rainfall of year T and earlier (known at the end of the season),
  * the district-crop's own yields/areas of years < T,
  * a reference climatology fitted on config.REF_PERIOD only,
  * static district facts (location, state).
It may NOT use the yield of year T, any later year, or any other district's
yield from year T.

Run:  python -m src.features
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C

KEY = ["dist_code", "crop"]
YEARS = range(1966, 2018)


# ---------------------------------------------------------------------------
# Rainfall features
# ---------------------------------------------------------------------------
def _fit_spi(ref_values: np.ndarray):
    """Gamma fit on the reference period (WMO-style SPI). Returns (q0, shape, scale) or None."""
    x = ref_values[~np.isnan(ref_values)]
    if len(x) < 15:
        return None
    pos = x[x > 0]
    if len(pos) < 12:
        return None
    a, _, scale = stats.gamma.fit(pos, floc=0)
    return (1 - len(pos) / len(x), a, scale)


def _spi(values: pd.Series, params):
    if params is None:
        return pd.Series(np.nan, index=values.index)
    q0, a, scale = params
    p = q0 + (1 - q0) * stats.gamma.cdf(values.clip(lower=0), a, scale=scale)
    return pd.Series(stats.norm.ppf(np.clip(p, 1e-4, 1 - 1e-4)), index=values.index).where(values.notna())


def build_rain_features(rain: pd.DataFrame) -> pd.DataFrame:
    r = rain.copy().sort_values(["dist_code", "year"])
    r["rain_junjul"] = r[["rain_jun", "rain_jul"]].sum(axis=1, min_count=2)
    r["rain_augsep"] = r[["rain_aug", "rain_sep"]].sum(axis=1, min_count=2)
    lo, hi = C.REF_PERIOD
    ref = r[(r.year >= lo) & (r.year <= hi)]

    windows = {"annual": "annual_mm", "monsoon": "monsoon_mm", "early": "rain_junjul", "late": "rain_augsep"}
    out = []
    for d, g in r.groupby("dist_code"):
        gref = ref[ref.dist_code == d]
        f = g[["dist_code", "year"]].copy()
        # seasonal anomaly ratios vs reference mean
        for name, col in [("annual", "annual_mm"), ("monsoon", "monsoon_mm"), ("pre", "pre_monsoon_mm"),
                          ("winter", "winter_mm"), ("post", "post_monsoon_mm")]:
            m = gref[col].mean() if gref[col].notna().sum() >= 15 else np.nan
            f[f"{name}_ratio"] = g[col] / m if m and m > 0 else np.nan
            if name == "annual":
                f["clim_annual_mm"] = m
            if name == "monsoon":
                f["clim_monsoon_mm"] = m
        # SPI per window
        for name, col in windows.items():
            f[f"spi_{name}"] = _spi(g[col], _fit_spi(gref[col].to_numpy()))
        # month-level monsoon structure
        mon_ratio = []
        for mth in ["jun", "jul", "aug", "sep"]:
            col = f"rain_{mth}"
            m = gref[col].mean()
            ratio = g[col] / m if (m is not None and m >= 5) else pd.Series(np.nan, index=g.index)
            f[f"{mth}_ratio"] = ratio
            mon_ratio.append(ratio)
        mr = pd.concat(mon_ratio, axis=1)
        f["n_deficit_months"] = (mr < 0.5).sum(axis=1).where(mr.notna().all(axis=1))
        f["jul_aug_share"] = (g.rain_jul + g.rain_aug) / g.monsoon_mm.where(g.monsoon_mm > 0)
        out.append(f)
    f = pd.concat(out, ignore_index=True)

    # previous-year weather (carry-over soil moisture / groundwater): merge on year-1,
    # NOT a row shift, because some district-years are missing.
    prev = f[["dist_code", "year", "monsoon_ratio", "spi_monsoon", "annual_ratio"]].copy()
    prev["year"] += 1
    prev.columns = ["dist_code", "year", "monsoon_ratio_prev", "spi_monsoon_prev", "annual_ratio_prev"]
    f = f.merge(prev, on=["dist_code", "year"], how="left")
    f["monsoon_ratio_2yr"] = f[["monsoon_ratio", "monsoon_ratio_prev"]].mean(axis=1, skipna=False)
    return f


# ---------------------------------------------------------------------------
# Yield-history features
# ---------------------------------------------------------------------------
def build_history_features(crop: pd.DataFrame) -> pd.DataFrame:
    """Per district-crop history on a complete calendar-year grid (so shift(1) really
    means 'previous calendar year'). All statistics are computed on shifted data."""
    c = crop.copy()
    # all flagged outliers are hidden from history; high-side ones are dropped entirely
    # Cleaning labels (is_outlier, outlier_dir) come from make_dataset and are treated as
    # given inputs. They use full-series statistics: a documented, minor limitation.
    c["outlier_high"] = ((c.is_outlier == 1) & (c.outlier_dir > 0)).astype(int)
    c["outlier_low"] = ((c.is_outlier == 1) & (c.outlier_dir <= 0)).astype(int)

    pairs = c[KEY].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({"year": list(YEARS)}), how="cross")
    grid = grid.merge(c[KEY + ["year", "yield_kg_ha", "area_1000ha", "is_outlier", "outlier_high",
                               "outlier_low"]], on=KEY + ["year"], how="left")
    grid = grid.sort_values(KEY + ["year"]).reset_index(drop=True)

    grid["hy"] = grid.yield_kg_ha.where(grid.is_outlier.fillna(0) == 0)
    grid["harea"] = grid.area_1000ha.where(grid.is_outlier.fillna(0) == 0)
    g = grid.groupby(KEY, sort=False)

    s1 = g["hy"].shift(1)
    grid["y_lag1"], grid["y_lag2"], grid["y_lag3"] = s1, g["hy"].shift(2), g["hy"].shift(3)
    grid["_s1"] = s1
    g = grid.groupby(KEY, sort=False)
    roll = lambda w, mp, fn: g._s1.transform(lambda s: getattr(s.rolling(w, min_periods=mp), fn)())
    grid["y_roll3"] = roll(3, 2, "mean")
    grid["y_roll5"] = roll(5, 3, "mean")
    grid["y_roll8_std"] = roll(8, 4, "std")
    grid["y_cv8"] = grid.y_roll8_std / roll(8, 4, "mean")
    grid["y_max5"] = roll(5, 3, "max")
    grid["y_min5"] = roll(5, 3, "min")
    grid["n_hist10"] = g._s1.transform(lambda s: s.rolling(10, min_periods=1).count())

    # local log-linear trend over the previous 10 years, from running sums
    t0 = grid.year - 2000
    M = grid["hy"].notna().astype(float)
    Y = np.log(grid["hy"]).fillna(0)
    parts = {"n": M, "sx": M * t0, "sy": Y, "sxy": M * t0 * Y, "sxx": M * t0 * t0}
    for k, v in parts.items():
        grid[f"_{k}"] = v
    g = grid.groupby(KEY, sort=False)
    S = {k: g[f"_{k}"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum()) for k in parts}
    den = S["n"] * S["sxx"] - S["sx"] ** 2
    slope = (S["n"] * S["sxy"] - S["sx"] * S["sy"]) / den.where(den > 1e-9)
    slope = slope.where(S["n"] >= 6).clip(-0.08, 0.08)           # guard against wild extrapolation
    icpt = (S["sy"] - slope * S["sx"]) / S["n"]
    grid["y_trend_slope"] = slope
    grid["y_trend_pred"] = np.exp(icpt + slope * t0)

    # area history (sowing area is known before harvest)
    grid["_a1"] = grid.groupby(KEY, sort=False)["harea"].shift(1)
    grid["area_roll5"] = grid.groupby(KEY, sort=False)._a1.transform(lambda s: s.rolling(5, min_periods=3).mean())
    grid["area_ratio"] = grid.area_1000ha / grid.area_roll5

    drop = [c_ for c_ in grid.columns if c_.startswith("_")] + ["hy", "harea"]
    return grid.drop(columns=drop)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def assign_split(year: pd.Series) -> pd.Series:
    return np.select([year <= C.TRAIN_END, year.between(*C.VALID_YEARS), year.between(*C.TEST_YEARS)],
                     ["train", "valid", "test"], default="none")


def build_model_table(crop, rain, dist, points) -> pd.DataFrame:
    c = crop[(crop.is_aggregate == 0) & crop.crop.isin(C.MODEL_CROPS)]
    hist = build_history_features(c)
    rf = build_rain_features(rain)

    t = hist[hist.yield_kg_ha.notna() & (hist.outlier_high.fillna(0) == 0)
             & (hist.year <= C.LAST_RAIN_YEAR)].copy()
    t = t.merge(rf, on=["dist_code", "year"], how="left")
    t = t.merge(dist[["dist_code", "dist_name", "state_name"]], on="dist_code", how="left")
    t = t.merge(points, on="dist_code", how="left")
    t["season_type"] = t.crop.map(C.CROP_SEASON)
    t["is_failure_outlier"] = t.outlier_low.fillna(0).astype(int)

    t["log_yield"] = np.log(t.yield_kg_ha)
    t["target_logratio"] = np.log(t.yield_kg_ha / t.y_roll5)      # modelling target in Phase 5
    t["split"] = assign_split(t.year)
    t = t.drop(columns=["is_outlier", "outlier_high", "outlier_low"])
    return t.sort_values(KEY + ["year"]).reset_index(drop=True)


def district_points():
    cached = C.PROCESSED / "district_points.csv"
    if cached.exists():
        return pd.read_csv(cached)
    from src.geo import district_geometries
    geo, _ = district_geometries()
    p = geo.geometry.representative_point()
    return pd.DataFrame({"dist_code": geo.dist_code, "lon": p.x, "lat": p.y})


# columns that must never be used as model inputs
NON_FEATURES = ["yield_kg_ha", "log_yield", "target_logratio", "split", "dist_name", "dist_code",
                "is_failure_outlier", "weather_complete"]
CATEGORICAL = ["crop", "state_name", "season_type"]
ID_COLS = ["year"]          # kept in the table; whether 'year' is a model input is decided in Phase 5


def feature_columns(t: pd.DataFrame, include_year: bool = False) -> list:
    cols = [c for c in t.columns if c not in NON_FEATURES]
    return cols if include_year else [c for c in cols if c != "year"]


def main():
    crop = pd.read_csv(C.PROCESSED / "crop_yield.csv")
    rain = pd.read_csv(C.PROCESSED / "rainfall.csv")
    dist = pd.read_csv(C.PROCESSED / "districts.csv")
    t = build_model_table(crop, rain, dist, district_points())
    t.to_parquet(C.PROCESSED / "model_table.parquet", index=False)
    summary = {"rows": len(t), "columns": t.shape[1],
               "rows_by_split": t.split.value_counts().to_dict(),
               "rows_by_crop": t.crop.value_counts().to_dict(),
               "failure_outliers_kept": int(t.is_failure_outlier.sum())}
    (C.REPORTS / "model_table_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return t


if __name__ == "__main__":
    main()
