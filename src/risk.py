"""Phase 6: agricultural risk analysis (descriptive, historical 1966-2017).

Per district-crop:   volatility, downside frequency, expected shortfall (CVaR),
                     drought sensitivity (loss in SPI < -1 years, slope on SPI)
Per district:        K-Means risk tier (Low / Medium / High)
Events:              crop-failure years and whether they coincide with drought
Simulation:          bootstrap Monte Carlo of next-season yield / revenue, normal vs drought

Run:  python -m src.risk
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import features as F

MIN_OBS = 20


def _detrended(crop: pd.DataFrame) -> pd.DataFrame:
    c = crop[(crop.is_aggregate == 0)].copy()
    c = c[~((c.is_outlier == 1) & (c.outlier_dir > 0))]            # drop likely data errors, keep real failures
    c = c.sort_values(["dist_code", "crop", "year"])
    base = c.groupby(["dist_code", "crop"]).yield_kg_ha.transform(
        lambda s: s.rolling(11, center=True, min_periods=6).mean())
    c["ratio"] = c.yield_kg_ha / base
    return c


def get_ratios():
    """Cached detrended-ratio table (written by main()); falls back to rebuilding it."""
    p = C.PROCESSED / "risk_ratios.parquet"
    return pd.read_parquet(p) if p.exists() else load_ratios()


def load_ratios():
    crop = pd.read_csv(C.PROCESSED / "crop_yield.csv")
    rain = pd.read_csv(C.PROCESSED / "rainfall.csv")
    rf = F.build_rain_features(rain)[["dist_code", "year", "spi_monsoon", "monsoon_ratio"]]
    d = pd.read_csv(C.PROCESSED / "districts.csv")
    c = _detrended(crop).merge(rf, on=["dist_code", "year"], how="left").merge(
        d[["dist_code", "dist_name", "state_name"]], on="dist_code")
    return c[c.ratio.notna()]


def district_crop_metrics(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dc, cr), g in c.groupby(["dist_code", "crop"]):
        if len(g) < MIN_OBS:
            continue
        r = g.ratio.to_numpy()
        k = max(2, int(np.ceil(0.1 * len(r))))
        worst = np.sort(r)[:k]
        gd = g.dropna(subset=["spi_monsoon"])
        dry = gd[gd.spi_monsoon < -1]
        beta = np.polyfit(gd.spi_monsoon, gd.ratio, 1)[0] if len(gd) >= 15 else np.nan
        rows.append({
            "dist_code": dc, "dist_name": g.dist_name.iloc[0], "state_name": g.state_name.iloc[0], "crop": cr,
            "n_years": len(g), "cv_detrended": float(np.std(r)),
            "downside_freq": float((r < 0.85).mean()), "failure_freq": float((r < 0.70).mean()),
            "cvar10_loss": float(1 - worst.mean()),
            "drought_years_n": len(dry),
            "drought_loss": float(1 - dry.ratio.mean()) if len(dry) >= 3 else np.nan,
            "drought_beta": float(beta) if beta == beta else np.nan,
            "recent_yield_kg_ha": float(g[g.year >= 2008].yield_kg_ha.mean()) if (g.year >= 2008).sum() >= 3 else np.nan})
    m = pd.DataFrame(rows)
    z = m[["cv_detrended", "downside_freq", "cvar10_loss"]].apply(lambda s: (s - s.mean()) / s.std())
    m["risk_score"] = (z.mean(axis=1).rank(pct=True) * 100).round(1)
    return m


TIERS = ["Low", "Medium", "High"]


def kmeans_tiers(df: pd.DataFrame, cols, k=3, seed=42):
    X = StandardScaler().fit_transform(df[cols])
    sil = {kk: float(silhouette_score(X, KMeans(kk, n_init=10, random_state=seed).fit_predict(X)))
           for kk in range(2, 6)}
    km = KMeans(k, n_init=20, random_state=seed).fit(X)
    order = pd.Series(km.cluster_centers_[:, 0]).rank().astype(int) - 1       # by volatility
    return np.array(TIERS)[order[km.labels_].to_numpy()], sil


def district_level(m: pd.DataFrame):
    d = m.groupby(["dist_code", "dist_name", "state_name"]).agg(
        n_crops=("crop", "nunique"), cv=("cv_detrended", "mean"), downside=("downside_freq", "mean"),
        cvar10=("cvar10_loss", "mean"), drought_loss=("drought_loss", "mean"),
        risk_score=("risk_score", "mean")).reset_index()
    d = d[d.n_crops >= 3].copy()
    d["risk_tier"], sil = kmeans_tiers(d, ["cv", "downside", "cvar10"])
    return d, sil


def failure_events(c: pd.DataFrame) -> pd.DataFrame:
    e = c[c.ratio < 0.6][["dist_code", "dist_name", "state_name", "crop", "year", "yield_kg_ha", "ratio",
                          "spi_monsoon"]].copy()
    e["drought_year"] = (e.spi_monsoon < -1).astype("Int64").where(e.spi_monsoon.notna())
    return e.sort_values("ratio")


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------
def simulate(c: pd.DataFrame, dist_code: int, crop: str, scenario="normal", n=10000, area_ha=1.0,
             price_per_kg=None, seed=7):
    """Bootstrap next-season yield = recent baseline x resampled historical yield ratio.
    scenario='drought' resamples ratios from SPI(monsoon) < -1 years, partially pooled with
    the state's drought years for the same crop (weight on own data = n / (n + 10))."""
    g = c[(c.dist_code == dist_code) & (c.crop == crop)].sort_values("year")
    if len(g) < MIN_OBS:
        raise ValueError("not enough history for this district-crop")
    baseline = float(g.yield_kg_ha.tail(5).mean())
    rng = np.random.default_rng(seed)
    pool_note = "district history"
    regional_n = 0
    if scenario == "drought":
        # Partial pooling: a district has only a handful of SPI < -1 years, so its own drought
        # sample is noisy. Blend it with the same crop's drought years in the same state
        # (or all-India if the state sample is small); weight on own data = n / (n + 10).
        own = g[g.spi_monsoon < -1].ratio.dropna().to_numpy()
        state = g.state_name.iloc[0]
        reg = c[(c.crop == crop) & (c.state_name == state) & (c.spi_monsoon < -1)].ratio.dropna().to_numpy()
        reg_label = state
        if len(reg) < 30:
            reg = c[(c.crop == crop) & (c.spi_monsoon < -1)].ratio.dropna().to_numpy()
            reg_label = "all-India"
        w = len(own) / (len(own) + 10)
        use_own = rng.random(n) < w
        draws = np.where(use_own, rng.choice(own, n) if len(own) else 1.0, rng.choice(reg, n))
        regional_n = len(reg)
        pool_note = (f"{w:.0%} own drought years (n={len(own)}) + {1-w:.0%} {reg_label} {crop} drought years "
                     f"(n={regional_n})")
        pool = own
        shocks = draws * np.exp(rng.normal(0, 0.03, n))
    else:
        pool = g.ratio.to_numpy()
        shocks = rng.choice(pool, n) * np.exp(rng.normal(0, 0.03, n))        # small kernel smoothing
    y = baseline * shocks
    q = np.percentile(y, [5, 25, 50, 75, 95])
    var5 = np.percentile(y, 5)
    out = {"baseline_kg_ha": baseline, "scenario": scenario, "pool": pool_note, "pool_size": int(len(pool)),
           "mean": float(y.mean()), "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
           "prob_below_80pct_of_baseline": float((y < 0.8 * baseline).mean()),
           "cvar5_kg_ha": float(y[y <= var5].mean()), "samples": y}
    if price_per_kg:
        rev = y * area_ha * price_per_kg
        out["revenue"] = {"mean": float(rev.mean()), "p5": float(np.percentile(rev, 5)),
                          "p50": float(np.percentile(rev, 50)), "p95": float(np.percentile(rev, 95)),
                          "prob_below_baseline_revenue_80pct": float((y < 0.8 * baseline).mean())}
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


# --------------------------------------------------------------------------------------
def figures(dl, m, fe, sil):
    import matplotlib.pyplot as plt
    from src.geo import district_geometries
    import geopandas as gpd
    FIG = C.REPORTS / "figures"
    geo, _ = district_geometries()
    geo["geometry"] = geo.geometry.simplify(0.02)
    base = gpd.read_file(C.RAW / "india_district_boundaries.geojson")
    base["geometry"] = base.geometry.simplify(0.02)
    g = geo.merge(dl[["dist_code", "risk_tier"]], on="dist_code", how="left")
    colors = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#d62828"}
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    base.plot(ax=ax, color="#f3f3f3", edgecolor="white", linewidth=.15)
    g[g.risk_tier.isna()].plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=.2)
    for t_, col in colors.items():
        g[g.risk_tier == t_].plot(ax=ax, color=col, edgecolor="white", linewidth=.2, label=f"{t_} risk")
    ax.legend(loc="lower left", frameon=False); ax.axis("off")
    cnt = dl.risk_tier.value_counts().to_dict()
    ax.set_title(f"District crop-yield risk tiers (K-Means): {cnt.get('High',0)} high, {cnt.get('Medium',0)} medium, "
                 f"{cnt.get('Low',0)} low", loc="left", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "risk_map.png", dpi=200); plt.close()

    # state profile
    st = dl.groupby("state_name").agg(score=("risk_score", "mean"), n=("dist_code", "size")).query("n >= 4") \
        .sort_values("score")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(st.index, st.score, color="#6d597a"); ax.set_xlabel("Mean district risk score (0-100)")
    ax.set_title("Average yield-risk score by state (states with >= 4 districts)", loc="left", fontweight="bold",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "risk_by_state.png", dpi=200); plt.close()

    # drought loss by crop
    dc = m.dropna(subset=["drought_loss"]).groupby("crop").drought_loss.agg(["median", "size"])
    dc = dc[dc["size"] >= 20].sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(dc.index, dc["median"] * 100, color="#d62828")
    ax.set_xlabel("Median yield loss vs local trend in SPI < -1 monsoon years (%)")
    ax.set_title("Which crops lose most in drought years", loc="left", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "risk_drought_loss_by_crop.png", dpi=200); plt.close()

    # failure events by year
    ev = fe.groupby("year").size()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(ev.index, ev.values, color=np.where(ev.index.isin([1972, 1979, 1987, 2002, 2009, 2014, 2015]),
                                               "#d62828", "#9db4c0"))
    ax.set_title("Crop-failure events (yield < 60% of local trend) per year; red = known drought years",
                 loc="left", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "risk_failure_years.png", dpi=200); plt.close()


def main():
    c = load_ratios()
    c.to_parquet(C.PROCESSED / "risk_ratios.parquet", index=False)
    m = district_crop_metrics(c)
    sub = m.dropna(subset=["cv_detrended", "downside_freq", "cvar10_loss"])
    tiers, _ = kmeans_tiers(sub, ["cv_detrended", "downside_freq", "cvar10_loss"])
    m.loc[sub.index, "crop_tier"] = tiers
    dl, sil = district_level(m)
    fe = failure_events(c)
    ex = C.ROOT / "exports"; ex.mkdir(exist_ok=True)
    m.round(4).to_csv(ex / "district_crop_risk.csv", index=False)
    dl.round(4).to_csv(ex / "district_risk_tiers.csv", index=False)
    fe.round(4).to_csv(ex / "crop_failure_events.csv", index=False)
    figures(dl, m, fe, sil)

    yrs = fe.groupby("year").size()
    known = [1972, 1979, 1987, 2002, 2009, 2014, 2015]
    summ = {"district_crop_pairs": int(len(m)), "districts_tiered": int(len(dl)),
            "silhouette_by_k": sil, "tier_counts": dl.risk_tier.value_counts().to_dict(),
            "tier_profile": dl.groupby("risk_tier")[["cv", "downside", "cvar10", "risk_score"]].mean().round(3).to_dict("index"),
            "failure_events": int(len(fe)),
            "failure_events_in_known_drought_years_pct": round(float(yrs.loc[yrs.index.isin(known)].sum() / yrs.sum() * 100), 1),
            "known_drought_years_share_of_years_pct": round(len(known) / yrs.shape[0] * 100, 1),
            "failure_with_spi_lt_-1_pct": round(float(fe.drought_year.dropna().mean() * 100), 1),
            "top_risk_districts": dl.sort_values("risk_score", ascending=False).head(8)[["dist_name", "state_name", "risk_score"]].to_dict("records"),
            "safest_districts": dl.sort_values("risk_score").head(8)[["dist_name", "state_name", "risk_score"]].to_dict("records"),
            "drought_loss_median_by_crop_pct": (m.dropna(subset=["drought_loss"]).groupby("crop").drought_loss.median().mul(100).round(1).sort_values(ascending=False).head(6).to_dict())}
    (C.REPORTS / "risk_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
