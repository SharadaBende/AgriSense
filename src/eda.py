"""Phase 3: exploratory analysis. Writes figures to reports/figures and
numbers to reports/eda_findings.json.   Run: python -m src.eda
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from src import config as C
from src.geo import district_geometries

warnings.filterwarnings("ignore")
FIG = C.REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})
PAL = {"Rice": "#2a9d8f", "Wheat": "#e9a03b", "Maize": "#e76f51", "Cotton": "#6d597a",
       "Groundnut": "#8d6e63", "Pearl Millet": "#4361ee", "Kharif Sorghum": "#7a9e3b"}
F = {}  # findings


def load():
    crop = pd.read_csv(C.PROCESSED / "crop_yield.csv")
    rain = pd.read_csv(C.PROCESSED / "rainfall.csv")
    norm = pd.read_csv(C.PROCESSED / "rainfall_normal.csv")
    dist = pd.read_csv(C.PROCESSED / "districts.csv")
    return crop, rain, norm, dist


def add_anomalies(crop, rain):
    """Within-district, trend-adjusted yield ratio and rainfall anomaly ratios."""
    c = crop[(crop.is_aggregate == 0) & (crop.is_outlier == 0)].copy()
    c = c.sort_values(["dist_code", "crop", "year"])
    roll = c.groupby(["dist_code", "crop"]).yield_kg_ha.transform(
        lambda s: s.rolling(11, center=True, min_periods=6).mean())
    c["yield_ratio"] = c.yield_kg_ha / roll
    r = rain.copy()
    for col in ["annual_mm", "monsoon_mm", "pre_monsoon_mm", "winter_mm", "post_monsoon_mm"]:
        r[col.replace("_mm", "_ratio")] = r[col] / r.groupby("dist_code")[col].transform("mean")
    return c.merge(r, on=["dist_code", "year"], how="left")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def trend_pct(years, y):
    slope = np.polyfit(years, np.log(y), 1)[0]
    return (np.exp(slope) - 1) * 100


# ---------------------------------------------------------------- fig 1
def fig_trends(crop):
    c = crop[(crop.is_aggregate == 0) & (crop.is_outlier == 0)]
    nat = (c.groupby(["crop", "year"])[["production_1000t", "area_1000ha"]].sum()
           .assign(y=lambda d: d.production_1000t * 1000 / d.area_1000ha).y.reset_index())
    crops = ["Rice", "Wheat", "Maize", "Cotton"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    F["trend"] = {}
    for ax, cr in zip(axes.ravel(), crops):
        d = nat[nat.crop == cr]
        ax.plot(d.year, d.y, color=PAL[cr], alpha=.35, lw=1.2)
        ax.plot(d.year, d.y.rolling(5, center=True, min_periods=3).mean(), color=PAL[cr], lw=2.6)
        a = d[d.year < 1990]; b = d[d.year >= 1990]
        ga, gb = trend_pct(a.year, a.y), trend_pct(b.year, b.y)
        F["trend"][cr] = {"growth_pct_per_yr_1966_89": round(ga, 2), "growth_pct_per_yr_1990_2017": round(gb, 2),
                          "yield_1966_70": round(d[d.year <= 1970].y.mean()), "yield_2013_17": round(d[d.year >= 2013].y.mean())}
        ax.axvline(1990, color="grey", ls=":", lw=1)
        ax.set_title(f"{cr}: {ga:+.1f}%/yr before 1990, {gb:+.1f}%/yr after", loc="left")
        ax.set_ylabel("kg / ha")
    fig.suptitle("India's district yields: national area-weighted trend, 1966-2017", y=1.01, fontweight="bold")
    save("fig01_yield_trends.png")


# ---------------------------------------------------------------- fig 2
def fig_crop_mix(crop):
    c = crop[crop.is_aggregate == 0]
    g = c.groupby(["year", "crop_group"]).area_1000ha.sum().unstack().fillna(0)
    share = g.div(g.sum(axis=1), axis=0)
    order = ["cereal", "millet", "pulse", "oilseed", "cash"]
    cols = ["#2a9d8f", "#7a9e3b", "#e9a03b", "#e76f51", "#6d597a"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.stackplot(share.index, [share[o] for o in order], labels=[o.title() for o in order], colors=cols, alpha=.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlim(1966, 2017); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=5, frameon=False, bbox_to_anchor=(0, 1.12))
    s0, s1 = share.loc[1966:1970].mean(), share.loc[2013:2017].mean()
    F["crop_mix_share_1966_70_vs_2013_17"] = {k: [round(s0[k] * 100, 1), round(s1[k] * 100, 1)] for k in order}
    ax.set_title(f"Crop mix shift (share of recorded area): millets {s0['millet']*100:.0f}% -> {s1['millet']*100:.0f}%, "
                 f"oilseeds {s0['oilseed']*100:.0f}% -> {s1['oilseed']*100:.0f}%", loc="left", y=1.14)
    save("fig02_crop_mix.png")


# ---------------------------------------------------------------- fig 3
def fig_state_heat(crop, dist):
    c = crop[(crop.is_aggregate == 0) & (crop.is_outlier == 0)].merge(dist, on="dist_code")
    c["decade"] = (c.year // 10 * 10).astype(str) + "s"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, cr, cmap in zip(axes, ["Rice", "Wheat"], ["YlGn", "YlOrBr"]):
        d = c[c.crop == cr]
        t = d.groupby(["state_name", "decade"])[["production_1000t", "area_1000ha"]].sum()
        t = (t.production_1000t * 1000 / t.area_1000ha).unstack()
        t = t[t.index.isin(d.groupby("state_name").area_1000ha.sum().loc[lambda s: s > 500].index)]
        t = t.loc[t.iloc[:, -1].sort_values(ascending=False).index]
        sns.heatmap(t, annot=True, fmt=".0f", cmap=cmap, cbar=False, ax=ax, linewidths=.5, annot_kws={"size": 8})
        ax.set_title(f"{cr} yield (kg/ha) by state and decade", loc="left"); ax.set_xlabel(""); ax.set_ylabel("")
    save("fig03_state_decade_heatmap.png")
    r = c[(c.crop == "Rice") & (c.year >= 2008)].groupby("state_name")[["production_1000t", "area_1000ha"]].sum()
    F["rice_state_yield_2008_17"] = (r.production_1000t * 1000 / r.area_1000ha).round().sort_values(ascending=False).head(5).to_dict()


# ---------------------------------------------------------------- fig 4 & 5
BANDS = [(0, .70, "<70%"), (.70, .85, "70-85%"), (.85, 1.15, "85-115%"), (1.15, 1.30, "115-130%"), (1.30, 9, ">130%")]


def fig_rain_response(an):
    crops = ["Rice", "Maize", "Pearl Millet", "Kharif Sorghum", "Groundnut", "Cotton"]
    a = an[an.crop.isin(crops) & an.monsoon_ratio.notna() & an.yield_ratio.notna()].copy()
    a["band"] = pd.cut(a.monsoon_ratio, [b[0] for b in BANDS] + [9], labels=[b[2] for b in BANDS], right=False)
    t = a.groupby(["crop", "band"]).yield_ratio.agg(["mean", "sem", "size"]).reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    F["drought_penalty_pct"] = {}
    for ax, cr in zip(axes.ravel(), crops):
        d = t[t.crop == cr]
        ax.bar(range(len(d)), d["mean"] - 1, yerr=1.96 * d["sem"], bottom=1, color=PAL[cr], alpha=.9, capsize=3)
        ax.axhline(1, color="k", lw=.8)
        ax.set_xticks(range(len(d))); ax.set_xticklabels(d.band, fontsize=8)
        low = d[d.band == "<70%"]["mean"].iloc[0]
        F["drought_penalty_pct"][cr] = {"severe_deficit_lt70": round((low - 1) * 100, 1),
                                        "n_district_years": int(d[d.band == "<70%"]["size"].iloc[0])}
        ax.set_title(f"{cr}: {(low-1)*100:+.0f}% in severe deficit", loc="left", fontsize=11)
    for ax in axes[1]: ax.set_xlabel("Monsoon rain vs district's own average")
    for ax in axes[:, 0]: ax.set_ylabel("Yield vs local trend")
    fig.suptitle("How yields respond to monsoon rainfall (within-district, trend-adjusted; bars = 95% CI)",
                 y=1.0, fontweight="bold")
    save("fig04_rainfall_response.png")


def fig_corr(an):
    seasons = {"winter_ratio": "Winter\n(Jan-Feb)", "pre_monsoon_ratio": "Pre-monsoon\n(Mar-May)",
               "monsoon_ratio": "Monsoon\n(Jun-Sep)", "post_monsoon_ratio": "Post-monsoon\n(Oct-Dec)"}
    crops = ["Rice", "Maize", "Pearl Millet", "Kharif Sorghum", "Rabi Sorghum", "Groundnut", "Cotton",
             "Sugarcane", "Wheat", "Chickpea", "Rapeseed And Mustard"]
    rows = {}
    for cr in crops:
        d = an[(an.crop == cr) & an.yield_ratio.notna()]
        rows[cr] = {lab: d[["yield_ratio", k]].dropna().corr().iloc[0, 1] for k, lab in seasons.items()}
    m = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="RdBu", center=0, vmin=-.5, vmax=.5, ax=ax, linewidths=.5,
                cbar_kws={"label": "correlation with yield anomaly"})
    ax.set_title("Which season's rain matters for which crop", loc="left"); ax.set_ylabel("")
    F["rain_corr"] = m.round(2).to_dict("index")
    save("fig05_rain_season_correlation.png")


# ---------------------------------------------------------------- fig 6
KNOWN_DROUGHTS = [1972, 1979, 1987, 2002, 2009, 2014, 2015]


def fig_monsoon(rain):
    r = rain.copy()
    r["ratio"] = r.monsoon_mm / r.groupby("dist_code").monsoon_mm.transform("mean")
    y = r.groupby("year").ratio.agg(["mean", "count"])
    y = y[y["count"] >= 150]
    low = y["mean"].nsmallest(8)
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.bar(y.index, y["mean"] - 1, bottom=1, color=np.where(y["mean"] < .93, "#d62828", "#9db4c0"))
    ax.axhline(1, color="k", lw=.8)
    for yr in KNOWN_DROUGHTS:
        if yr in y.index:
            ax.annotate(str(yr), (yr, y.loc[yr, "mean"]), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color="#d62828")
    ax.set_title("Average monsoon rainfall across districts vs their long-run mean (red = >7% below)", loc="left")
    ax.set_ylabel("Monsoon rain ratio"); ax.set_ylim(.62, 1.32)
    F["driest_monsoon_years"] = {int(k): round(float(v), 3) for k, v in low.items()}
    F["known_droughts_in_8_driest"] = [int(k) for k in low.index if k in KNOWN_DROUGHTS]
    save("fig06_monsoon_history.png")


# ---------------------------------------------------------------- fig 7
def fig_coverage(crop):
    c = crop[crop.is_aggregate == 0]
    t = c.groupby(["crop", "year"]).dist_code.nunique().unstack().fillna(0)
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(t, cmap="viridis", ax=ax, cbar_kws={"label": "districts with valid data"}, xticklabels=5)
    ax.set_title("Data coverage: how many districts report each crop, per year", loc="left")
    ax.set_xlabel(""); ax.set_ylabel("")
    save("fig07_data_coverage.png")


# ---------------------------------------------------------------- fig 8
def fig_maps(an):
    geo, xw = district_geometries()
    geo["geometry"] = geo.geometry.simplify(0.02)
    import geopandas as gpd
    base = gpd.read_file(C.RAW / "india_district_boundaries.geojson")
    base["geometry"] = base.geometry.simplify(0.02)
    w = an[(an.crop == "Wheat") & (an.year >= 2008)].groupby("dist_code").yield_kg_ha.agg(["mean", "size"])
    w = w[w["size"] >= 6]["mean"].rename("wheat_yield")
    r = an[an.crop == "Rice"].dropna(subset=["yield_ratio"]).groupby("dist_code").yield_ratio.agg(["std", "size"])
    r = r[r["size"] >= 25]["std"].rename("rice_volatility")
    g = geo.merge(w, on="dist_code", how="left").merge(r, on="dist_code", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    for ax, col, cmap, ttl in [(axes[0], "wheat_yield", "YlOrBr", "Wheat yield, 2008-17 (kg/ha)"),
                               (axes[1], "rice_volatility", "Reds", "Rice yield volatility (SD of trend-adjusted yield)")]:
        base.plot(ax=ax, color="#f3f3f3", edgecolor="white", linewidth=.15)
        g.plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=.2)
        g.dropna(subset=[col]).plot(column=col, cmap=cmap, legend=True, ax=ax, edgecolor="white", linewidth=.2,
                                    legend_kwds={"shrink": .6})
        ax.set_title(ttl, loc="left"); ax.axis("off")
    fig.text(0.01, 0.01, "Dark grey = ICRISAT district without enough data. Pale areas = states outside the dataset or newer districts "
             "split off after 1966 (ICRISAT counts them inside their parent). See README.", fontsize=8, color="grey")
    save("fig08_district_maps.png")
    F["n_districts_wheat_map"] = int(w.shape[0]); F["n_districts_rice_vol_map"] = int(r.shape[0])
    F["rice_volatility_top5"] = (r.sort_values(ascending=False).head(5).round(3)
                                 .rename(lambda k: xw.set_index("dist_code").dist_name[k]).to_dict())


def main():
    crop, rain, norm, dist = load()
    an = add_anomalies(crop, rain)
    fig_trends(crop); fig_crop_mix(crop); fig_state_heat(crop, dist)
    fig_rain_response(an); fig_corr(an); fig_monsoon(rain); fig_coverage(crop); fig_maps(an)
    (C.REPORTS / "eda_findings.json").write_text(json.dumps(F, indent=2))
    print(json.dumps(F, indent=2))


if __name__ == "__main__":
    main()
