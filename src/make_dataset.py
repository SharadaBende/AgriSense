"""Phase 2 pipeline: raw ICRISAT files -> clean tidy tables -> SQLite database.

Run:  python -m src.make_dataset
"""
import json
import sqlite3

import numpy as np
import pandas as pd

from src import config as C

KEYS = ["Dist Code", "Year"]


# --------------------------------------------------------------------------
# Districts
# --------------------------------------------------------------------------
def build_districts(crop_raw: pd.DataFrame, rain_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["Dist Code", "Dist Name", "State Code", "State Name"]
    d = pd.concat([crop_raw[cols], rain_raw[cols]]).drop_duplicates()
    d = d.drop_duplicates("Dist Code")
    d.columns = ["dist_code", "dist_name", "state_code", "state_name"]
    d["dist_name"] = d["dist_name"].str.strip()
    d["state_name"] = d["state_name"].str.strip().replace(C.STATE_RENAME)
    return d.sort_values("dist_code").reset_index(drop=True)


# --------------------------------------------------------------------------
# Crops: wide -> long
# --------------------------------------------------------------------------
def build_crop_long(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    num_cols = df.columns[5:]
    df[num_cols] = df[num_cols].replace(C.MISSING_CODE, np.nan)

    frames = []
    for crop, (group, is_agg) in C.CROP_META.items():
        a, p, y = (f"{crop} AREA (1000 ha)", f"{crop} PRODUCTION (1000 tons)",
                   f"{crop} YIELD (Kg per ha)")
        t = df[KEYS + [a, p, y]].copy()
        t.columns = ["dist_code", "year", "area_1000ha", "production_1000t", "yield_reported"]
        t.insert(2, "crop", crop.title())
        t["crop_group"] = group
        t["is_aggregate"] = int(is_agg)
        frames.append(t)
    long = pd.concat(frames, ignore_index=True)

    n_all = len(long)
    # 'not grown / not reported': any of area, production missing or <= 0
    has_area = long.area_1000ha > 0
    has_prod = long.production_1000t > 0
    valid = has_area & has_prod
    # three mutually exclusive reasons a cell is dropped
    stats = {
        "crop_district_year_cells": int(n_all),
        "dropped_not_grown_or_missing": int((~has_area & ~has_prod).sum()),
        "dropped_area_without_production": int((has_area & ~has_prod).sum()),
        "dropped_production_without_area": int((~has_area & has_prod).sum()),
    }
    long = long[valid].copy()
    stats["kept_observations"] = int(len(long))

    # yield is recomputed so area/production/yield are always consistent
    long["yield_kg_ha"] = long.production_1000t / long.area_1000ha * 1000
    rel = (long.yield_kg_ha - long.yield_reported).abs() / long.yield_reported
    long["yield_mismatch"] = (rel > 0.05).astype(int)
    stats["yield_mismatch_gt5pct"] = int(long.yield_mismatch.sum())

    # outlier flag: robust z-score of log-yield inside each (crop, district)
    # (flagged, NOT dropped - the modelling phase decides what to do)
    ly = np.log(long.yield_kg_ha)
    grp = ly.groupby([long.crop, long.dist_code])
    med = grp.transform("median")
    mad = (ly - med).abs().groupby([long.crop, long.dist_code]).transform("median")
    n = grp.transform("size")
    rz = (ly - med) / (1.4826 * mad.replace(0, np.nan))
    long["is_outlier"] = ((rz.abs() > 4) & (n >= 10)).astype(int)
    # direction vs the district-crop median: +1 high (likely data error), -1 low (error or real failure)
    long["outlier_dir"] = np.where(long.is_outlier == 1, np.sign(ly - med), 0).astype(int)
    stats["flagged_outliers"] = int(long.is_outlier.sum())

    long = long.drop(columns="yield_reported")
    return long.sort_values(["dist_code", "crop", "year"]).reset_index(drop=True), stats


def build_land_use(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    cols = [f"{c} AREA (1000 ha)" for c in C.AREA_ONLY]
    df[cols] = df[cols].replace(C.MISSING_CODE, np.nan)
    out = df[KEYS + cols].copy()
    out.columns = ["dist_code", "year"] + [c.lower().replace(" ", "_") + "_area_1000ha"
                                           for c in C.AREA_ONLY]
    return out


# --------------------------------------------------------------------------
# Rainfall
# --------------------------------------------------------------------------
def build_rainfall(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    mcols = [f"{m} RAINFALL (Millimeters)" for m in C.MONTHS]
    df[mcols + ["ANNUAL RAINFALL (Millimeters)"]] = df[
        mcols + ["ANNUAL RAINFALL (Millimeters)"]].replace(C.MISSING_CODE, np.nan)

    r = df[KEYS + mcols + ["ANNUAL RAINFALL (Millimeters)"]].copy()
    r.columns = ["dist_code", "year"] + [f"rain_{m}" for m in C.MON3] + ["annual_reported"]
    mon = [f"rain_{m}" for m in C.MON3]

    # months that are all-zero in a year = failed record, not a real drought
    all_zero = (r[mon].fillna(0).sum(axis=1) == 0)
    r.loc[all_zero, mon] = np.nan
    stats = {"rainfall_rows": int(len(r)), "all_zero_or_missing_year": int(all_zero.sum())}

    # derived totals only when ALL contributing months are present
    r["annual_mm"] = r[mon].sum(axis=1, min_count=12)
    for s, ms in C.SEASONS.items():
        r[f"{s}_mm"] = r[[f"rain_{m}" for m in ms]].sum(axis=1, min_count=len(ms))
    ok = r.annual_mm.notna() & r.annual_reported.notna()
    stats["annual_sum_vs_reported_within_2pct"] = round(float(
        ((r.annual_mm[ok] - r.annual_reported[ok]).abs() / r.annual_reported[ok].clip(lower=1) < 0.02).mean() * 100), 1)
    stats["rows_with_valid_annual"] = int(r.annual_mm.notna().sum())
    return r.drop(columns="annual_reported"), stats


def build_normals(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    mcols = [f"{m} NORMAL RAINFALL (Millimeters)" for m in C.MONTHS]
    df[mcols] = df[mcols].replace(C.MISSING_CODE, np.nan)
    n = df[["Dist Code"] + mcols].copy()
    n.columns = ["dist_code"] + [f"normal_{m}" for m in C.MON3]
    n["normal_annual_mm"] = n[[f"normal_{m}" for m in C.MON3]].sum(axis=1, min_count=12)
    n["normal_monsoon_mm"] = n[[f"normal_{m}" for m in C.SEASONS["monsoon"]]].sum(axis=1, min_count=4)
    return n


# --------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------
VIEW_SQL = """
CREATE VIEW IF NOT EXISTS v_crop_rain AS
SELECT c.dist_code, d.dist_name, d.state_name, c.year, c.crop, c.crop_group,
       c.area_1000ha, c.production_1000t, c.yield_kg_ha, c.is_outlier,
       r.annual_mm, r.monsoon_mm, r.pre_monsoon_mm, r.winter_mm, r.post_monsoon_mm,
       n.normal_annual_mm, n.normal_monsoon_mm
FROM crop_yield c
JOIN districts d USING (dist_code)
LEFT JOIN rainfall r ON r.dist_code = c.dist_code AND r.year = c.year
LEFT JOIN rainfall_normal n ON n.dist_code = c.dist_code
WHERE c.is_aggregate = 0;
"""


def write_sqlite(tables: dict) -> None:
    C.DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(C.DB_PATH)
    for name, df in tables.items():
        df.to_sql(name, con, index=False)
    con.executescript("""
        CREATE INDEX ix_cy_key  ON crop_yield (dist_code, year);
        CREATE INDEX ix_cy_crop ON crop_yield (crop, year);
        CREATE INDEX ix_rain    ON rainfall (dist_code, year);
        CREATE UNIQUE INDEX ix_dist ON districts (dist_code);
    """)
    con.executescript(VIEW_SQL)
    con.commit()
    con.close()


def main() -> dict:
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    crop_raw = pd.read_csv(C.CROP_FILE)
    rain_raw = pd.read_csv(C.RAIN_FILE)
    norm_raw = pd.read_csv(C.NORMAL_FILE)

    districts = build_districts(crop_raw, rain_raw)
    crop, crop_stats = build_crop_long(crop_raw)
    land = build_land_use(crop_raw)
    rain, rain_stats = build_rainfall(rain_raw)
    normals = build_normals(norm_raw)

    tables = {"districts": districts, "crop_yield": crop, "land_use": land,
              "rainfall": rain, "rainfall_normal": normals}
    for name, df in tables.items():
        df.to_csv(C.PROCESSED / f"{name}.csv", index=False)
    write_sqlite(tables)

    report = {"districts": int(len(districts)), "states": int(districts.state_name.nunique()),
              "years": [int(crop.year.min()), int(crop.year.max())],
              "crop": crop_stats, "rainfall": rain_stats}
    (C.REPORTS / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
