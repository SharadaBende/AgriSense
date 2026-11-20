"""Dashboard-ready CSVs for Power BI / Tableau.   Run: python -m src.exports"""
import sqlite3

import pandas as pd

from src import config as C
from src import risk


def main():
    ex = C.ROOT / "exports"
    ex.mkdir(exist_ok=True)
    con = sqlite3.connect(C.DB_PATH)
    h = pd.read_sql_query("SELECT * FROM v_crop_rain WHERE year <= 2015 AND is_outlier = 0", con)
    r = risk.get_ratios()[["dist_code", "crop", "year", "ratio", "spi_monsoon", "monsoon_ratio"]]
    h = h.merge(r.rename(columns={"ratio": "yield_ratio", "monsoon_ratio": "monsoon_ratio"}),
                on=["dist_code", "crop", "year"], how="left")
    pts = pd.read_csv(C.PROCESSED / "district_points.csv")
    h = h.merge(pts, on="dist_code", how="left")
    h.round(4).to_csv(ex / "yield_history.csv", index=False)
    d = pd.read_csv(ex / "district_risk_tiers.csv").merge(pts, on="dist_code", how="left")
    d.to_csv(ex / "district_risk_tiers.csv", index=False)
    print({"yield_history_rows": len(h), "districts_with_coords": int(d.lat.notna().sum())})


if __name__ == "__main__":
    main()
