"""Prediction service used by the Streamlit app and the FastAPI.

Features are rebuilt with the SAME functions used in training (src.features), so there
is no train/serve skew. A "what-if" multiplies the Jun-Sep rainfall of the chosen year.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import features as F

MODELS = C.ROOT / "models"
MONSOON_MONTHS = ["rain_jun", "rain_jul", "rain_aug", "rain_sep"]


class Predictor:
    def __init__(self):
        self.meta = json.loads((MODELS / "metadata.json").read_text())
        self.model = joblib.load(MODELS / "final_model.joblib")
        crop = pd.read_csv(C.PROCESSED / "crop_yield.csv")
        self.crop = crop[(crop.is_aggregate == 0) & crop.crop.isin(C.MODEL_CROPS)]
        self.rain = pd.read_csv(C.PROCESSED / "rainfall.csv")
        self.dist = pd.read_csv(C.PROCESSED / "districts.csv")
        self.points = pd.read_csv(C.PROCESSED / "district_points.csv")
        self.available = (self.crop.groupby(["dist_code", "crop"]).year.agg(["min", "max", "size"])
                          .query("size >= 15").reset_index())
        self.trained_through = self.meta["final_trained_through"]

    # ------------------------------------------------------------------ helpers
    def crops_for(self, dist_code):
        return sorted(self.available[self.available.dist_code == dist_code].crop.unique())

    def years_for(self, dist_code, crop):
        y = self.crop[(self.crop.dist_code == dist_code) & (self.crop.crop == crop)
                      & (self.crop.year <= C.LAST_RAIN_YEAR) & (self.crop.is_outlier == 0)].year
        return sorted(y.unique())

    def _rain_subset(self, dist_code, year, factor):
        r = self.rain[self.rain.dist_code == dist_code].copy()
        if factor != 1.0:
            m = r.year == year
            r.loc[m, MONSOON_MONTHS] = r.loc[m, MONSOON_MONTHS] * factor
            # keep the season totals consistent with the edited months
            r.loc[m, "monsoon_mm"] = r.loc[m, MONSOON_MONTHS].sum(axis=1, min_count=4)
            other = r.loc[m, ["rain_jan", "rain_feb", "rain_mar", "rain_apr", "rain_may", "rain_oct",
                              "rain_nov", "rain_dec"]].sum(axis=1, min_count=8)
            r.loc[m, "annual_mm"] = r.loc[m, "monsoon_mm"] + other
        return r

    def build_row(self, dist_code, crop, year, monsoon_factor=1.0):
        sub = self.crop[(self.crop.dist_code == dist_code) & (self.crop.crop == crop)]
        if sub.empty:
            raise ValueError("no data for this district and crop")
        hist = F.build_history_features(sub)
        row = hist[hist.year == year]
        if row.empty or row.y_roll5.isna().all():
            raise ValueError("not enough yield history before this year")
        rf = F.build_rain_features(self._rain_subset(dist_code, year, monsoon_factor))
        row = row.merge(rf, on=["dist_code", "year"], how="left")
        row = row.merge(self.dist[["dist_code", "dist_name", "state_name"]], on="dist_code", how="left")
        row = row.merge(self.points, on="dist_code", how="left")
        row["season_type"] = row.crop.map(C.CROP_SEASON)
        return row.reset_index(drop=True)

    def _model_predict(self, row):
        feats = self.meta["features"]
        X = row[feats].copy()
        kind = self.meta["kind"]
        if kind in ("lightgbm", "xgboost"):
            for c, lv in self.meta["levels"].items():
                X[c] = pd.Categorical(X[c], categories=lv)
        return float(np.asarray(self.model.predict(X))[0])

    # ------------------------------------------------------------------ public
    def predict(self, dist_code, crop, year, monsoon_factor=1.0):
        row = self.build_row(dist_code, crop, year, monsoon_factor)
        base = float(row.y_roll5.iloc[0])
        pred_lr = self._model_predict(row)
        q10, q90 = self.meta["conformal_signed_residual_q10_q90"][crop]
        actual = row.yield_kg_ha.iloc[0]
        spi = row.spi_monsoon.iloc[0]
        return {
            "dist_code": int(dist_code), "district": row.dist_name.iloc[0], "state": row.state_name.iloc[0],
            "crop": crop, "year": int(year), "monsoon_factor": float(monsoon_factor),
            "baseline_kg_ha": base,
            "predicted_kg_ha": base * float(np.exp(pred_lr)),
            "lo80_kg_ha": base * float(np.exp(pred_lr + q10)), "hi80_kg_ha": base * float(np.exp(pred_lr + q90)),
            "actual_kg_ha": None if pd.isna(actual) else float(actual),
            "weather_available": bool(pd.notna(row.monsoon_ratio.iloc[0])),
            "monsoon_ratio": None if pd.isna(row.monsoon_ratio.iloc[0]) else float(row.monsoon_ratio.iloc[0]),
            "spi_monsoon": None if pd.isna(spi) else float(spi),
            "in_sample": bool(year <= self.trained_through),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def spi_label(spi):
    if spi is None:
        return "unknown"
    return ("extremely dry" if spi <= -2 else "severely dry" if spi <= -1.5 else "moderately dry" if spi <= -1
            else "near normal" if spi < 1 else "moderately wet" if spi < 1.5 else "very wet")
