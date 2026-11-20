"""Does weather actually help? Same tuned LightGBM, different feature sets, train-only fit.
Run: python -m src.ablation"""
import json

import numpy as np

from src import config as C
from src import features as F
from src import train as T

WEATHER = ["annual_ratio", "monsoon_ratio", "pre_ratio", "winter_ratio", "post_ratio", "spi_annual",
           "spi_monsoon", "spi_early", "spi_late", "jun_ratio", "jul_ratio", "aug_ratio", "sep_ratio",
           "n_deficit_months", "jul_aug_share", "monsoon_ratio_prev", "spi_monsoon_prev", "annual_ratio_prev",
           "monsoon_ratio_2yr", "clim_annual_mm", "clim_monsoon_mm"]
HISTORY = ["y_lag1", "y_lag2", "y_lag3", "y_roll3", "y_roll5", "y_roll8_std", "y_cv8", "y_max5", "y_min5",
           "n_hist10", "y_trend_slope", "y_trend_pred", "area_1000ha", "area_roll5", "area_ratio"]


def main():
    t = T.load_table()
    allf = F.feature_columns(t)
    tj = json.loads((C.REPORTS / "lgbm_tuning.json").read_text())
    levels = T.cat_levels(t)
    ok = t.is_failure_outlier == 0
    train = t[(t.split == "train") & ok]
    sets = {"history + location only (no weather)": [f for f in allf if f not in WEATHER],
            "weather + location only (no yield history except baseline)": [f for f in allf if f not in HISTORY or f == "y_roll5"],
            "full model (history + weather)": allf}
    out = {}
    for name, feats in sets.items():
        m = T.Wrap("lightgbm", tj["params"], int(tj["rounds"] * 1.1))
        m.fit(T.prep(train, feats, levels, "gbm"), train[T.TARGET])
        res = {}
        for sp in ["valid", "test"]:
            d = t[(t.split == sp) & ok & t.weather_complete]
            res[sp] = T.metrics(d, m.predict(T.prep(d, feats, levels, "gbm")))
        out[name] = {"n_features": len(feats), "valid_skill_pct": res["valid"]["skill_MAE_pct"],
                     "test_skill_pct": res["test"]["skill_MAE_pct"], "test_R2_deviation": res["test"]["R2_deviation"]}
        print(f"{name:62s} feats={len(feats):2d} valid {res['valid']['skill_MAE_pct']:+.1f}%  test {res['test']['skill_MAE_pct']:+.1f}%  R2dev {res['test']['R2_deviation']:+.3f}", flush=True)
    (C.REPORTS / "ablation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
