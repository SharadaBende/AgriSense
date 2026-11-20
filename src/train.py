"""Phase 5: train, tune, compare, explain, calibrate.

Target  : target_logratio = log(yield / y_roll5)   (deviation from the local 5-year norm)
Predict : yield_hat = y_roll5 * exp(prediction)

Protocol (no peeking):
  * boosting rounds / hyper-parameters come from expanding-window CV INSIDE the training years
  * model selection uses the validation years (2005-09) only
  * the test years (2010-15) are used for reporting, never for choices
  * final model is refit on train+valid; prediction intervals are split-conformal
    (calibrated on validation residuals of the train-only model)

Run:  python -m src.train --trials 12
"""
import argparse
import json
import time
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
MODELS_DIR = C.ROOT / "models"
FIG = C.REPORTS / "figures"
SEED = 42
TARGET = "target_logratio"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_table(frac=1.0):
    t = pd.read_parquet(C.PROCESSED / "model_table.parquet")
    if frac < 1:
        t = t.sample(frac=frac, random_state=SEED)
    t = t[t.y_roll5.notna() & t[TARGET].notna()].copy()
    t["weather_complete"] = t.monsoon_ratio.notna() & t.spi_monsoon.notna()
    return t.reset_index(drop=True)


def cat_levels(t):
    return {c: sorted(t[c].dropna().unique().tolist()) for c in F.CATEGORICAL}


def prep(df, feats, levels, kind):
    X = df[feats].copy()
    if kind == "gbm":                      # lightgbm / xgboost: pandas category dtype
        for c, lv in levels.items():
            X[c] = pd.Categorical(X[c], categories=lv)
    return X


def metrics(df, pred):
    """df must hold yield_kg_ha, y_roll5, target_logratio. pred = predicted log-ratio."""
    yhat = df.y_roll5.to_numpy() * np.exp(pred)
    y = df.yield_kg_ha.to_numpy()
    e = yhat - y
    base = df.y_roll5.to_numpy() - y
    dev = df[TARGET].to_numpy()
    return {"n": int(len(df)),
            "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MedAPE_pct": float(np.median(np.abs(e) / y) * 100),
            "R2_yield": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "R2_deviation": float(1 - ((dev - pred) ** 2).sum() / ((dev - dev.mean()) ** 2).sum()),
            "MAE_baseline": float(np.abs(base).mean()),
            "skill_MAE_pct": float((1 - np.abs(e).mean() / np.abs(base).mean()) * 100)}


# ----------------------------------------------------------------------------
# model wrappers
# ----------------------------------------------------------------------------
class Wrap:
    def __init__(self, kind, params=None, rounds=None):
        self.kind, self.params, self.rounds = kind, params or {}, rounds
        self.kindX = "gbm" if kind in ("lightgbm", "xgboost") else "plain"
        self.model = None

    def _sk_pipe(self, X, est):
        num = [c for c in X.columns if c not in F.CATEGORICAL]
        pre = ColumnTransformer([
            ("n", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                            ("s", StandardScaler())]), num),
            ("c", OneHotEncoder(handle_unknown="ignore"), F.CATEGORICAL)])
        return Pipeline([("p", pre), ("m", est)])

    def fit(self, X, y, Xv=None, yv=None):
        k = self.kind
        if k == "ridge":
            self.model = self._sk_pipe(X, Ridge(alpha=self.params.get("alpha", 30.0))).fit(X, y)
        elif k == "rf":
            self.model = self._sk_pipe(X, RandomForestRegressor(
                n_estimators=120, max_depth=16, min_samples_leaf=25, max_features=0.5,
                max_samples=0.5, n_jobs=1, random_state=SEED)).fit(X, y)
        elif k == "lightgbm":
            p = {**dict(objective="regression", learning_rate=0.06, verbose=-1, seed=SEED, num_threads=1),
                 **self.params}
            ds = lgb.Dataset(X, y)
            if Xv is not None:
                dv = lgb.Dataset(Xv, yv, reference=ds)
                self.model = lgb.train(p, ds, 800, valid_sets=[dv],
                                       callbacks=[lgb.early_stopping(40, verbose=False)])
                self.best_iter = self.model.best_iteration
            else:
                self.model = lgb.train(p, ds, self.rounds)
        elif k == "xgboost":
            p = {**dict(learning_rate=0.06, max_depth=7, subsample=0.8, colsample_bytree=0.7,
                        min_child_weight=20, reg_lambda=2.0, tree_method="hist", enable_categorical=True,
                        random_state=SEED, n_jobs=1), **self.params}
            if Xv is not None:
                self.model = xgb.XGBRegressor(n_estimators=800, early_stopping_rounds=40, **p)
                self.model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
                self.best_iter = self.model.best_iteration + 1
            else:
                self.model = xgb.XGBRegressor(n_estimators=self.rounds, **p).fit(X, y)
        elif k == "catboost":
            p = {**dict(learning_rate=0.08, depth=7, l2_leaf_reg=5, random_seed=SEED, verbose=0,
                        thread_count=1, cat_features=F.CATEGORICAL), **self.params}
            if Xv is not None:
                self.model = CatBoostRegressor(iterations=800, early_stopping_rounds=40, **p)
                self.model.fit(X, y, eval_set=(Xv, yv))
                self.best_iter = self.model.get_best_iteration() + 1
            else:
                self.model = CatBoostRegressor(iterations=self.rounds, **p).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X))


# ----------------------------------------------------------------------------
# cv + tuning
# ----------------------------------------------------------------------------
def cv_splits(t):
    ok = t.is_failure_outlier == 0
    for train_end, (v0, v1) in C.CV_FOLDS:
        tr = t[(t.year <= train_end) & ok]
        va = t[t.year.between(v0, v1) & ok]
        yield tr, va


def tune_lgbm(t, feats, levels, trials):
    folds = list(cv_splits(t))

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 95),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 250),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0), bagging_freq=1,
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
            objective=trial.suggest_categorical("objective", ["regression", "huber"]))
        if params["objective"] == "huber":
            params["alpha"] = 0.35
        maes, iters = [], []
        for tr, va in folds:
            m = Wrap("lightgbm", params).fit(prep(tr, feats, levels, "gbm"), tr[TARGET],
                                             prep(va, feats, levels, "gbm"), va[TARGET])
            maes.append(np.abs(m.predict(prep(va, feats, levels, "gbm")) - va[TARGET]).mean())
            iters.append(m.best_iter)
        trial.set_user_attr("rounds", int(np.mean(iters)))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params)
    if best.get("objective") == "huber":
        best["alpha"] = 0.35
    best.update(bagging_freq=1)
    return best, study.best_trial.user_attrs["rounds"], study


def rounds_from_cv(kind, t, feats, levels):
    """Rounds from the LAST CV fold only (train<=1999, val 2000-04): cheap and leakage-free."""
    tr, va = list(cv_splits(t))[-1]
    m = Wrap(kind).fit(prep(tr, feats, levels, "gbm" if kind == "xgboost" else "plain"), tr[TARGET],
                       prep(va, feats, levels, "gbm" if kind == "xgboost" else "plain"), va[TARGET])
    return m.best_iter


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(trials, frac=1.0):
    import mlflow
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("agrisense-yield")
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

    t = load_table(frac)
    feats = F.feature_columns(t)
    levels = cat_levels(t)
    ok = t.is_failure_outlier == 0
    train = t[(t.split == "train") & ok]
    valid, test = t[(t.split == "valid") & ok], t[(t.split == "test") & ok]
    log(f"rows train={len(train)} valid={len(valid)} test={len(test)}  features={len(feats)}")

    # --- tuning (LightGBM) and round counts (XGB / CatBoost)
    cache = C.REPORTS / "lgbm_tuning.json"
    if cache.exists() and frac == 1.0:
        tj = json.loads(cache.read_text())
        lgb_params, lgb_rounds = tj["params"], tj["rounds"]
        study = type("S", (), {"best_value": tj["cv_mae"]})()
        log("loaded cached LightGBM tuning:", lgb_params, "rounds", lgb_rounds, "cv_mae", round(tj["cv_mae"], 4))
    else:
        log(f"tuning LightGBM, {trials} trials x 3 expanding folds ...")
        lgb_params, lgb_rounds, study = tune_lgbm(t, feats, levels, trials)
        log("best lightgbm", lgb_params, "rounds", lgb_rounds, "cv_mae", round(study.best_value, 4))
        if frac == 1.0:
            cache.write_text(json.dumps({"params": lgb_params, "rounds": lgb_rounds, "cv_mae": study.best_value,
                                         "trials": trials}, indent=2))
    xgb_rounds = rounds_from_cv("xgboost", t, feats, levels)
    cb_rounds = rounds_from_cv("catboost", t, feats, levels)
    log("rounds xgb", xgb_rounds, "catboost", cb_rounds)

    specs = {"ridge": ("ridge", {}, None), "random_forest": ("rf", {}, None),
             "xgboost": ("xgboost", {}, int(xgb_rounds * 1.1)),
             "catboost": ("catboost", {}, int(cb_rounds * 1.1)),
             "lightgbm_tuned": ("lightgbm", lgb_params, int(lgb_rounds * 1.1))}

    results, fitted = {}, {}
    for name, (kind, params, rounds) in specs.items():
        t0 = time.time()
        m = Wrap(kind, params, rounds)
        kx = m.kindX
        m.fit(prep(train, feats, levels, kx), train[TARGET])
        fitted[name] = m
        res = {}
        for sp, d in [("valid", valid), ("test", test)]:
            p = m.predict(prep(d, feats, levels, kx))
            res[sp] = {"all": metrics(d, p), "weather_complete": metrics(d[d.weather_complete], p[d.weather_complete.to_numpy()])}
        results[name] = res
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()} | {"rounds": rounds})
            for sp in res:
                for k, v in res[sp]["weather_complete"].items():
                    mlflow.log_metric(f"{sp}_{k}", v)
        log(f"{name:15s} valid skill {res['valid']['weather_complete']['skill_MAE_pct']:+.1f}%  "
            f"test skill {res['test']['weather_complete']['skill_MAE_pct']:+.1f}%  ({time.time()-t0:.0f}s)")

    # --- selection on VALIDATION only
    best = max(results, key=lambda n: results[n]["valid"]["weather_complete"]["skill_MAE_pct"])
    log("selected on validation:", best)
    bm = fitted[best]

    # --- conformal intervals: signed residual quantiles per crop from VALIDATION residuals
    pv = bm.predict(prep(valid, feats, levels, bm.kindX))
    vres = valid.assign(res=valid[TARGET].to_numpy() - pv)
    conf = {c: [float(g.res.quantile(0.10)), float(g.res.quantile(0.90))] for c, g in vres.groupby("crop")}
    pt = bm.predict(prep(test, feats, levels, bm.kindX))
    lo = np.array([conf[c][0] for c in test.crop]); hi = np.array([conf[c][1] for c in test.crop])
    inside = (test[TARGET].to_numpy() >= pt + lo) & (test[TARGET].to_numpy() <= pt + hi)
    drought = (test.spi_monsoon < -1).to_numpy()
    wc = test.weather_complete.to_numpy()
    coverage = {"nominal": 0.80, "test_all": float(inside.mean()), "test_weather_complete": float(inside[wc].mean()),
                "test_drought_spi_lt_-1": float(inside[drought & wc].mean()), "n_drought": int((drought & wc).sum()),
                "test_normal": float(inside[~drought & wc].mean())}
    log("conformal coverage", coverage)

    # --- drought vs normal skill of selected model (train-only fit)
    dsk = {}
    for lab, mask in [("drought_spi_lt_-1", drought & wc), ("normal", ~drought & wc)]:
        dsk[lab] = metrics(test[mask], pt[mask])
    per_crop = {}
    for c in test.crop.unique():
        mk = ((test.crop == c) & test.weather_complete).to_numpy()
        per_crop[c] = metrics(test[mk], pt[mk])
    per_year = {}
    for y in sorted(test.year.unique()):
        mk = ((test.year == y) & test.weather_complete).to_numpy()
        per_year[int(y)] = metrics(test[mk], pt[mk])

    # --- refit final model on train+valid and score test once more
    full = pd.concat([train, valid])
    final = Wrap(specs[best][0], specs[best][1], int(specs[best][2] * 1.1))
    final.fit(prep(full, feats, levels, final.kindX), full[TARGET])
    pf = final.predict(prep(test, feats, levels, final.kindX))
    final_test = {"all": metrics(test, pf), "weather_complete": metrics(test[wc], pf[wc])}
    log("final (train+valid) test skill", round(final_test["weather_complete"]["skill_MAE_pct"], 1), "%")

    # --- save artifacts
    joblib.dump(final.model, MODELS_DIR / "final_model.joblib")
    meta = {"model": best, "kind": specs[best][0], "params": specs[best][1], "rounds": final.rounds,
            "features": feats, "categorical": F.CATEGORICAL, "levels": levels, "target": TARGET,
            "conformal_signed_residual_q10_q90": conf, "ref_period": C.REF_PERIOD,
            "train_years": [1966 + 0, C.TRAIN_END], "valid_years": list(C.VALID_YEARS),
            "test_years": list(C.TEST_YEARS), "final_trained_through": C.VALID_YEARS[1]}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    out = {"selected": best, "comparison": results, "final_refit_test": final_test, "coverage": coverage,
           "drought_vs_normal_test": dsk, "per_crop_test": per_crop, "per_year_test": per_year,
           "lightgbm_cv_best_mae": study.best_value, "lightgbm_trials": trials,
           "baseline_note": "baseline = mean of previous 5 years (y_roll5)"}
    (C.REPORTS / "model_results.json").write_text(json.dumps(out, indent=2))

    # --- predictions export (Power BI friendly)
    exp = test[["dist_code", "dist_name", "state_name", "crop", "year", "yield_kg_ha", "y_roll5",
                "spi_monsoon", "monsoon_ratio"]].copy()
    exp["pred_yield_final"] = test.y_roll5 * np.exp(pf)
    exp["pred_lo80"] = test.y_roll5 * np.exp(pf + lo); exp["pred_hi80"] = test.y_roll5 * np.exp(pf + hi)
    (C.ROOT / "exports").mkdir(exist_ok=True)
    exp.to_csv(C.ROOT / "exports" / "test_predictions.csv", index=False)

    figures(results, best, test, pt, pf, final, feats, levels, per_crop, wc)
    log("done")


# ----------------------------------------------------------------------------
# figures (incl. SHAP)
# ----------------------------------------------------------------------------
def figures(results, best, test, pt, pf, final, feats, levels, per_crop, wc):
    import shap
    sns.set_theme(style="whitegrid")
    # 1 model comparison
    rows = [{"model": n, "split": sp, "skill": r[sp]["weather_complete"]["skill_MAE_pct"]}
            for n, r in results.items() for sp in ["valid", "test"]]
    d = pd.DataFrame(rows)
    order = d[d.split == "valid"].sort_values("skill").model
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=d, y="model", x="skill", hue="split", order=order, ax=ax, palette=["#9db4c0", "#2a9d8f"])
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("MAE reduction vs 'mean of last 5 years' baseline (%)")
    ax.set_title("Model comparison (weather-complete rows)", loc="left", fontweight="bold"); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(FIG / "model_comparison.png", dpi=200); plt.close()

    # 2 skill by crop
    pc = pd.DataFrame(per_crop).T.sort_values("skill_MAE_pct")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(pc.index, pc.skill_MAE_pct, color=np.where(pc.skill_MAE_pct > 0, "#2a9d8f", "#d62828"))
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("MAE reduction vs baseline (%), test 2010-15")
    ax.set_title("Where the model helps (and where it doesn't)", loc="left", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "model_skill_by_crop.png", dpi=200); plt.close()

    # 3 predicted vs actual deviation
    d = test[wc]; p = pt[wc.to_numpy() if hasattr(wc, "to_numpy") else wc]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.hexbin(p, d[TARGET], gridsize=60, bins="log", cmap="viridis", extent=(-1, 1, -1, 1), mincnt=1)
    ax.plot([-1, 1], [-1, 1], "r--", lw=1); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("Predicted deviation from 5-yr norm (log)"); ax.set_ylabel("Actual deviation (log)")
    ax.set_title("Test set: do we predict the bad and good years?", loc="left", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "model_pred_vs_actual.png", dpi=200); plt.close()

    # 4 SHAP: each library's native exact tree contributions, plotted with shap
    samp = test.sample(min(6000, len(test)), random_state=1)
    X = prep(samp, feats, levels, final.kindX)
    if final.kind == "lightgbm":
        contrib = final.model.predict(X, pred_contrib=True)[:, :-1]
    elif final.kind == "xgboost":
        contrib = final.model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
    elif final.kind == "catboost":
        from catboost import Pool
        contrib = final.model.get_feature_importance(Pool(X, cat_features=F.CATEGORICAL), type="ShapValues")[:, :-1]
    else:
        return
    Xp = X.copy()
    for c in F.CATEGORICAL:                       # shap plots need numeric colouring
        Xp[c] = pd.Categorical(Xp[c]).codes
    for kind, name in [("bar", "shap_importance.png"), ("dot", "shap_beeswarm.png")]:
        plt.figure(figsize=(8, 7))
        shap.summary_plot(contrib, Xp, plot_type=kind, max_display=18, show=False)
        plt.title("What drives predicted yield deviation (SHAP)", loc="left", fontweight="bold")
        plt.tight_layout(); plt.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
    (C.REPORTS / "shap_importance.json").write_text(json.dumps(imp.round(5).to_dict(), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="10%% of data, 1 trial: code-path check only")
    a = ap.parse_args()
    main(1 if a.smoke else a.trials, 0.1 if a.smoke else 1.0)
