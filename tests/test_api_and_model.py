"""Serving tests: train/serve consistency, intervals, API contract."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import features as F
from src.predict import get_predictor

pytestmark = pytest.mark.skipif(not (C.ROOT / "models" / "final_model.joblib").exists(),
                                reason="run `python -m src.train` first")


@pytest.fixture(scope="module")
def P():
    return get_predictor()


@pytest.fixture(scope="module")
def table():
    return pd.read_parquet(C.PROCESSED / "model_table.parquet")


def test_serving_features_match_training_table(P, table):
    """No train/serve skew: rebuilding a row on the fly gives the same features as the table."""
    t = table[(table.year == 2012) & table.y_roll5.notna() & table.monsoon_ratio.notna()].sample(8, random_state=3)
    feats = [f for f in P.meta["features"] if f not in F.CATEGORICAL]
    for r in t.itertuples():
        row = P.build_row(r.dist_code, r.crop, r.year)
        for f in feats:
            a, b = getattr(r, f), row[f].iloc[0]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b, rel=1e-6), f


def test_prediction_is_close_to_baseline_and_interval_is_ordered(P):
    out = P.predict(239, "Rice", 2012) if 239 in P.available.dist_code.values else None
    if out is None:
        d = P.available.iloc[0]; out = P.predict(int(d.dist_code), d.crop, 2012)
    assert out["lo80_kg_ha"] < out["predicted_kg_ha"] < out["hi80_kg_ha"]
    assert 0.3 < out["predicted_kg_ha"] / out["baseline_kg_ha"] < 3


def test_drought_whatif_lowers_prediction_for_a_rainfed_crop(P):
    rows = P.available[P.available.crop == "Pearl Millet"].head(40)
    deltas = []
    for r in rows.itertuples():
        try:
            lo = P.predict(r.dist_code, "Pearl Millet", 2012, 0.5)["predicted_kg_ha"]
            hi = P.predict(r.dist_code, "Pearl Millet", 2012, 1.2)["predicted_kg_ha"]
            deltas.append(hi - lo)
        except ValueError:
            continue
    assert len(deltas) > 10 and np.mean(np.array(deltas) > 0) > 0.7


def test_conformal_coverage_roughly_nominal_on_test():
    import json
    cov = json.loads((C.REPORTS / "model_results.json").read_text())["coverage"]
    assert 0.65 < cov["test_weather_complete"] < 0.92


def test_api_contract(P):
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    d = c.get("/districts", params={"state": "Maharashtra"}).json()
    assert len(d) > 10
    code = d[0]["dist_code"]
    crops = c.get(f"/crops/{code}").json()["crops"]
    r = c.post("/predict", json={"dist_code": code, "crop": crops[0], "year": 2012})
    assert r.status_code == 200 and r.json()["predicted_kg_ha"] > 0
    assert c.post("/predict", json={"dist_code": code, "crop": "Nonexistent", "year": 2012}).status_code == 404
    assert c.post("/predict", json={"dist_code": code, "crop": crops[0], "year": 1800}).status_code == 422
    rk = c.get("/risk", params={"dist_code": code, "crop": crops[0], "scenario": "drought"})
    assert rk.status_code in (200, 404)


def test_model_inputs_are_exactly_the_documented_feature_set(P, table):
    """Guards against helper columns silently becoming model inputs."""
    assert list(P.meta["features"]) == F.feature_columns(table)
    assert "weather_complete" not in P.meta["features"]


def test_drought_scenario_is_worse_than_normal_on_average():
    from src import risk
    r = risk.get_ratios()
    sel = r[r.crop == "Pearl Millet"].dist_code.drop_duplicates().head(40)
    worse = 0; n = 0
    for dc in sel:
        try:
            a = risk.simulate(r, int(dc), "Pearl Millet", "normal")["p50"]
            b = risk.simulate(r, int(dc), "Pearl Millet", "drought")["p50"]
        except ValueError:
            continue
        n += 1; worse += b < a
    assert n > 15 and worse / n > 0.8, (worse, n)
