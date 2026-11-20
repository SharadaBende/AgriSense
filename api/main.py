"""AgriSense prediction API.   Run:  uvicorn api.main:app --reload"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import risk
from src.predict import get_predictor, spi_label

app = FastAPI(title="AgriSense API", version="1.0",
              description="District-level crop yield prediction and risk analysis (ICRISAT data, 1966-2017).")


class PredictRequest(BaseModel):
    dist_code: int = Field(..., examples=[239])
    crop: str = Field(..., examples=["Rice"])
    year: int = Field(..., ge=1975, le=2015)
    monsoon_factor: float = Field(1.0, ge=0.3, le=2.0, description="Multiplier on Jun-Sep rainfall of that year")


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "model": p.meta["model"], "trained_through": p.trained_through}


@app.get("/districts")
def districts(state: str | None = None):
    p = get_predictor()
    d = p.dist[p.dist.dist_code.isin(p.available.dist_code)]
    if state:
        d = d[d.state_name.str.lower() == state.lower()]
    return d[["dist_code", "dist_name", "state_name"]].sort_values(["state_name", "dist_name"]).to_dict("records")


@app.get("/crops/{dist_code}")
def crops(dist_code: int):
    c = get_predictor().crops_for(dist_code)
    if not c:
        raise HTTPException(404, "district not found or insufficient history")
    return {"dist_code": dist_code, "crops": c}


@app.post("/predict")
def predict(req: PredictRequest):
    p = get_predictor()
    if req.crop not in p.crops_for(req.dist_code):
        raise HTTPException(404, "no usable history for this district and crop")
    try:
        out = p.predict(req.dist_code, req.crop, req.year, req.monsoon_factor)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["monsoon_description"] = spi_label(out["spi_monsoon"])
    return out


@app.get("/risk")
def risk_profile(dist_code: int, crop: str, scenario: str = Query("normal", pattern="^(normal|drought)$"),
                 n: int = Query(5000, ge=500, le=50000)):
    try:
        sim = risk.simulate(risk.get_ratios(), dist_code, crop, scenario=scenario, n=n)
    except ValueError as e:
        raise HTTPException(404, str(e))
    sim.pop("samples")
    return sim
