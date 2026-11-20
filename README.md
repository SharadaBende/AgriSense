# 🌾 AgriSense: Crop Yield Prediction & Agricultural Risk Analysis

District-level crop yield prediction, drought analysis and risk scoring for India, built on the
**ICRISAT District-Level Database** (311 districts, 20 states, 1966-2017, ~228,000 district-crop-year yield
observations) with monthly district rainfall.

**What it does**
1. Predicts a district's crop yield from weather, yield history and location, with **80% prediction intervals**.
2. Lets you ask **what-if** questions: *what would the model expect if the monsoon had been 30% weaker?*
3. Scores **agricultural risk** per district and crop (volatility, downside frequency, tail loss, drought sensitivity)
   and groups districts into Low / Medium / High tiers.
4. Runs a **Monte Carlo** simulation of next-season yield and revenue under normal and drought scenarios.
5. Ships as a **Streamlit app**, a **FastAPI** service, and **Power BI-ready exports**.

> **Headline result, stated honestly.** On 2010-2015 data the model never saw, it cuts mean absolute error by
> **4.7%** versus a strong naive baseline (the district's previous 5-year average): 319 vs 335 kg/ha. That is a modest
> gain. Year-to-year yield swings are driven largely by irrigation, inputs and management, none of which are in
> this dataset. See [Results](#results) and [Limitations](#limitations).

---

## Results

### Prediction (test = 2010-2015, never used for any choice; 14,203 rows with rainfall records)
| | Value |
|---|---|
| Baseline MAE (mean of previous 5 years) | 335 kg/ha |
| **Model MAE (LightGBM, tuned)** | **319 kg/ha (-4.7%)** |
| Median absolute % error | 14.3% |
| Share of year-to-year *deviation* explained (R²) | 0.11 |
| 80% interval actual coverage | 75% (nominal 80%) |

R² on raw yield is 0.90 but is **misleading**: most of it is districts simply differing from each other. The honest
metric is how much of the *deviation from a district's normal* is explained: about 11%.

**Model comparison** (skill = MAE reduction vs baseline; model chosen on validation 2005-09 only):

| Model | Validation | Test |
|---|---|---|
| Ridge | -1.2% | +1.4% |
| Random Forest | +1.2% | +2.2% |
| XGBoost | +3.0% | +2.4% |
| CatBoost | +3.4% | +3.2% |
| **LightGBM (Optuna, 12 trials, 3 time-based CV folds)** | **+4.3%** | **+4.2%** |

The final model is refit on train+valid (test skill +4.7%).

**Does weather help?** (ablation, same model, test 2010-15 / validation 2005-09)
| Inputs | Test skill | Validation skill |
|---|---|---|
| Yield history + location, no weather | +1.8% | +4.2% |
| Weather + location only | +0.4% | -0.1% |
| **Both** | **+4.2%** | **+4.3%** |

Weather more than doubles skill on the test years but added almost nothing on the validation years, so its
exact contribution is **uncertain**. In SHAP terms, yield history accounts for ~60% of attribution, weather ~25%,
crop/state/location ~15%.

**Where it works:** all 13 crops beat the baseline, most for barley (+9.6%), cotton (+8.1%), wheat (+8.0%), chickpea (+7.3%);
least for rice (+2.0%), sesamum (+2.0%) and sugarcane (+1.1%).
**Where it doesn't:** in **2014** (a drought year) skill was **-0.1%**, and on drought rows overall (SPI < -1) skill was
4.0% vs 4.2% on normal rows, so it is *not* better at catching droughts than normal years. The 80% intervals are slightly
overconfident (75%; 73% in drought years).

### Exploratory findings (see `reports/figures/`)
- **Green Revolution slowdown:** wheat yield grew +3.3%/yr before 1990 and +1.4%/yr after; rice +2.5% → +1.6%;
  maize and cotton went the other way (+1.5% → +3.0%, +2.4% → +3.0%).
- **Drought hurts more than rain helps:** when monsoon rain is below 70% of a district's own normal, yields fall
  ~14% (pearl millet), 12% (kharif sorghum), 11% (rice, groundnut), 8% (cotton), 7% (maize) versus local trend;
  surplus rain adds only 0-6%, and heavy surplus hurts maize and sorghum.
- **Data sanity check:** 7 of the 8 driest monsoons in the data are well-known droughts (1972, 1979, 1987, 2002, 2009, 2014, 2015).
- **Crop mix:** millets fell from 24.5% to 8.9% of recorded area; oilseeds rose from 9.1% to 15.4%.

### Risk analysis (descriptive, historical)
- 4,266 district-crop pairs scored; 309 districts clustered with K-Means into **101 High / 163 Medium / 45 Low** risk.
  High risk concentrates in rain-fed Rajasthan, Gujarat, the Deccan plateau and parts of Jharkhand; low risk in Assam, Kerala, coastal Odisha.
- **Caveat:** cluster separation is modest (silhouette 0.48 at k=3, 0.49 at k=2): the tiers segment a continuum, they
  are not natural groups. Hill and tiny districts can rank "risky" partly because their records are noisy.
- 13,933 "crop failure" observations (yield < 60% of local trend); 24% fall in the 7 known drought years (13.5% of all
  years) and 32% have SPI < -1. Drought explains part, not all, of them; some are reporting problems.

---

## Quick start

```bash
pip install -r requirements.txt          # app + API only
streamlit run app/app.py                 # http://localhost:8501
uvicorn api.main:app --reload            # http://localhost:8000/docs
```

Full pipeline (data → features → EDA → train → risk → tests), about 12 minutes on one CPU core:
```bash
pip install -r requirements-dev.txt
make all          # or run the steps below individually
python -m src.make_dataset      # clean raw ICRISAT files -> SQLite + tidy tables
python -m src.features          # leakage-safe feature table
python -m src.eda               # figures
python -m src.train --trials 12 # models, SHAP, conformal intervals, MLflow logging
python -m src.ablation          # weather vs history
python -m src.risk              # risk tiers, failure events, exports
python -m src.exports           # Power BI CSVs
python -m pytest tests -q
```
Experiments are logged to a local MLflow store (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

### API
`GET /health` · `GET /districts?state=` · `GET /crops/{dist_code}` · `POST /predict` · `GET /risk?dist_code=&crop=&scenario=normal|drought`
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"dist_code": 1, "crop": "Rice", "year": 2012, "monsoon_factor": 0.7}'
```
(Find real `dist_code` values with `GET /districts`.)

## How it is built

```
raw ICRISAT CSVs ─► make_dataset ─► SQLite + tidy tables ─► features ─► model_table.parquet
                                         │                       │
                                   sql/01_analysis.sql      train.py (Ridge/RF/XGB/CatBoost/LightGBM+Optuna)
                                         │                       │  SHAP · conformal intervals · MLflow
                                      eda.py                 predict.py ─► Streamlit app + FastAPI
                                                                 risk.py ─► tiers · failures · Monte Carlo
```

**Leakage control (the part most student projects get wrong).**
- Time-based split: train ≤ 2004, validation 2005-09, test 2010-15. Never random: panel data with a strong trend.
- Every feature for year *T* uses only rainfall up to *T*, the same district-crop's own yields before *T*, and a fixed
  1966-89 reference climate (so no validation/test year ever touches an anomaly or SPI).
- Tests **corrupt all future data** (×10 yields after 1999, ×3 rainfall after 1995) and require past features to be
  unchanged. I also injected a deliberate leak to confirm the tests catch it.
- Boosting rounds come from expanding-window CV inside the training years; model selection uses validation only.
- Modelling target is the *deviation from the local 5-year norm* (`log(yield / mean of last 5 years)`), which lets tree
  models focus on what weather can explain rather than on the long-run trend they cannot extrapolate.

**Data cleaning decisions:** `-1` is ICRISAT's missing code → NaN; wide → long (one row per district-crop-year);
yield recomputed as production ÷ area; high-side outliers (likely errors) dropped, low-side ones kept as potential real
failures but hidden from history features; aggregate crops flagged to avoid double counting.

## Repository layout
```
src/        config · make_dataset · geo · features · eda · train · ablation · risk · predict · exports
app/        Streamlit app (4 pages)           api/   FastAPI service
sql/        analysis queries                  tests/ 30 tests (data integrity, leakage, serving, API, app)
models/     final model + metadata            exports/ Power BI-ready CSVs
reports/    figures, result JSON, feature dictionary
docs/       PROJECT_REPORT · INTERVIEW_GUIDE · POWERBI_GUIDE · DEPLOY
```
Feature definitions: `reports/feature_dictionary.md`. Data provenance: `DATA_SOURCES.md`.

## Deploy
See [`docs/DEPLOY.md`](docs/DEPLOY.md) (GitHub → Streamlit Community Cloud, Hugging Face Spaces, Docker).

## Limitations
- **No irrigation, soil or price data.** Rainfall effects are muted where fields are irrigated (e.g., Punjab wheat barely
  responds to rainfall). Prices in the Monte Carlo are supplied by the user.
- **Historical, not live.** Rainfall ends in 2015, so predictions replay seasons up to 2015; this is not an operational forecaster.
  The what-if is an *association* learned by the model, not a causal simulation.
- **Modest skill.** +4.7% MAE vs a strong baseline; ~11% of year-to-year deviation explained; no better in drought years.
- **Weather data gaps:** 22 districts have no rainfall record, and rainfall is missing for ~87 districts in 2010-11.
  Headline numbers use rows with rainfall; all-row numbers are almost identical (+4.7% / `reports/model_results.json`).
- **Boundaries:** ICRISAT uses fixed 1966 districts; maps approximate split districts using modern polygons (256 exact name
  matches, 41 reviewed fuzzy matches, 16 manual merges).
- **Cleaning flags use full-series medians** (a minor, disclosed limitation; they are not model features).
- Data are third-party GitHub copies of public ICRISAT files (two independent copies were byte-identical), not re-verified
  against the official portal. District boundary polygons are GADM (free for non-commercial use).

## Credits
ICRISAT / TCI District-Level Database for Indian agriculture · GADM administrative boundaries.
Code: MIT (see LICENSE); data licences differ.
