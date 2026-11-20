# AgriSense: Project Report

## 1. Problem
Indian crop yields swing year to year with monsoon rainfall. Insurers, planners and lenders need (a) a realistic
expectation of yield and its uncertainty, and (b) a map of where yield risk is structurally high. AgriSense builds both
from public district-level data and is explicit about how much the data can and cannot support.

## 2. Data
ICRISAT District-Level Database: crop area, production and yield for ~20 crops in 311 districts (20 states), 1966-2017
(16,146 district-years; ~228k valid district-crop-year yield observations), plus monthly district rainfall
(313 districts, 1966-2015). Cleaning: missing code `-1` → NaN; wide → long; yield recomputed as production/area
(reported yield agreed within 1% for 99.9% of rice rows); implausible high-side outliers dropped, low-side ones kept
as possible genuine failures; aggregate crops flagged. Integrity: two independent copies of the crop file were
byte-identical; the 8 driest monsoons in the rainfall data include 7 known drought years.

## 3. Exploratory findings
Yield growth slowed after 1990 for wheat (+3.3%/yr → +1.4%) and rice (+2.5% → +1.6%) but accelerated for maize and cotton.
Rain-fed crops lose 7-14% of yield versus local trend when monsoon rain is under 70% of normal, while surplus rain adds
little (0-6%), so the response to rainfall is strongly asymmetric. Millet area shrank from 24.5% to 8.9% of recorded area.

## 4. Modelling
*Target:* deviation from the local 5-year norm, `log(yield / mean of previous 5 yields)`.
*Features (41):* own yield history (lags, rolling mean/std/max/min, 10-year local trend), sown area, seasonal rainfall
anomalies, SPI (gamma fit on a fixed 1966-89 reference period) for annual, monsoon, early and late monsoon windows,
deficit-month counts, previous-year weather, location, crop, state.
*Validation:* strictly time-based (train ≤ 2004, validation 2005-09, test 2010-15) with expanding-window CV inside
training for tuning. Tests corrupt all future data and require past features to be unchanged; a deliberately injected
leak was caught by the tests.
*Models:* Ridge, Random Forest, XGBoost, CatBoost, LightGBM (Optuna, 12 trials). Chosen on validation only.
*Uncertainty:* split-conformal 80% intervals per crop.

## 5. Results
On the untouched test years the tuned LightGBM reduced MAE from 335 to 319 kg/ha (-4.7%) versus the 5-year-average
baseline and explained ~11% of year-to-year deviation. Weather roughly doubled skill on the test years (1.8% → 4.2%)
but added little on the validation years, so its contribution is positive but imprecisely measured. All 13 crops beat
the baseline. Intervals covered 75% of outcomes (nominal 80%). In the 2014 drought the model was no better than the
baseline, and it was not better in drought years overall (4.0% vs 4.2%).

## 6. Risk analysis
Per district-crop: detrended volatility, frequency of years >15% below trend, expected shortfall in the worst decile,
drought-year loss. K-Means on district-level metrics gave 101 High / 163 Medium / 45 Low tiers (silhouette 0.48: a
segmentation of a continuum). A bootstrap Monte Carlo simulates next-season yield and revenue; the drought scenario
partially pools a district's few drought years with its state's (weight n/(n+10)) because a first version using
district-only samples produced a drought median *above* the normal median.

## 7. Limitations
No irrigation, soil or price data; rainfall ends 2015 (predictions are historical replays, not live forecasts); the what-if
is associational; weather gaps (22 districts, ~87 districts in 2010-11); 1966 district boundaries approximated on maps;
cleaning flags use full-series medians; data are third-party copies of public files.

## 8. Next steps
Add irrigation and soil layers (ICRISAT DLD irrigation tables, soil maps); extend weather beyond 2015 (NASA POWER / IMD
gridded rainfall) and use the 1997-2020 season-wise government data; model seasons separately; hierarchical or spatial
models that borrow strength across neighbouring districts; calibrate intervals under drought (currently 73%).
