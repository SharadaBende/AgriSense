# Interview guide: how to talk about AgriSense

The project only helps you if you can explain and defend it. Read this, then re-read the code for each answer.
Every number below is from this repo's result files (`reports/model_results.json`, `reports/ablation.json`,
`reports/risk_summary.json`).

## 30-second pitch
"I built an end-to-end system on the ICRISAT district database, 311 districts over 51 years, that predicts crop yield
from weather and yield history, gives uncertainty intervals, and scores district risk. The main thing I focused on was
avoiding leakage in panel data: strictly time-based splits and tests that corrupt future data to prove past features
don't change. The model beats a strong 5-year-average baseline by 4.7% MAE on unseen 2010-15 data, which is modest, and I
show why: no irrigation data, and yield history already captures most of what is predictable."

## Likely questions and honest answers

**1. Why not a random train/test split?**
Rows from the same district are strongly correlated and yields trend upward. A random split lets the model see 2012 while
predicting 2011 and gives inflated scores. I split by time (train ≤2004, valid 2005-09, test 2010-15).

**2. How do you know there is no leakage?**
Features for year T use only rainfall ≤T and the district's own yields <T; the climate reference (anomalies, SPI) is fixed
to 1966-89. Tests multiply all yields after 1999 by 10 and rainfall after 1995 by 3 and assert earlier features are
identical. I also injected a deliberate leak (removed the one-year shift) and 3 tests failed, which showed the tests work.
Honest caveat: outlier flags use full-series medians; they are cleaning labels, not features.

**3. R² is 0.90: why do you say that is misleading?**
Most variance is between districts (Punjab vs Rajasthan). The useful question is how much of the *deviation from a
district's normal* the model explains: about 11%.

**4. Why predict log(yield / 5-year mean) instead of yield?**
Tree models cannot extrapolate a trend and spend capacity on between-district differences. The ratio target isolates
what weather and recent history can explain; I convert back with `yield = baseline × exp(prediction)`.

**5. Only +4.7% over baseline. Is that useful?**
It is real and consistent (all 13 crops positive, 5 of 6 test years clearly positive, the model chosen on validation was also best
on test), but modest. Year-to-year variation is dominated by irrigation, inputs and management that are not in the data.
In 2014, a drought year, skill was -0.1%, so I do not claim it predicts droughts well.

**6. Does weather matter?**
Ablation: with weather 4.2% vs without 1.8% on test, but on validation 4.3% vs 4.2%. So weather helps, but the size
is uncertain; one split can't resolve it. In SHAP, yield history is ~60% of attribution, weather ~25%.

**7. Why LightGBM? How did you tune?**
Compared Ridge, RF, XGBoost, CatBoost, LightGBM. Optuna, 12 trials, objective = MAE averaged over 3 expanding-window time folds
inside the training years; boosting rounds from CV; selection on validation; test reported, not used for choices.
Honest: only LightGBM was tuned; the others used reasonable defaults, so the gaps partly reflect tuning effort.

**8. What is SPI and why a fixed reference period?**
Standardized Precipitation Index: fit a gamma distribution to rainfall in a reference period, convert to a z-score.
Fixing the reference to 1966-89 means no CV, validation or test year influences the fit. Downside: 24 years is shorter than the
30 recommended.

**9. How are the prediction intervals made? Are they good?**
Split conformal: signed residual 10th/90th percentiles per crop from validation residuals. Actual test coverage is 75%
for a nominal 80% (73% in drought years): slightly overconfident, because residual spread differs between periods.

**10. What does the what-if slider do? Is it causal?**
It multiplies Jun-Sep rainfall of the chosen year and recomputes all monsoon-derived features with the same code used in
training. It shows the model's learned association, not a causal effect; irrigated districts barely respond (Ludhiana wheat),
rain-fed ones do (Solapur sorghum, Jodhpur millet).

**11. How did you define risk? Are the tiers meaningful?**
Detrended volatility, share of years >15% below trend, expected shortfall in the worst 10% of years, drought-year loss;
K-Means on district means → 101 High / 163 Medium / 45 Low. Silhouette is 0.48, so tiers segment a continuum rather than reveal
natural clusters. Hill and tiny districts can look risky because of noisy records.

**12. A bug you found yourself?** (strong stories: pick two)
- A helper column (`weather_complete`) leaked into the feature list; tests exposed it, I fixed it and retrained from scratch.
- My drought Monte Carlo produced a *higher* median than normal for one district (6 drought years); I replaced it with partial
  pooling toward state-level drought years and added a test.
- A chart silently clipped the driest year (2002); I caught it by looking at the figure.

**13. Data quality issues?**
`-1` missing codes; 22 districts with no rainfall; rainfall missing for ~87 districts in 2010-11; 1966 vs modern district
boundaries; high-side outliers up to 68× the district norm. I documented each and reported results on rows with rainfall.

**14. What would you do next?**
Add irrigation and soil data, extend weather past 2015, use the 1997-2020 season-level government data, model seasons
separately, try spatial/hierarchical models, and recalibrate intervals under drought.

## Things NOT to say
- "99% accuracy" or any accuracy claim that isn't skill versus a baseline.
- "It predicts droughts" (it doesn't, per the 2014 result).
- "The slider shows the effect of rainfall on yield" (association only).
- "I built a live forecasting system" (data end in 2015).

## Resume bullets (accurate as written)
- Built an end-to-end crop-yield prediction and agricultural-risk analytics system on the ICRISAT district database
  (311 districts, 1966-2017, ~228k district-crop-year observations) with a SQLite pipeline, 41 leakage-safe features,
  and 30 automated tests.
- Designed time-based validation and leakage tests (corrupt-the-future checks); tuned LightGBM with Optuna and
  expanding-window CV, reducing MAE 4.7% vs a 5-year-average baseline on unseen 2010-15 data; explained drivers with SHAP and
  quantified uncertainty with conformal intervals.
- Delivered district risk tiers (K-Means), drought-scenario Monte Carlo, a Streamlit app, a FastAPI service and Power
  BI-ready exports.
