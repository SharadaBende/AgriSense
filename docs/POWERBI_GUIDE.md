# Power BI dashboard guide (build time: ~60-90 min)

I cannot generate a `.pbix` file, so this is a precise build spec. All inputs are in `exports/`.

## 1. Load data (Home > Get data > Text/CSV)
| Table | File | Grain |
|---|---|---|
| YieldHistory | `exports/yield_history.csv` | district x crop x year (1966-2015), with rainfall and anomalies |
| DistrictCropRisk | `exports/district_crop_risk.csv` | district x crop risk metrics |
| DistrictRisk | `exports/district_risk_tiers.csv` | district K-Means tier and score |
| FailureEvents | `exports/crop_failure_events.csv` | yield < 60% of local trend |
| TestPredictions | `exports/test_predictions.csv` | model predictions vs actuals, 2010-15 |

Relationships (many-to-one): `YieldHistory[dist_code]`, `DistrictCropRisk[dist_code]`, `FailureEvents[dist_code]`,
`TestPredictions[dist_code]` -> `DistrictRisk[dist_code]`.

## 2. Pages
1. **Overview** - cards: districts, crops, years; line chart of average yield by year (legend = crop);
   bar chart of risk tier counts; slicers: State, Crop group.
2. **Yield trends** - small multiples (crop) of yield by year; state x decade matrix with conditional formatting.
3. **Rainfall & drought** - scatter `monsoon_ratio` vs `yield_ratio`; column chart of failure events per year (highlight 1972, 1979, 1987, 2002, 2009, 2014, 2015).
4. **Risk** - filled map (Latitude/Longitude from DistrictRisk, or Shape map with the GeoJSON) coloured by `risk_tier`; table of top 10 districts by `risk_score`; scatter `cv` vs `downside`.
5. **Model performance** - clustered columns `yield_kg_ha` vs `pred_yield_final` by year; error by crop; coverage KPI.

## 3. Useful DAX measures
```
Avg Yield = AVERAGE(YieldHistory[yield_kg_ha])
Weighted Yield = DIVIDE(SUM(YieldHistory[production_1000t]) * 1000, SUM(YieldHistory[area_1000ha]))
Failure Events = COUNTROWS(FailureEvents)
MAE = AVERAGEX(TestPredictions, ABS(TestPredictions[pred_yield_final] - TestPredictions[yield_kg_ha]))
Baseline MAE = AVERAGEX(TestPredictions, ABS(TestPredictions[y_roll5] - TestPredictions[yield_kg_ha]))
Skill % = 1 - DIVIDE([MAE], [Baseline MAE])
Interval Coverage = DIVIDE(
    COUNTROWS(FILTER(TestPredictions, TestPredictions[yield_kg_ha] >= TestPredictions[pred_lo80] && TestPredictions[yield_kg_ha] <= TestPredictions[pred_hi80])),
    COUNTROWS(TestPredictions))
```

## 4. Honest-labelling checklist
Put a text box on every page: "Source: ICRISAT District-Level Database 1966-2017. District boundaries are 1966-based.
Risk tiers are descriptive (historical), not forecasts."
