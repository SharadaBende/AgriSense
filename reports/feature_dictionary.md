# AgriSense - Feature Dictionary (model_table.parquet)

One row = one district x crop x year (1966-2015). 150,556 rows, 13 crops, 311 districts.
Target: `yield_kg_ha` (modelled in Phase 5 as `target_logratio = log(yield / y_roll5)`).

**Leakage rule:** every feature for year T uses only (a) rainfall of year <= T, (b) the same
district-crop's yield/area of years < T, (c) a reference climatology fitted on 1966-1989 only,
(d) static location facts. Verified by tests that corrupt all future data and require identical past features.

## Yield history (own past only)
| Feature | Meaning |
|---|---|
| y_lag1, y_lag2, y_lag3 | Yield 1, 2, 3 calendar years earlier |
| y_roll3, y_roll5 | Mean of previous 3 / 5 years (min 2 / 3 valid) |
| y_roll8_std, y_cv8 | Std and coefficient of variation of previous 8 years (volatility) |
| y_max5, y_min5 | Best / worst yield of previous 5 years |
| n_hist10 | Number of valid yields in previous 10 years (data quality signal) |
| y_trend_slope | Log-linear yield trend over previous 10 years (clipped to +/-8%/yr) |
| y_trend_pred | Extrapolated trend value for year T |
| area_1000ha | Sown area in year T (known at sowing) |
| area_roll5, area_ratio | Mean area of previous 5 years; current area relative to it |

## Weather (year T and earlier; anomalies relative to 1966-1989 reference means)
| Feature | Meaning |
|---|---|
| annual_ratio, monsoon_ratio, pre_ratio, winter_ratio, post_ratio | Seasonal rain / reference mean (IMD seasons) |
| spi_annual, spi_monsoon, spi_early (Jun-Jul), spi_late (Aug-Sep) | Standardised Precipitation Index (gamma fit on reference period) |
| jun_ratio ... sep_ratio | Monthly monsoon rain vs reference |
| n_deficit_months | Number of Jun-Sep months below 50% of normal |
| jul_aug_share | Share of monsoon rain falling in Jul-Aug |
| monsoon_ratio_prev, spi_monsoon_prev, annual_ratio_prev | Previous year's weather (carry-over moisture) |
| monsoon_ratio_2yr | Mean of this and last monsoon (drought persistence) |
| clim_annual_mm, clim_monsoon_mm | Reference-period mean rainfall (aridity of the district) |

## Static
| Feature | Meaning |
|---|---|
| lat, lon | Representative point of the district polygon |
| state_name, crop, season_type | Categorical (kharif / rabi / annual) |

## Not features (kept for modelling / evaluation)
`yield_kg_ha`, `log_yield`, `target_logratio`, `split` (train <=2004, valid 2005-09, test 2010-15),
`is_failure_outlier` (low-side outliers kept for risk analysis), `dist_code`, `dist_name`.

## Known limitations
- Outlier flags use full-series medians (cleaning step, not features): minor, disclosed.
- 22 districts have no rainfall record; 2010 and 2011 miss rainfall for ~87 districts.
- No irrigation or soil data: rainfall effects are partly masked in irrigated districts.
- Reference period 1966-1989 is only 24 years (WMO suggests 30) - chosen to keep every CV fold leakage-free.
