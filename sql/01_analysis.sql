-- AgriSense: first-pass SQL analysis on agrisense.db  (SQLite)
-- Each block starts with "-- name:" so the runner can execute them separately.

-- name: q1_coverage_by_state
SELECT d.state_name,
       COUNT(DISTINCT c.dist_code)  AS districts,
       MIN(c.year) AS first_year, MAX(c.year) AS last_year,
       COUNT(*)                     AS crop_observations
FROM crop_yield c JOIN districts d USING (dist_code)
WHERE c.is_aggregate = 0
GROUP BY d.state_name
ORDER BY crop_observations DESC;

-- name: q2_national_yield_by_decade_weighted
-- area-weighted yield (total production / total area), not a naive average of ratios
SELECT crop,
       (year/10)*10 AS decade,
       ROUND(SUM(production_1000t)*1000.0 / SUM(area_1000ha), 0) AS yield_kg_ha,
       ROUND(SUM(area_1000ha), 0) AS total_area_1000ha
FROM crop_yield
WHERE crop IN ('Rice','Wheat','Maize','Cotton') AND is_outlier = 0
GROUP BY crop, decade
ORDER BY crop, decade;

-- name: q3_top_wheat_districts_recent
-- districts with at least 8 valid years in 2008-2017
SELECT d.dist_name, d.state_name,
       COUNT(*) AS years,
       ROUND(AVG(c.yield_kg_ha), 0) AS avg_yield_kg_ha
FROM crop_yield c JOIN districts d USING (dist_code)
WHERE c.crop = 'Wheat' AND c.year BETWEEN 2008 AND 2017 AND c.is_outlier = 0
GROUP BY c.dist_code
HAVING years >= 8
ORDER BY avg_yield_kg_ha DESC
LIMIT 10;

-- name: q4_rainfall_anomaly_vs_yield_rice
-- Within-district comparison: monsoon rain relative to the district's own average,
-- and yield relative to the district's own average (after removing the long-run
-- trend by comparing to a +/-5 year rolling mean).
WITH base AS (
  SELECT dist_code, year, yield_kg_ha, monsoon_mm
  FROM v_crop_rain
  WHERE crop = 'Rice' AND is_outlier = 0 AND monsoon_mm IS NOT NULL
),
rel AS (
  SELECT dist_code, year,
         yield_kg_ha * 1.0 / AVG(yield_kg_ha) OVER w AS yield_ratio,
         monsoon_mm   * 1.0 / AVG(monsoon_mm)   OVER (PARTITION BY dist_code) AS rain_ratio
  FROM base
  WINDOW w AS (PARTITION BY dist_code ORDER BY year ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING)
),
banded AS (
  SELECT *, CASE WHEN rain_ratio < 0.70 THEN '1 severe deficit (<70%)'
                 WHEN rain_ratio < 0.85 THEN '2 deficit (70-85%)'
                 WHEN rain_ratio < 1.15 THEN '3 normal (85-115%)'
                 WHEN rain_ratio < 1.30 THEN '4 surplus (115-130%)'
                 ELSE '5 heavy surplus (>130%)' END AS rain_band
  FROM rel
)
SELECT rain_band, COUNT(*) AS district_years,
       ROUND(AVG(yield_ratio), 3) AS avg_yield_vs_local_trend
FROM banded GROUP BY rain_band ORDER BY rain_band;

-- name: q5_most_volatile_rice_districts
-- coefficient of variation of detrended yield ratio; min 25 years
WITH r AS (
  SELECT dist_code, year,
         yield_kg_ha * 1.0 / AVG(yield_kg_ha) OVER (PARTITION BY dist_code ORDER BY year
              ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) AS ratio
  FROM crop_yield WHERE crop = 'Rice' AND is_outlier = 0
),
s AS (
  SELECT dist_code, COUNT(*) AS n, AVG(ratio) AS m,
         AVG(ratio*ratio) - AVG(ratio)*AVG(ratio) AS var
  FROM r GROUP BY dist_code HAVING n >= 25
)
SELECT d.dist_name, d.state_name, n AS years,
       ROUND(SQRT(var)/m, 3) AS detrended_cv
FROM s JOIN districts d USING (dist_code)
ORDER BY detrended_cv DESC LIMIT 10;
