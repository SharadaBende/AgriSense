# AgriSense - Data Sources & Provenance

| File | What it is | Rows | Coverage | Original publisher | Retrieved from |
|---|---|---|---|---|---|
| icrisat_crop_area_production_yield.csv | District crop area, production, yield (~20 crops) | 16,146 | 311 districts, 20 states, 1966-2017 | ICRISAT / TCI District-Level Database (data.icrisat.org/dld) | GitHub: ma5755-capstone-project; identical copy in GitHub: AgriData-Explorer (byte-for-byte equal) |
| icrisat_monthly_rainfall.csv | Monthly + annual district rainfall (mm) | 14,527 | 313 districts, 1966-2015 | ICRISAT DLD | GitHub: ma5755-capstone-project |
| icrisat_normal_rainfall_30yr.csv | 30-year normal monthly rainfall per district | 313 | - | ICRISAT DLD | GitHub: ma5755-capstone-project |
| india_district_boundaries.geojson | District polygons for maps (GADM-style, 594 features) | 594 | - | GADM (check licence: free for non-commercial use) | GitHub: ma5755-capstone-project |

## Caveats (state these in the README)
- Files are third-party GitHub copies of public data; not yet re-verified against the official portal.
- ICRISAT uses fixed 1966-based district boundaries (311 districts); GeoJSON has modern boundaries -> name matching needed for maps.
- Rainfall ends 2015, so weather-driven modeling covers 1966-2015.
- ICRISAT uses -1 as a missing-value code. Must be converted to NaN.
- Soil and irrigation data were not sourced (a real gap: rainfall effects are muted in irrigated districts).
- `icrisat_normal_rainfall_30yr.csv` is cleaned but deliberately NOT used as a feature: its reference period is not
  documented, so it could include validation/test years. Anomalies use a fixed 1966-89 reference computed in `src/features.py`.
- The 1997-2020 season-wise government dataset I located was not used and is not included.