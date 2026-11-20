"""Central configuration for AgriSense: paths and crop metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DB_PATH = PROCESSED / "agrisense.db"

CROP_FILE = RAW / "icrisat_crop_area_production_yield.csv"
RAIN_FILE = RAW / "icrisat_monthly_rainfall.csv"
NORMAL_FILE = RAW / "icrisat_normal_rainfall_30yr.csv"

# ICRISAT encodes "missing" as -1
MISSING_CODE = -1

# crop -> (group, is_aggregate)
# Aggregates are totals/sub-series of other crops. They are kept in the data
# but flagged, so that summing area across crops never double counts.
CROP_META = {
    "RICE": ("cereal", False),
    "WHEAT": ("cereal", False),
    "MAIZE": ("cereal", False),
    "BARLEY": ("cereal", False),
    "SORGHUM": ("millet", True),            # total = kharif + rabi
    "KHARIF SORGHUM": ("millet", False),
    "RABI SORGHUM": ("millet", False),
    "PEARL MILLET": ("millet", False),
    "FINGER MILLET": ("millet", False),
    "CHICKPEA": ("pulse", False),
    "PIGEONPEA": ("pulse", False),
    "MINOR PULSES": ("pulse", False),
    "GROUNDNUT": ("oilseed", False),
    "SESAMUM": ("oilseed", False),
    "RAPESEED AND MUSTARD": ("oilseed", False),
    "SAFFLOWER": ("oilseed", False),
    "CASTOR": ("oilseed", False),
    "LINSEED": ("oilseed", False),
    "SUNFLOWER": ("oilseed", False),
    "SOYABEAN": ("oilseed", False),
    "OILSEEDS": ("oilseed", True),          # total of all oilseeds above
    "SUGARCANE": ("cash", False),
    "COTTON": ("cash", False),
}

# Crops reported with area only (no production/yield)
AREA_ONLY = ["FRUITS", "VEGETABLES", "FRUITS AND VEGETABLES", "POTATOES", "ONION", "FODDER"]

STATE_RENAME = {"Orissa": "Odisha"}

# IMD standard meteorological seasons
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
          "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
MON3 = [m[:3].lower() for m in MONTHS]
SEASONS = {
    "winter": ["jan", "feb"],
    "pre_monsoon": ["mar", "apr", "may"],
    "monsoon": ["jun", "jul", "aug", "sep"],
    "post_monsoon": ["oct", "nov", "dec"],
}


# ---------------------------------------------------------------- Phase 4
# Modelling scope: crops with >= 7,000 valid observations. "Minor Pulses" is a
# bundle of several crops, so it is excluded as heterogeneous.
MODEL_CROPS = ["Rice", "Maize", "Chickpea", "Wheat", "Sugarcane", "Sesamum", "Pigeonpea",
               "Groundnut", "Rapeseed And Mustard", "Kharif Sorghum", "Pearl Millet",
               "Barley", "Cotton"]
CROP_SEASON = {"Rice": "kharif", "Maize": "kharif", "Pearl Millet": "kharif",
               "Kharif Sorghum": "kharif", "Groundnut": "kharif", "Cotton": "kharif",
               "Pigeonpea": "kharif", "Sesamum": "kharif",
               "Wheat": "rabi", "Chickpea": "rabi", "Rapeseed And Mustard": "rabi",
               "Barley": "rabi", "Sugarcane": "annual"}

# Rainfall is only available up to 2015.
LAST_RAIN_YEAR = 2015
# Reference climatology for anomalies and SPI. It ends at the END OF THE EARLIEST
# CV TRAINING WINDOW, so no validation/test year ever influences a feature.
REF_PERIOD = (1966, 1989)

# Time-based split (never random: this is panel data with a strong time trend)
TRAIN_END = 2004
VALID_YEARS = (2005, 2009)
TEST_YEARS = (2010, 2015)
# Expanding-window CV inside the training set (used for tuning in Phase 5)
CV_FOLDS = [(1989, (1990, 1994)), (1994, (1995, 1999)), (1999, (2000, 2004))]
