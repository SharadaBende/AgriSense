"""Match ICRISAT districts (1966 boundaries) to GADM district polygons."""
import re, difflib
import geopandas as gpd, pandas as pd
from src import config as C

GEO = C.RAW / "india_district_boundaries.geojson"
GADM_STATE = {"Odisha": "Orissa", "Uttarakhand": "Uttaranchal", "Telangana": "Andhra Pradesh"}

def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())

def _variants(name):
    parts = [p.strip() for p in re.split(r"/", name)] + [re.sub(r"\(.*?\)", "", name)]
    return {_norm(p) for p in parts if p}

# Old (1966) districts that were later renamed/split: the modern polygons are
# dissolved into one shape. Approximate by nature - documented in the README.
MANUAL = {
    "Kadapa YSR": ["Cuddapah"],
    "Bangalore": ["Bangalore Rural", "Bangalore Urban"],
    "Thirunelveli": ["Tirunelveli Kattabo"],
    "Bombay": ["Greater Bombay"],
    "Balasore": ["Baleshwar"],
    "24 Parganas": ["North 24 Parganas", "South 24 Parganas"],
    "Burdwan": ["Barddhaman"],
    "Hooghly": ["Hugli"],
    "Howrah": ["Haora"],
    "Cooch Behar": ["Kochbihar"],
    "Midnapur": ["East Midnapore", "West Midnapore"],
    "West Dinajpur": ["Uttar Dinajpur", "Dakshin Dinajpur"],
    "Champaran": ["Pashchim Champaran", "Purba Champaran"],
    "Mungair": ["Munger"],
    "Shahabad (now part of Bhojpur district)": ["Bhojpur", "Buxar", "Rohtas"],
    "Singhbhum": ["Pashchim Singhbhum", "Purba Singhbhum", "Saraikela Kharsawan"],
}


def build_crosswalk(cutoff=0.80):
    g = gpd.read_file(GEO)[["NAME_1", "NAME_2", "VARNAME_2", "geometry"]]
    d = pd.read_csv(C.PROCESSED / "districts.csv")
    rows = []
    for r in d.itertuples():
        st = GADM_STATE.get(r.state_name, r.state_name)
        cand = g[g.NAME_1 == st]
        keys = {_norm(n): n for n in cand.NAME_2}
        for _, row in cand.iterrows():                      # alternate spellings
            for v in str(row.VARNAME_2).split("|"):
                keys.setdefault(_norm(v), row.NAME_2)
        hit, how = None, None
        for v in _variants(r.dist_name):
            if v in keys:
                hit, how = keys[v], "exact"; break
        if hit is None:
            best = max(((difflib.SequenceMatcher(None, v, k).ratio(), k) for v in _variants(r.dist_name) for k in keys),
                       default=(0, None))
            if best[0] >= cutoff:
                hit, how = keys[best[1]], f"fuzzy({best[0]:.2f})"
        rows.append((r.dist_code, r.dist_name, r.state_name, hit, how))
    x = pd.DataFrame(rows, columns=["dist_code", "dist_name", "state_name", "gadm_name", "match"])
    taken = set(zip(x.state_name.map(lambda v: GADM_STATE.get(v, v)), x.gadm_name))
    for i, r in x.iterrows():
        if r.dist_name in MANUAL:
            st = GADM_STATE.get(r.state_name, r.state_name)
            names = [n for n in MANUAL[r.dist_name]
                     if (st, n) not in taken or n == (r.gadm_name or "")]
            x.loc[i, "gadm_name"], x.loc[i, "match"] = "|".join(names), "manual"
    return x, g


def district_geometries():
    """One (multi)polygon per ICRISAT district, dissolving split districts."""
    x, g = build_crosswalk()
    recs = []
    for r in x.dropna(subset=["gadm_name"]).itertuples():
        st = GADM_STATE.get(r.state_name, r.state_name)
        names = r.gadm_name.split("|")
        sel = g[(g.NAME_1 == st) & (g.NAME_2.isin(names))]
        if len(sel):
            recs.append({"dist_code": r.dist_code, "geometry": sel.geometry.union_all()})
    return gpd.GeoDataFrame(recs, crs=g.crs), x

if __name__ == "__main__":
    x, g = build_crosswalk()
    print(x.match.fillna("NONE").str.replace(r"\(.*\)", "", regex=True).value_counts())
    print("\n-- fuzzy matches to eyeball --")
    print(x[x.match.fillna("").str.startswith("fuzzy")][["dist_name", "state_name", "gadm_name", "match"]].to_string(index=False))
    print("\n-- unmatched --")
    print(x[x.gadm_name.isna()][["dist_name", "state_name"]].to_string(index=False))
