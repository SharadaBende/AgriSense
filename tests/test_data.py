import sqlite3
import numpy as np
import pandas as pd
import pytest
from src import config as C

@pytest.fixture(scope="module")
def con():
    return sqlite3.connect(C.DB_PATH)

def q(con, sql):
    return pd.read_sql_query(sql, con)

def test_no_missing_code_left(con):
    assert q(con, "select count(*) n from crop_yield where area_1000ha<0 or production_1000t<0").n[0] == 0
    assert q(con, "select count(*) n from rainfall where rain_jun<0 or annual_mm<0").n[0] == 0

def test_only_positive_area_and_production(con):
    assert q(con, "select count(*) n from crop_yield where area_1000ha<=0 or production_1000t<=0").n[0] == 0

def test_yield_is_consistent(con):
    df = q(con, "select area_1000ha a, production_1000t p, yield_kg_ha y from crop_yield")
    assert np.allclose(df.y, df.p / df.a * 1000, rtol=1e-9)

def test_unique_keys(con):
    assert q(con, "select count(*) n from (select dist_code,crop,year from crop_yield group by 1,2,3 having count(*)>1)").n[0] == 0
    assert q(con, "select count(*) n from (select dist_code,year from rainfall group by 1,2 having count(*)>1)").n[0] == 0

def test_every_crop_row_has_a_district(con):
    assert q(con, "select count(*) n from crop_yield c left join districts d using(dist_code) where d.dist_code is null").n[0] == 0

def test_annual_equals_sum_of_months(con):
    df = q(con, "select * from rainfall where annual_mm is not null")
    months = [c for c in df.columns if c.startswith("rain_")]
    assert np.allclose(df.annual_mm, df[months].sum(axis=1))

def test_plausible_ranges(con):
    assert q(con, "select max(yield_kg_ha) m from crop_yield where crop in ('Rice','Wheat')").m[0] < 8000
    assert q(con, "select max(annual_mm) m from rainfall").m[0] < 12000
    assert q(con, "select min(year) a, max(year) b from crop_yield").values.tolist() == [[1966, 2017]]

def test_odisha_renamed(con):
    assert q(con, "select count(*) n from districts where state_name='Orissa'").n[0] == 0
