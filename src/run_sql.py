"""Run every '-- name:' block in a .sql file against agrisense.db and print results."""
import re, sqlite3, sys
import pandas as pd
from src import config as C

def run(path):
    text = open(path).read()
    blocks = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)[1:]
    con = sqlite3.connect(C.DB_PATH)
    out = {}
    for name, sql in zip(blocks[::2], blocks[1::2]):
        out[name] = pd.read_sql_query(sql, con)
    return out

if __name__ == "__main__":
    for name, df in run(sys.argv[1]).items():
        print(f"\n=== {name} ({len(df)} rows)")
        print(df.to_string(index=False))
