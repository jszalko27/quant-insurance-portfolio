"""Run every query in queries.sql against cyber_book.db, print the results and save each as CSV."""
import re, sqlite3, pathlib
import pandas as pd

con = sqlite3.connect("cyber_book.db")
out = pathlib.Path("output"); out.mkdir(exist_ok=True)
blocks = re.split(r"^-- name: (\w+)\s*$", pathlib.Path("queries.sql").read_text(), flags=re.M)[1:]
pd.set_option("display.width", 160)
for name, sql in zip(blocks[::2], blocks[1::2]):
    df = pd.read_sql(sql, con)
    df.to_csv(out / f"sql_{name}.csv", index=False)
    print(f"\n== {name} ==\n{df.to_string(index=False)}")
