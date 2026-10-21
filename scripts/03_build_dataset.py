"""Run sql/build_dataset.sql against data/nhtsa.db to create the model_data table."""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nhtsa.db"
SQL_PATH = ROOT / "sql" / "build_dataset.sql"


def main() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(SQL_PATH.read_text())
        n, pos = con.execute("SELECT COUNT(*), AVG(recall_within_1y) FROM model_data").fetchone()
    print(f"model_data: {n:,} complaint rows, {pos:.1%} followed by a recall within 1 year")


if __name__ == "__main__":
    main()
