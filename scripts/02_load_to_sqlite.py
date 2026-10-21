"""Load the raw NHTSA flat files into a SQLite database (data/nhtsa.db).

Creates two tables:
  complaints  one row per complaint x component (NHTSA's CMPLID)
  recalls     one row per recall campaign x vehicle x component

Column layouts come from NHTSA's CMPL.txt and RCL.txt field descriptions.
Only the columns used downstream are kept.
"""

import csv
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "nhtsa.db"

# Full 51-field complaint layout (CMPL.txt)
COMPLAINT_COLS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN",
    "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]
COMPLAINT_KEEP = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "LDATE", "MILES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "FUEL_TYPE", "VEH_SPEED",
    "PROD_TYPE", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
]

# Full 29-field recall layout (RCL.txt)
RECALL_COLS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE",
    "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS",
    "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
    "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO",
    "DO_NOT_DRIVE", "PARK_OUTSIDE",
]
RECALL_KEEP = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "COMPNAME",
    "RCLTYPECD", "POTAFF", "INFLUENCED_BY", "RCDATE",
]


def read_flat_file(zip_path: Path, columns: list[str], keep: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=columns,
                usecols=keep,
                dtype=str,
                quoting=csv.QUOTE_NONE,
                encoding="latin-1",
                on_bad_lines="skip",
            )
    print(f"  {zip_path.name}: {len(df):,} rows")
    return df


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)

    print("Loading complaints")
    for i, zip_path in enumerate(sorted(RAW_DIR.glob("COMPLAINTS_RECEIVED_*.zip"))):
        df = read_flat_file(zip_path, COMPLAINT_COLS, COMPLAINT_KEEP)
        df.to_sql("complaints", con, if_exists="replace" if i == 0 else "append", index=False)

    print("Loading recalls")
    df = read_flat_file(RAW_DIR / "FLAT_RCL_POST_2010.zip", RECALL_COLS, RECALL_KEEP)
    df.to_sql("recalls", con, if_exists="replace", index=False)

    print("Creating indexes")
    con.executescript(
        """
        CREATE INDEX idx_cmpl_vehicle ON complaints (MAKETXT, MODELTXT, YEARTXT);
        CREATE INDEX idx_rcl_vehicle ON recalls (MAKETXT, MODELTXT, YEARTXT);
        """
    )
    for table in ("complaints", "recalls"):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows")
    con.close()


if __name__ == "__main__":
    main()
