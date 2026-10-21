"""Download NHTSA complaint and recall flat files.

Source: NHTSA Office of Defects Investigation (ODI) public flat files
https://www.nhtsa.gov/nhtsa-datasets-and-apis

Files are tab-delimited text inside zip archives. They are saved to data/raw/
(which is git-ignored because the files are large).
"""

from pathlib import Path

import requests

BASE = "https://static.nhtsa.gov/odi/ffdd"
FILES = {
    # Complaints, split into 5-year chunks by the date NHTSA received them
    "COMPLAINTS_RECEIVED_2015-2019.zip": f"{BASE}/cmpl/COMPLAINTS_RECEIVED_2015-2019.zip",
    "COMPLAINTS_RECEIVED_2020-2024.zip": f"{BASE}/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip",
    # All recalls filed since 2010
    "FLAT_RCL_POST_2010.zip": f"{BASE}/rcl/FLAT_RCL_POST_2010.zip",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print(f"Downloading {name}")
        download(url, RAW_DIR / name)


if __name__ == "__main__":
    main()
