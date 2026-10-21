"""Exploratory figures: how the recall rate varies by component and by recent
complaint volume. Uses SQL aggregation directly on the model_data table."""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nhtsa.db"
FIG_DIR = ROOT / "figures"

BY_COMPONENT = """
SELECT comp_group, COUNT(*) AS n, AVG(recall_within_1y) AS recall_rate
FROM model_data
GROUP BY comp_group
HAVING COUNT(*) >= 3000
ORDER BY recall_rate
"""

BY_PRIOR_COMPLAINTS = """
SELECT CASE
         WHEN prior_cmpl_90d = 0 THEN '0'
         WHEN prior_cmpl_90d <= 2 THEN '1-2'
         WHEN prior_cmpl_90d <= 9 THEN '3-9'
         WHEN prior_cmpl_90d <= 49 THEN '10-49'
         ELSE '50+'
       END AS bucket,
       MIN(prior_cmpl_90d) AS sort_key,
       COUNT(*) AS n,
       AVG(recall_within_1y) AS recall_rate
FROM model_data
GROUP BY bucket
ORDER BY sort_key
"""


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        comp = pd.read_sql(BY_COMPONENT, con)
        prior = pd.read_sql(BY_PRIOR_COMPLAINTS, con)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].barh(comp["comp_group"].str.title(), comp["recall_rate"] * 100, color="#4C72B0")
    axes[0].set(title="Share of complaints followed by a recall within 1 year",
                xlabel="% of complaints")
    axes[1].bar(prior["bucket"], prior["recall_rate"] * 100, color="#DD8452")
    axes[1].set(title="Recall rate by complaint volume in prior 90 days\n(same vehicle + component)",
                xlabel="Earlier complaints in the past 90 days", ylabel="% followed by a recall")
    for ax in axes:
        ax.grid(alpha=0.3, axis="x" if ax is axes[0] else "y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recall_rate_explore.png", dpi=150)
    print(comp.to_string(index=False))
    print(prior.to_string(index=False))


if __name__ == "__main__":
    main()
