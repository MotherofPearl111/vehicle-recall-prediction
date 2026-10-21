#!/usr/bin/env bash
# Reproduce the full project: download -> SQL database -> dataset -> models
set -euo pipefail
uv run python scripts/01_download_data.py
uv run python scripts/02_load_to_sqlite.py
uv run python scripts/03_build_dataset.py
uv run python scripts/04_train_models.py
uv run python scripts/05_explore_figures.py
