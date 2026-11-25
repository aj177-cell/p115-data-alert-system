p115 Market Data Quality Detectors - v3 final

Run from the project root:
  pip install pandas numpy pyarrow matplotlib pytest
  python -m pytest detectors
  python analyze_v2.py      # all checks -> results_real_v2/
  python analyze_v3.py      # if needed: missing_intervals v3 rule (min_gap_s=5.0) -> results_real_v3/
  python make_charts.py     # charts -> results_real_v3/charts/

Notes:
- Raw data (data/*/data.parquet) is NOT included and must be obtained separately
  (place it under data/ in the project root). Precomputed flags are included, so
  pytest and make_charts.py work without it.
- Team pipeline files (team_pipeline/, databento_pipeline.py, notebook) are not included.
- detectors/config.py is v3; config_v1.py / config_v2.py kept for reference.
- make_charts_v1.py is the old version that read results_real/allflags.parquet.
