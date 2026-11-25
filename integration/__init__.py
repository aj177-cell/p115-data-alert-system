"""Add-only glue that runs the Milestone 1 alert engine and the v3 detectors side by side.

Nothing in Milestone 1 (README.md, requirements.txt, run_checks.py, databento_pipeline.py,
alert_system/, tests/, sample_data/, docs/research.md) or in the v3 package (p115_detectors/)
is edited or monkey-patched. ``python -m integration.verify`` checks this byte for byte.
"""
