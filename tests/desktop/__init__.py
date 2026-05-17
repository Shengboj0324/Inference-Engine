"""Phase 1 — local-first backend test suite.

Validates the modules under :mod:`app.local` and the :mod:`app.desktop`
sidecar launcher.  Every test sets ``SMR_DATA_DIR`` to a per-test
``tmp_path`` so no test ever touches the operator's real data directory.
"""
