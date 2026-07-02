# Compatibility shim: some legacy baseline dependencies (genetic_selection, mcdm) still import
# the private module `sklearn.utils._joblib`, which was removed in modern scikit-learn. Alias it
# to `joblib` so those baselines can be imported. Harmless for the other baselines.
import sys as _sys
try:  # pragma: no cover
    import sklearn.utils._joblib  # noqa: F401
except Exception:
    import joblib as _joblib
    _sys.modules['sklearn.utils._joblib'] = _joblib
