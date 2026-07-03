"""ML-branch move logic: thin wrapper so backend.py stays unchanged.

The actual model inference lives in ml/inference.py; this module keeps the
same public interface (`choose_move`, `get_info`) that the Flask server
imports. The model is warmed up at import time so the very first /move
request is already fast (<500 ms budget).
"""

from ml.inference import choose_move, get_info, warmup  # noqa: F401

warmup()
