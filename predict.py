"""
predict.py
==========
RNN-Based Student Performance Evaluator — Prediction Module

Exposes a single public function:

    predict_student(sequence_data) -> "Pass" or "Fail"

sequence_data must be a 2-D array-like of shape (3, 4):
    - 3 rows  = 3 consecutive weeks
    - 4 cols  = [attendance, quiz, assignment, study_hours]
    (raw / un-scaled values — the scaler is applied internally)
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import numpy as np
import joblib

# Suppress verbose TensorFlow logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────
# FILE PATHS  (must match what train_rnn.py saved)
# ──────────────────────────────────────────────
MODEL_PATH  = "model.h5"
SCALER_PATH = "scaler.pkl"

# ──────────────────────────────────────────────
# LOAD MODEL & SCALER ONCE AT MODULE IMPORT
# ──────────────────────────────────────────────
print("🔄 Loading model and scaler...")
_model  = load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)
print("✅ Ready for predictions.")


# ──────────────────────────────────────────────
# PUBLIC FUNCTION
# ──────────────────────────────────────────────
def predict_student(sequence_data) -> str:
    """
    Predict whether a student will Pass or Fail.

    Parameters
    ----------
    sequence_data : array-like, shape (3, 4)
        Three weeks of raw (un-scaled) student data.
        Columns (in order): attendance, quiz, assignment, study_hours

    Returns
    -------
    str
        "Pass" if predicted probability >= 0.5, else "Fail"
    """
    # ── 1. Convert to numpy array ──────────────────────────────────────
    seq = np.array(sequence_data, dtype=np.float32)   # (3, 4)

    if seq.shape != (3, 4):
        raise ValueError(
            f"Expected shape (3, 4) but got {seq.shape}. "
            "Provide exactly 3 weeks × 4 features."
        )

    # ── 2. Normalise using the saved scaler ───────────────────────────
    # Reshape to (3, 4) → flatten to (3*4=12,) → scale → reshape back
    # StandardScaler works row-wise, so we scale each week individually.
    seq_scaled = _scaler.transform(seq)               # (3, 4)

    # ── 3. Add batch dimension: (1, 3, 4) ─────────────────────────────
    seq_batch = seq_scaled[np.newaxis, ...]            # (1, 3, 4)

    # ── 4. Run inference ───────────────────────────────────────────────
    prob = float(_model.predict(seq_batch, verbose=0)[0][0])

    # ── 5. Threshold at 0.5 ───────────────────────────────────────────
    return "Pass" if prob >= 0.5 else "Fail"


# ──────────────────────────────────────────────
# QUICK SELF-TEST  (runs only when called directly)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Quick self-test ---")

    # High-performing student (expect Pass)
    sample_pass = [
        [95.0, 88.0, 90.0, 8.0],   # week 1
        [90.0, 82.0, 87.0, 7.5],   # week 2
        [88.0, 85.0, 92.0, 9.0],   # week 3
    ]

    # Low-performing student (expect Fail)
    sample_fail = [
        [40.0, 30.0, 35.0, 1.5],   # week 1
        [45.0, 28.0, 30.0, 1.0],   # week 2
        [50.0, 25.0, 40.0, 2.0],   # week 3
    ]

    result_pass = predict_student(sample_pass)
    result_fail = predict_student(sample_fail)

    print(f"High-performer → {result_pass}")
    print(f"Low-performer  → {result_fail}")
