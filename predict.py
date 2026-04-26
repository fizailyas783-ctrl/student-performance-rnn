"""
predict.py
==========
Student Performance Evaluator — Prediction Module
"""

import numpy as np
import joblib
import os

MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Load model and scaler
_model  = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)


def predict_student(sequence_data) -> str:
    """
    Predict Pass or Fail.
    sequence_data: shape (3, 4) — 3 weeks x [attendance, quiz, assignment, study_hours]
    """
    seq = np.array(sequence_data, dtype=np.float32)
    seq_scaled = _scaler.transform(seq)
    seq_flat   = seq_scaled.flatten().reshape(1, -1)
    pred = _model.predict(seq_flat)[0]
    return "Pass" if pred == 1 else "Fail"
