"""
train_rnn.py
============
RNN-Based Student Performance Evaluator — Training Script

This script:
  1. Loads the dataset from dataset.xlsx
  2. Preprocesses and sequences the data (3 weeks per sample)
  3. Trains an LSTM-based model to predict Pass / Fail
  4. Saves the trained model as model.h5 and the scaler as scaler.pkl
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Suppress verbose TensorFlow logs (optional)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_PATH  = "dataset.xlsx"
MODEL_PATH    = "model.h5"
SCALER_PATH   = "scaler.pkl"
SEQUENCE_LEN  = 3          # number of weeks used per prediction
FEATURES      = ["attendance", "quiz", "assignment", "study_hours"]
LABEL_COL     = "result"
EPOCHS        = 10
BATCH_SIZE    = 16
TEST_SPLIT    = 0.2
RANDOM_SEED   = 42

# ──────────────────────────────────────────────
# STEP 1 — LOAD & SORT DATA
# ──────────────────────────────────────────────
print("=" * 50)
print("  RNN Student Performance Evaluator — Training")
print("=" * 50)

print("\n📂 Loading dataset...")
df = pd.read_excel(DATASET_PATH)

# Sort by student then by week so time order is preserved
df = df.sort_values(["student_id", "week"]).reset_index(drop=True)
print(f"   Loaded {len(df)} rows | {df['student_id'].nunique()} students")

# ──────────────────────────────────────────────
# STEP 2 — ENCODE LABELS
# ──────────────────────────────────────────────
# Pass → 1,  Fail → 0
df[LABEL_COL] = df[LABEL_COL].map({"Pass": 1, "Fail": 0})

# ──────────────────────────────────────────────
# STEP 3 — NORMALIZE FEATURES
# ──────────────────────────────────────────────
print("\n🔧 Normalizing features with StandardScaler...")
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# Save scaler so predictions can use the same scaling
joblib.dump(scaler, SCALER_PATH)
print(f"   Scaler saved → {SCALER_PATH}")

# ──────────────────────────────────────────────
# STEP 4 — BUILD SEQUENCES
# ──────────────────────────────────────────────
print(f"\n🔀 Building sequences (length = {SEQUENCE_LEN} weeks)...")

X_sequences = []   # shape: (n_samples, SEQUENCE_LEN, n_features)
y_labels    = []   # shape: (n_samples,)

# Group by student and slide a window of SEQUENCE_LEN rows
for student_id, group in df.groupby("student_id"):
    group = group.reset_index(drop=True)

    # We need at least SEQUENCE_LEN rows for one sample
    for i in range(len(group) - SEQUENCE_LEN + 1):
        window = group.iloc[i : i + SEQUENCE_LEN]

        # Feature matrix for the window
        seq    = window[FEATURES].values          # (SEQUENCE_LEN, n_features)
        # Label is the result of the LAST week in the window
        label  = window[LABEL_COL].iloc[-1]

        X_sequences.append(seq)
        y_labels.append(label)

X = np.array(X_sequences, dtype=np.float32)   # (N, 3, 4)
y = np.array(y_labels,    dtype=np.float32)   # (N,)

print(f"   Total samples : {len(X)}")
print(f"   X shape       : {X.shape}")
print(f"   Class balance : Pass={int(y.sum())}  Fail={int((1-y).sum())}")

# ──────────────────────────────────────────────
# STEP 5 — TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
)
print(f"\n📊 Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ──────────────────────────────────────────────
# STEP 6 — BUILD LSTM MODEL
# ──────────────────────────────────────────────
print("\n🧠 Building LSTM model...")

model = Sequential([
    # LSTM layer — reads a sequence of SEQUENCE_LEN time steps,
    #              each with len(FEATURES) values
    LSTM(
        units       = 64,
        input_shape = (SEQUENCE_LEN, len(FEATURES)),
        name        = "lstm_layer"
    ),

    # Output layer — sigmoid squashes to [0, 1] for binary classification
    Dense(
        units      = 1,
        activation = "sigmoid",
        name       = "output_layer"
    )
], name="student_lstm_model")

model.compile(
    optimizer = "adam",
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)

model.summary()

# ──────────────────────────────────────────────
# STEP 7 — TRAIN
# ──────────────────────────────────────────────
print(f"\n🚀 Training for {EPOCHS} epochs...\n")

history = model.fit(
    X_train, y_train,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = (X_test, y_test),
    verbose         = 1
)

# ──────────────────────────────────────────────
# STEP 8 — EVALUATE
# ──────────────────────────────────────────────
print("\n📈 Final evaluation on test set:")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss     : {loss:.4f}")
print(f"   Test Accuracy : {accuracy * 100:.2f}%")

# ──────────────────────────────────────────────
# STEP 9 — SAVE MODEL
# ──────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\n✅ Model saved → {MODEL_PATH}")
print("   Training complete!")
