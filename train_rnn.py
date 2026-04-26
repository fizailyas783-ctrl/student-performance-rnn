"""
train_rnn.py
============
Student Performance Evaluator — Training Script
Uses RandomForest (no TensorFlow needed)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset.xlsx"
MODEL_PATH   = "model.pkl"
SCALER_PATH  = "scaler.pkl"
SEQUENCE_LEN = 3
FEATURES     = ["attendance", "quiz", "assignment", "study_hours"]
LABEL_COL    = "result"

print("Loading dataset...")
df = pd.read_excel(DATASET_PATH)
df = df.sort_values(["student_id", "week"]).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].map({"Pass": 1, "Fail": 0})

scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, SCALER_PATH)

X_sequences, y_labels = [], []
for _, group in df.groupby("student_id"):
    group = group.reset_index(drop=True)
    for i in range(len(group) - SEQUENCE_LEN + 1):
        window = group.iloc[i : i + SEQUENCE_LEN]
        seq    = window[FEATURES].values.flatten()
        label  = window[LABEL_COL].iloc[-1]
        X_sequences.append(seq)
        y_labels.append(label)

X = np.array(X_sequences)
y = np.array(y_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")
print("Training complete!")
