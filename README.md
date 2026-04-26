# 🎓 RNN-Based Student Performance Evaluator

A machine-learning system that uses a **Recurrent Neural Network (LSTM)** to predict
whether a student will **Pass** or **Fail** based on three consecutive weeks of
academic performance data.

---

## 📚 What is an RNN?

A **Recurrent Neural Network (RNN)** is a type of neural network designed to work
with **sequential / time-series data**. Unlike a regular feed-forward network that
looks at a single snapshot, an RNN remembers information from previous steps in the
sequence.

An **LSTM (Long Short-Term Memory)** is a special type of RNN that is much better at
remembering long-term patterns because it has internal "gates" that control what to
remember and what to forget.

```
Week 1 data → ┐
Week 2 data → ├──► LSTM ──► Pass / Fail prediction
Week 3 data → ┘
```

In this project each "time step" is one week of a student's activity, and the LSTM
learns how multi-week trends (improving attendance, declining quiz scores, etc.)
relate to the final outcome.

---

## 🗂 Project Structure

```
project/
│
├── dataset.xlsx          # Student performance data (Excel)
├── generate_dataset.py   # Script to regenerate the dataset
├── train_rnn.py          # Model training script
├── predict.py            # Prediction helper module
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

After training, two additional files are created:

```
├── model.h5              # Saved Keras LSTM model
└── scaler.pkl            # Saved StandardScaler
```

---

## 🧠 How the Model Works

| Step | What happens |
|------|-------------|
| **Load data** | `dataset.xlsx` is read with pandas |
| **Sort** | Rows are sorted by `student_id` then `week` |
| **Sequence** | A sliding window of 3 weeks is created per student |
| **Normalize** | Features are scaled with `StandardScaler` (mean=0, std=1) |
| **Encode labels** | `Pass → 1`, `Fail → 0` |
| **LSTM model** | Input `(3, 4)` → LSTM 64 units → Dense sigmoid output |
| **Train** | `binary_crossentropy` loss, `adam` optimizer, 10 epochs |
| **Save** | `model.h5` and `scaler.pkl` saved to disk |

### Dataset Columns

| Column | Description |
|--------|-------------|
| `student_id` | Unique identifier per student |
| `week` | Week number (1, 2, 3, …) |
| `attendance` | Attendance percentage (0–100) |
| `quiz` | Quiz score (0–100) |
| `assignment` | Assignment score (0–100) |
| `study_hours` | Hours studied that week |
| `result` | `Pass` or `Fail` (label) |

---

## ⚙️ How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (skip if `dataset.xlsx` already exists)

```bash
python generate_dataset.py
```

### 3. Train the model

```bash
python train_rnn.py
```

This will:
- Print training progress for each epoch
- Save `model.h5` (trained model)
- Save `scaler.pkl` (feature scaler)

### 4. (Optional) Test the prediction module

```bash
python predict.py
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

---

## 🖥 Streamlit App Usage

1. Use the sliders to enter **3 weeks** of student data:
   - Attendance (%)
   - Quiz score (%)
   - Assignment score (%)
   - Study hours per week
2. Click **🔮 Predict Performance**
3. The model returns **✅ PASS** or **❌ FAIL** along with personalised insights.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | StandardScaler for feature normalisation |
| `tensorflow` | LSTM model (Keras API) |
| `streamlit` | Web application UI |
| `openpyxl` | Reading `.xlsx` files |
| `joblib` | Saving / loading the scaler |

---

## 🤖 Model Architecture

```
Model: "student_lstm_model"
_________________________________________________________________
Layer          Output Shape         Param #
=================================================================
lstm_layer     (None, 64)           17,920
output_layer   (None, 1)            65
=================================================================
Total params: 17,985
```

- **Input:**  `(batch, 3, 4)` — 3 weeks × 4 features
- **LSTM:**   64 units, processes the 3-step sequence
- **Output:** 1 unit with sigmoid → probability of passing

---

*Built with Python · TensorFlow · Streamlit*
