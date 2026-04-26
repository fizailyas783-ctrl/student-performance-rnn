"""
app.py
======
RNN-Based Student Performance Evaluator — Streamlit Web App

Run with:
    streamlit run app.py

The user enters 3 weeks of student data (attendance, quiz score,
assignment score, study hours) and the app predicts Pass / Fail.
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import streamlit as st
import numpy as np
import os

# Suppress verbose TensorFlow logs before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Student Performance Evaluator",
    page_icon  = "🎓",
    layout     = "centered"
)

# ──────────────────────────────────────────────
# LOAD MODEL (cached so it loads only once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the trained model and scaler exactly once."""
    from predict import predict_student
    return predict_student

predict_student = load_predictor()

# ──────────────────────────────────────────────
# TITLE & DESCRIPTION
# ──────────────────────────────────────────────
st.title("🎓 Student Performance Evaluator")
st.markdown(
    """
    This app uses an **LSTM-based Recurrent Neural Network (RNN)** to predict
    whether a student will **Pass** or **Fail** based on their last 3 weeks of
    academic activity.

    Fill in the form below and click **Predict** to get the result.
    """
)

st.divider()

# ──────────────────────────────────────────────
# INPUT FORM — 3 WEEKS
# ──────────────────────────────────────────────
st.subheader("📋 Enter 3 Weeks of Student Data")

# We'll collect data week by week using expandable sections
weekly_data = []

for week in range(1, 4):
    with st.expander(f"📅 Week {week}", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider(
                label   = f"Attendance (%) — Week {week}",
                min_value = 0,
                max_value = 100,
                value     = 75,
                step      = 1,
                key       = f"attendance_{week}"
            )
            quiz = st.slider(
                label   = f"Quiz Score (%) — Week {week}",
                min_value = 0,
                max_value = 100,
                value     = 65,
                step      = 1,
                key       = f"quiz_{week}"
            )

        with col2:
            assignment = st.slider(
                label   = f"Assignment Score (%) — Week {week}",
                min_value = 0,
                max_value = 100,
                value     = 70,
                step      = 1,
                key       = f"assignment_{week}"
            )
            study_hours = st.slider(
                label   = f"Study Hours / Week — Week {week}",
                min_value = 0.0,
                max_value = 12.0,
                value     = 5.0,
                step      = 0.5,
                key       = f"study_hours_{week}"
            )

        weekly_data.append([attendance, quiz, assignment, study_hours])

st.divider()

# ──────────────────────────────────────────────
# PREVIEW TABLE
# ──────────────────────────────────────────────
st.subheader("📊 Data Preview")

import pandas as pd

preview_df = pd.DataFrame(
    weekly_data,
    columns = ["Attendance (%)", "Quiz (%)", "Assignment (%)", "Study Hours"],
    index   = ["Week 1", "Week 2", "Week 3"]
)
st.dataframe(preview_df, use_container_width=True)

# ──────────────────────────────────────────────
# PREDICT BUTTON
# ──────────────────────────────────────────────
st.divider()

predict_btn = st.button("🔮 Predict Performance", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Running prediction..."):
        try:
            result = predict_student(weekly_data)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    st.divider()
    st.subheader("📢 Prediction Result")

    if result == "Pass":
        st.success(
            "## ✅ PASS\n"
            "The model predicts this student will **Pass**. "
            "Keep up the good work!",
            icon="🏆"
        )
        st.balloons()
    else:
        st.error(
            "## ❌ FAIL\n"
            "The model predicts this student is at risk of **Failing**. "
            "Consider additional support and study time.",
            icon="⚠️"
        )

    # ── Tips based on weak areas ──────────────────────────────────────
    avg_attendance  = np.mean([row[0] for row in weekly_data])
    avg_quiz        = np.mean([row[1] for row in weekly_data])
    avg_assignment  = np.mean([row[2] for row in weekly_data])
    avg_study       = np.mean([row[3] for row in weekly_data])

    st.divider()
    st.subheader("💡 Insights")

    tips = []
    if avg_attendance < 75:
        tips.append(f"📌 Average attendance is **{avg_attendance:.1f}%** — aim for at least 75%.")
    if avg_quiz < 60:
        tips.append(f"📌 Average quiz score is **{avg_quiz:.1f}%** — more practice quizzes may help.")
    if avg_assignment < 65:
        tips.append(f"📌 Average assignment score is **{avg_assignment:.1f}%** — review assignment feedback.")
    if avg_study < 4:
        tips.append(f"📌 Average study time is **{avg_study:.1f} hrs/week** — increasing this could improve results.")

    if tips:
        for tip in tips:
            st.markdown(tip)
    else:
        st.markdown("✨ All metrics look healthy. Maintain consistency!")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Built with 🐍 Python · TensorFlow · Streamlit  |  "
    "Model: LSTM-based RNN (64 units)  |  Sequence length: 3 weeks"
)
