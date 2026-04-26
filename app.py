"""
app.py
======
Student Performance Evaluator — Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Student Performance Evaluator",
    page_icon="🎓",
    layout="centered"
)

# Train model if not exists
if not os.path.exists("model.pkl"):
    import subprocess
    import sys
    with st.spinner("Training model for first time... please wait..."):
        subprocess.run([sys.executable, "train_rnn.py"], check=True)

@st.cache_resource
def load_predictor():
    from predict import predict_student
    return predict_student

predict_student = load_predictor()

st.title("🎓 Student Performance Evaluator")
st.markdown("""
This app predicts whether a student will **Pass** or **Fail**
based on **3 weeks** of academic performance data.
""")

st.divider()
st.subheader("📋 Enter 3 Weeks of Student Data")

weekly_data = []

for week in range(1, 4):
    with st.expander(f"📅 Week {week}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            attendance = st.slider(f"Attendance (%) — Week {week}", 0, 100, 75, key=f"att_{week}")
            quiz       = st.slider(f"Quiz Score (%) — Week {week}", 0, 100, 65, key=f"quiz_{week}")
        with col2:
            assignment  = st.slider(f"Assignment Score (%) — Week {week}", 0, 100, 70, key=f"asgn_{week}")
            study_hours = st.slider(f"Study Hours — Week {week}", 0.0, 12.0, 5.0, step=0.5, key=f"study_{week}")
        weekly_data.append([attendance, quiz, assignment, study_hours])

st.divider()
st.subheader("📊 Data Preview")
preview_df = pd.DataFrame(
    weekly_data,
    columns=["Attendance (%)", "Quiz (%)", "Assignment (%)", "Study Hours"],
    index=["Week 1", "Week 2", "Week 3"]
)
st.dataframe(preview_df, use_container_width=True)

st.divider()

if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
    with st.spinner("Running prediction..."):
        try:
            result = predict_student(weekly_data)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    st.divider()
    st.subheader("📢 Prediction Result")

    if result == "Pass":
        st.success("## ✅ PASS\nThe model predicts this student will **Pass**. Keep up the good work!", icon="🏆")
        st.balloons()
    else:
        st.error("## ❌ FAIL\nThe student is at risk of **Failing**. Consider additional support.", icon="⚠️")

    avg_attendance = np.mean([r[0] for r in weekly_data])
    avg_quiz       = np.mean([r[1] for r in weekly_data])
    avg_assignment = np.mean([r[2] for r in weekly_data])
    avg_study      = np.mean([r[3] for r in weekly_data])

    st.divider()
    st.subheader("💡 Insights")
    tips = []
    if avg_attendance < 75:
        tips.append(f"📌 Attendance **{avg_attendance:.1f}%** — aim for at least 75%.")
    if avg_quiz < 60:
        tips.append(f"📌 Quiz score **{avg_quiz:.1f}%** — practice more quizzes.")
    if avg_assignment < 65:
        tips.append(f"📌 Assignment score **{avg_assignment:.1f}%** — review feedback.")
    if avg_study < 4:
        tips.append(f"📌 Study hours **{avg_study:.1f}/week** — increase study time.")
    if tips:
        for tip in tips:
            st.markdown(tip)
    else:
        st.markdown("✨ All metrics look healthy. Maintain consistency!")

st.divider()
st.caption("Built with 🐍 Python · Scikit-Learn · Streamlit")
