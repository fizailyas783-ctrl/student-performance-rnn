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
import sys
import subprocess

st.set_page_config(
    page_title="Student Performance Evaluator",
    page_icon="🎓",
    layout="centered"
)

# Train model if not exists
if not os.path.exists("model.pkl"):
    with st.spinner("Training model for first time... please wait..."):
        subprocess.run([sys.executable, "train_rnn.py"], check=True)

@st.cache_resource
def load_predictor():
    from predict import predict_student
    return predict_student

predict_student = load_predictor()

# ── TITLE ──
st.title("🎓 Student Performance Evaluator")
st.markdown("Predict whether a student will **Pass** or **Fail** based on 3 weeks of academic data.")

st.divider()

# ── TABS ──
tab1, tab2 = st.tabs(["✍️ Manual Input", "📂 Upload Excel File"])

# ════════════════════════════════════════
# TAB 1 — MANUAL SLIDERS
# ════════════════════════════════════════
with tab1:
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
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True, key="manual_btn"):
        with st.spinner("Running prediction..."):
            try:
                result = predict_student(weekly_data)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        st.divider()
        st.subheader("📢 Result")
        if result == "Pass":
            st.success("## ✅ PASS\nThe model predicts this student will **Pass**!", icon="🏆")
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

# ════════════════════════════════════════
# TAB 2 — FILE UPLOAD
# ════════════════════════════════════════
with tab2:
    st.subheader("📂 Upload Student Excel File")
    st.markdown("""
    **File format required** — columns:
    `student_id`, `week`, `attendance`, `quiz`, `assignment`, `study_hours`
    """)

    # Sample file download
    sample_df = pd.DataFrame({
        "student_id": [1,1,1, 2,2,2],
        "week":       [1,2,3, 1,2,3],
        "attendance": [90,85,88, 40,45,50],
        "quiz":       [80,75,82, 30,28,25],
        "assignment": [85,80,88, 35,30,40],
        "study_hours":[7,6,8,   1.5,1,2]
    })
    st.download_button(
        label="⬇️ Download Sample File (CSV)",
        data=sample_df.to_csv(index=False).encode(),
        file_name="sample_student_data.csv",
        mime="text/csv"
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload your file (.xlsx or .csv)", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            required_cols = ["student_id", "week", "attendance", "quiz", "assignment", "study_hours"]
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ File loaded! {df['student_id'].nunique()} students found.")
                st.dataframe(df.head(10), use_container_width=True)

                st.divider()
                if st.button("🔮 Predict All Students", type="primary", use_container_width=True, key="file_btn"):
                    df = df.sort_values(["student_id", "week"]).reset_index(drop=True)
                    results = []
                    errors  = []

                    for student_id, group in df.groupby("student_id"):
                        group = group.reset_index(drop=True)
                        if len(group) < 3:
                            errors.append(f"Student {student_id}: less than 3 weeks — skipped.")
                            continue
                        seq = group.tail(3)[["attendance","quiz","assignment","study_hours"]].values.tolist()
                        try:
                            pred = predict_student(seq)
                            results.append({"Student ID": student_id, "Prediction": pred})
                        except Exception as e:
                            errors.append(f"Student {student_id}: error — {e}")

                    if results:
                        result_df = pd.DataFrame(results)
                        st.subheader("📢 Prediction Results")
                        st.dataframe(result_df, use_container_width=True)

                        pass_count = sum(1 for r in results if r["Prediction"] == "Pass")
                        fail_count = len(results) - pass_count
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total", len(results))
                        c2.metric("✅ Pass", pass_count)
                        c3.metric("❌ Fail", fail_count)

                        st.download_button(
                            "⬇️ Download Results (CSV)",
                            data=result_df.to_csv(index=False).encode(),
                            file_name="results.csv",
                            mime="text/csv"
                        )

                    if errors:
                        st.warning("⚠️ Skipped:")
                        for e in errors:
                            st.write(e)

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

st.divider()
st.caption("Built with 🐍 Python · Scikit-Learn · Streamlit")
