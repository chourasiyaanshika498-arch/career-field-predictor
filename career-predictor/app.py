"""
app.py
------
Streamlit app: a student enters their academic scores, skill ratings, and
activity counts, and the app returns a ranked list of best-fit career fields
with a SHAP-based explanation of why.

Run:
    streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Best-Fit Career Field Predictor", page_icon="🎯", layout="centered")

MODEL_DIR = "career-predictor/models"


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(f"{MODEL_DIR}/model.json")
    classes = np.load(f"{MODEL_DIR}/classes.npy", allow_pickle=True)
    with open(f"{MODEL_DIR}/features.json") as f:
        features = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, classes, features, explainer


model, classes, features, explainer = load_model()

st.title("🎯 Best-Fit Career Field Predictor")
st.write(
    "Enter your academic scores, skill ratings, and activity counts below. "
    "The model will recommend the career field(s) that best match your profile, "
    "along with the top reasons behind the recommendation."
)

st.subheader("Academic Scores (0-100)")
col1, col2 = st.columns(2)
with col1:
    dsa = st.slider("DSA Score", 0, 100, 70)
    oop = st.slider("OOP Score", 0, 100, 70)
    dbms = st.slider("DBMS Score", 0, 100, 65)
    os_score = st.slider("Operating Systems Score", 0, 100, 60)
with col2:
    networks = st.slider("Computer Networks Score", 0, 100, 55)
    math_stats = st.slider("Math & Statistics Score", 0, 100, 70)
    problem_solving_score = st.slider("Problem Solving Score", 0, 100, 70)
    research_interest = st.slider("Research Interest", 0, 100, 50)

st.subheader("Skill Ratings (0-100, self-assessed)")
col3, col4 = st.columns(2)
with col3:
    ml_skill = st.slider("ML/AI Skill", 0, 100, 50)
    web_skill = st.slider("Web Dev Skill", 0, 100, 50)
with col4:
    security_skill = st.slider("Security Skill", 0, 100, 40)
    design_skill = st.slider("Design Skill", 0, 100, 40)

st.subheader("Activity & Soft Skills")
col5, col6 = st.columns(2)
with col5:
    coding_hours_week = st.number_input("Coding hours/week", 0, 60, 10)
    projects_count = st.number_input("Number of projects built", 0, 20, 3)
    internships_count = st.number_input("Internships completed", 0, 5, 0)
    hackathons = st.number_input("Hackathons participated", 0, 20, 1)
with col6:
    github_repos = st.number_input("GitHub repos", 0, 50, 4)
    certifications = st.number_input("Certifications earned", 0, 20, 2)
    leadership_score = st.slider("Leadership Score", 0, 100, 50)
    communication_score = st.slider("Communication Score", 0, 100, 55)

if st.button("Predict Best-Fit Field", type="primary"):
    student = pd.DataFrame([{
        "dsa": dsa, "oop": oop, "dbms": dbms, "os": os_score, "networks": networks,
        "math_stats": math_stats, "coding_hours_week": coding_hours_week,
        "projects_count": projects_count, "internships_count": internships_count,
        "ml_skill": ml_skill, "web_skill": web_skill, "security_skill": security_skill,
        "design_skill": design_skill, "hackathons": hackathons, "github_repos": github_repos,
        "certifications": certifications, "leadership_score": leadership_score,
        "communication_score": communication_score, "problem_solving_score": problem_solving_score,
        "research_interest": research_interest,
    }])[features]

    probs = model.predict_proba(student)[0]
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    st.subheader("📊 Recommended Fields")
    for field, prob in ranked:
        st.write(f"**{field}** — {prob*100:.1f}% match")
        st.progress(float(prob))

    top_field = ranked[0][0]
    top_idx = list(classes).index(top_field)

    st.subheader(f"🔍 Why \"{top_field}\"?")
    shap_values = explainer.shap_values(student)

    if isinstance(shap_values, list):
        sv = shap_values[top_idx][0]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, top_idx]
    else:
        sv = shap_values[0]

    contrib = pd.Series(sv, index=features).sort_values(key=abs, ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2E75B6" if v > 0 else "#C00000" for v in contrib.values]
    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on prediction (SHAP value)")
    ax.set_title(f"Top factors driving '{top_field}' recommendation")
    st.pyplot(fig)

    st.caption(
        "Blue bars push the prediction toward this field, red bars push away from it. "
        "Bar length shows the strength of that feature's influence."
    )

st.divider()
st.caption("Built with XGBoost + SHAP. Trained on a labeled student profile dataset (see README for details).")
