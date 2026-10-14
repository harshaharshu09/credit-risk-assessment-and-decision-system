from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "credit_risk_dataset.csv"
MODEL = ROOT / "models" / "best_model.joblib"
METRICS = ROOT / "outputs" / "model_comparison.csv"
st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")
st.title("Credit Risk – Explainable Financial Decision Making")
st.caption("Educational demo using synthetic data only; not for real lending decisions.")
df = pd.read_csv(DATA)
metrics = pd.read_csv(METRICS)
model = joblib.load(MODEL)
overview, explore, predict, performance = st.tabs(["Portfolio Overview", "Explore Data", "Applicant Estimate", "Model Performance"])
with overview:
    a, b, c = st.columns(3)
    a.metric("Synthetic applicants", f"{len(df):,}")
    b.metric("Simulated default rate", f"{df.loan_default.mean():.1%}")
    c.metric("Median annual income", f"₹{df.annual_income.median():,.0f}")
    st.bar_chart(df.loan_default.value_counts().sort_index())
    st.dataframe(df.head(20), use_container_width=True)
with explore:
    purpose = st.selectbox("Loan purpose", ["All"] + sorted(df.loan_purpose.unique().tolist()))
    view = df if purpose == "All" else df[df.loan_purpose == purpose]
    st.write(f"Rows: {len(view):,}")
    st.dataframe(view, use_container_width=True)
    st.bar_chart(view.groupby("loan_purpose").loan_default.mean())
with predict:
    st.warning("This is a hypothetical estimate from a synthetic-data model, not a lending recommendation.")
    with st.form("applicant"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 21, 65, 32)
            income = st.number_input("Annual income (₹)", 120000, 3000000, 600000, step=25000)
            years = st.slider("Employment years", 0.0, 40.0, 4.0, step=.5)
            employment = st.selectbox("Employment type", ["Salaried", "Self-employed", "Contract", "Business"])
            purpose = st.selectbox("Loan purpose", ["Home", "Vehicle", "Education", "Personal", "Business"])
        with c2:
            loan = st.number_input("Loan amount (₹)", 25000, 2000000, 300000, step=10000)
            term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
            interest = st.slider("Interest rate (%)", 5.0, 25.0, 12.0, step=.25)
            history = st.slider("Credit history (years)", 0, 40, 8)
            existing = st.slider("Existing loans", 0, 10, 1)
            missed = st.slider("Missed payments", 0, 10, 0)
        submit = st.form_submit_button("Estimate simulated risk")
    if submit:
        row = pd.DataFrame([{"age": age, "annual_income": income, "employment_years": years,
            "employment_type": employment, "loan_amount": loan, "loan_term_months": term,
            "interest_rate": interest, "credit_history_years": history,
            "existing_loans": existing, "missed_payments": missed, "loan_purpose": purpose}])
        p = float(model.predict_proba(row)[0, 1])
        st.metric("Estimated simulated default probability", f"{p:.1%}")
        st.write("Model class:", "Simulated default" if p >= .5 else "Simulated non-default")
        st.caption("50% is an illustrative cutoff, not a validated business threshold.")
with performance:
    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics.set_index("model")[["accuracy", "precision_default", "recall_default", "f1_default", "roc_auc"]])
    image = ROOT / "outputs" / "plots" / "07_best_model_confusion_matrix.png"
    if image.exists():
        st.image(str(image))
