from __future__ import annotations

"""Credit Risk – Explainable Financial Decision Making.
Single-file project runner. Run with: python main.py

All records are synthetic and for educational use only.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED, N = 42, 5000
ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
PLOTS, REPORTS, MODELS = OUT / "plots", OUT / "reports", ROOT / "models"
for folder in (DATA, PLOTS, REPORTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)


def make_dataset(n=N):
    rng = np.random.default_rng(SEED)
    age = rng.integers(21, 66, n)
    income = np.clip(rng.lognormal(np.log(600000), .45, n), 120000, 3000000).round().astype(int)
    employment_years = np.clip(rng.gamma(2, 3, n), 0, 40).round(1)
    loan = np.clip(income * rng.uniform(.15, 1.5, n), 25000, 2000000).round().astype(int)
    term = rng.choice([12, 24, 36, 48, 60], n, p=[.08, .15, .35, .17, .25])
    interest = np.clip(rng.normal(12, 3, n), 5, 25).round(2)
    history = np.clip(age - rng.integers(18, 35, n), 0, 40)
    existing = rng.poisson(1.2, n)
    missed = rng.poisson(.45, n)
    employment = rng.choice(["Salaried", "Self-employed", "Contract", "Business"], n, p=[.52, .20, .13, .15])
    purpose = rng.choice(["Home", "Vehicle", "Education", "Personal", "Business"], n, p=[.22, .20, .12, .31, .15])
    ratio = loan / np.maximum(income, 1)
    # Fictional rules intentionally create a learnable synthetic outcome.
    logit = (-2.15 + 1.05*(ratio > .75) + .42*(ratio > 1.10) + .48*(interest > 15)
             + .55*(employment_years < 2) + .55*(existing >= 3) + .72*(missed >= 2)
             - .35*(history >= 8))
    prob = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob)
    return pd.DataFrame({
        "applicant_id": [f"APP{i:05d}" for i in range(1, n+1)],
        "age": age, "annual_income": income, "employment_years": employment_years,
        "employment_type": employment, "loan_amount": loan, "loan_term_months": term,
        "interest_rate": interest, "credit_history_years": history,
        "existing_loans": existing, "missed_payments": missed,
        "loan_purpose": purpose, "loan_default": default
    })


def save_plot(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close()


def create_dashboard():
    app = r'''from pathlib import Path
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
'''
    (ROOT / "app.py").write_text(app, encoding="utf-8")


def main():
    print("[1/8] Creating 5,000 fictional records...")
    df = make_dataset()
    df.to_csv(DATA / "credit_risk_dataset.csv", index=False)
    df.to_csv(ROOT / "credit_risk_dataset.csv", index=False)

    print("[2/8] Validating and summarizing data...")
    quality = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df],
        "missing_count": [int(df[c].isna().sum()) for c in df],
        "unique_values": [int(df[c].nunique()) for c in df]})
    quality.to_csv(OUT / "data_quality_report.csv", index=False)
    df.describe(include="all").T.to_csv(OUT / "descriptive_statistics.csv")
    df.loan_default.value_counts().sort_index().rename_axis("loan_default").reset_index(name="count").to_csv(OUT / "target_distribution.csv", index=False)
    (REPORTS / "data_quality_summary.txt").write_text(
        f"Rows: {len(df)}\\nColumns: {len(df.columns)}\\nMissing cells: {int(df.isna().sum().sum())}\\n"
        f"Duplicate rows: {int(df.duplicated().sum())}\\nTarget: 1=simulated default; 0=simulated non-default\\n"
        "All data is fictional. Not for real lending decisions.\\n", encoding="utf-8")

    print("[3/8] Creating EDA charts...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5)); sns.countplot(data=df, x="loan_default")
    plt.title("Target Distribution: Simulated Default"); plt.xlabel("Default (1=yes, 0=no)"); save_plot("01_target_distribution.png")
    plt.figure(figsize=(8, 5)); sns.histplot(data=df, x="annual_income", bins=35, kde=True)
    plt.title("Annual Income Distribution"); plt.xlabel("Annual income (₹)"); save_plot("02_annual_income_distribution.png")
    plt.figure(figsize=(8, 5)); sns.histplot(data=df, x="loan_amount", hue="loan_default", bins=35, element="step", stat="density", common_norm=False)
    plt.title("Loan Amount by Simulated Outcome"); save_plot("03_loan_amount_by_outcome.png")
    plt.figure(figsize=(9, 5)); df.groupby("loan_purpose").loan_default.mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Simulated Default Rate by Loan Purpose"); plt.ylabel("Simulated default share"); save_plot("04_default_rate_by_purpose.png")
    plt.figure(figsize=(8, 5)); sns.boxplot(data=df, x="loan_default", y="interest_rate")
    plt.title("Interest Rate by Simulated Outcome"); save_plot("05_interest_rate_by_outcome.png")

    print("[4/8] Splitting data and training models...")
    features = [c for c in df.columns if c not in ("applicant_id", "loan_default")]
    X, y = df[features], df["loan_default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=SEED, stratify=y)
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1)
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, subsample=.9, colsample_bytree=.9, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except Exception as exc:
        print(f"XGBoost unavailable; skipping: {exc}")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(iterations=250, depth=5, learning_rate=.05, verbose=False, random_seed=SEED, allow_writing_files=False)
    except Exception as exc:
        print(f"CatBoost unavailable; skipping: {exc}")

    results, fitted, predictions = [], {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results.append({"model": name, "accuracy": accuracy_score(y_test, pred),
            "precision_default": precision_score(y_test, pred, zero_division=0),
            "recall_default": recall_score(y_test, pred, zero_division=0),
            "f1_default": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, proba)})
        fitted[name], predictions[name] = pipe, (pred, proba)
        safe = name.lower().replace(" ", "_")
        joblib.dump(pipe, MODELS / f"{safe}.joblib")
        pd.DataFrame({"actual_loan_default": y_test.to_numpy(), "predicted_loan_default": pred,
            "predicted_default_probability": proba}).to_csv(OUT / f"predictions_{safe}.csv", index=False)
        (REPORTS / f"classification_report_{safe}.txt").write_text(
            classification_report(y_test, pred, target_names=["Non-default", "Default"], zero_division=0), encoding="utf-8")

    metrics = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    joblib.dump(best_model, MODELS / "best_model.joblib")
    (OUT / "best_model_summary.json").write_text(json.dumps({"best_model_by_roc_auc": best_name, "metrics": metrics.iloc[0].to_dict()}, indent=2), encoding="utf-8")

    print("[5/8] Creating model evaluation charts...")
    long = metrics.melt(id_vars="model", value_vars=["accuracy", "precision_default", "recall_default", "f1_default", "roc_auc"], var_name="metric", value_name="score")
    plt.figure(figsize=(11, 6)); sns.barplot(data=long, x="metric", y="score", hue="model"); plt.ylim(0, 1)
    plt.title("Model Performance Comparison"); plt.xticks(rotation=20, ha="right"); save_plot("06_model_metrics_comparison.png")
    pred, proba = predictions[best_name]
    plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Non-default", "Default"], yticklabels=["Non-default", "Default"])
    plt.title(f"Confusion Matrix — {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); save_plot("07_best_model_confusion_matrix.png")
    plt.figure(figsize=(8, 6))
    for name, (_, p) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, p); plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC Curves"); plt.legend(); save_plot("08_roc_curves.png")
    plt.figure(figsize=(8, 6))
    for name, (_, p) in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, p); plt.plot(recall, precision, label=name)
    plt.xlabel("Recall for default"); plt.ylabel("Precision for default"); plt.title("Precision–Recall Curves"); plt.legend(); save_plot("09_precision_recall_curves.png")
    X_train.assign(loan_default=y_train.to_numpy()).to_csv(OUT / "train_set.csv", index=False)
    X_test.assign(loan_default=y_test.to_numpy()).to_csv(OUT / "test_set.csv", index=False)

    print("[6/8] Calculating feature importance and attempting SHAP...")
    try:
        perm = permutation_importance(best_model, X_test, y_test, n_repeats=4, random_state=SEED, scoring="roc_auc", n_jobs=-1)
        imp = pd.DataFrame({"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
        imp.to_csv(OUT / "permutation_feature_importance.csv", index=False)
        top = imp.head(12).sort_values("importance_mean")
        plt.figure(figsize=(9, 6)); plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        plt.title(f"Permutation Feature Importance — {best_name}"); plt.xlabel("Decrease in ROC-AUC"); save_plot("10_permutation_feature_importance.png")
    except Exception as exc:
        print(f"Permutation importance skipped: {exc}")
    try:
        import shap
        sample = X_test.sample(min(80, len(X_test)), random_state=SEED)
        transformed = best_model.named_steps["preprocess"].transform(sample)
        explainer = shap.Explainer(best_model.named_steps["model"], transformed)
        values = explainer(transformed)
        names = best_model.named_steps["preprocess"].get_feature_names_out()
        plt.figure()
        shap.summary_plot(values.values, transformed, feature_names=names, show=False, max_display=15)
        save_plot("11_shap_summary.png")
        pd.DataFrame({"feature": names, "mean_absolute_shap": np.abs(values.values).mean(axis=0)}).sort_values("mean_absolute_shap", ascending=False).to_csv(OUT / "shap_feature_importance.csv", index=False)
    except Exception as exc:
        (REPORTS / "shap_status.txt").write_text(f"SHAP could not be generated in this environment: {exc}\\nInstall compatible SHAP dependencies and rerun.\\n", encoding="utf-8")
        print(f"SHAP skipped safely: {exc}")

    print("[7/8] Generating dashboard, requirements and README...")
    create_dashboard()
    (ROOT / "requirements.txt").write_text("pandas\\nnumpy\\nmatplotlib\\nseaborn\\nscikit-learn\\njoblib\\nxgboost\\ncatboost\\nshap\\nstreamlit\\ntabulate\\n", encoding="utf-8")
    (ROOT / ".gitignore").write_text(".venv/\\n__pycache__/\\n*.pyc\\n.DS_Store\\n.vscode/\\n", encoding="utf-8")
    table = metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    readme = f'''# Credit Risk – Explainable Financial Decision Making

An educational end-to-end ML project using a **synthetic fictional loan dataset**.

> **Important:** The data and labels are simulated, not real customer records. Scores reflect the artificial rules used to create the data and are not evidence of real-world lending performance. Do not use this project for actual loan decisions.

## Workflow
1. Generate 5,000 fictional applicant records.
2. Validate data and create quality reports.
3. Perform EDA with Pandas, Matplotlib and Seaborn.
4. Split training and test data before preprocessing.
5. Compare Logistic Regression, Random Forest, XGBoost and CatBoost when installed.
6. Evaluate accuracy, precision, recall, F1 and ROC-AUC.
7. Save predictions, models, charts and reports.
8. Attempt SHAP explanations.
9. Explore the Streamlit dashboard.

## Target
`loan_default`: 1 = simulated default, 0 = simulated non-default.

## Model comparison
```text
{table}
```
Best model by holdout ROC-AUC: **{best_name}**.

## Visualizations
![Target distribution](outputs/plots/01_target_distribution.png)
![Income distribution](outputs/plots/02_annual_income_distribution.png)
![Loan amount](outputs/plots/03_loan_amount_by_outcome.png)
![Purpose rates](outputs/plots/04_default_rate_by_purpose.png)
![Model comparison](outputs/plots/06_model_metrics_comparison.png)
![Confusion matrix](outputs/plots/07_best_model_confusion_matrix.png)
![ROC curves](outputs/plots/08_roc_curves.png)
![Precision-recall curves](outputs/plots/09_precision_recall_curves.png)
![Permutation importance](outputs/plots/10_permutation_feature_importance.png)

## Structure
```text
Credit_risk/
├── main.py
├── app.py
├── requirements.txt
├── README.md
├── data/credit_risk_dataset.csv
├── models/*.joblib
└── outputs/
    ├── plots/*.png
    ├── reports/*
    ├── model_comparison.csv
    ├── predictions_*.csv
    └── train_set.csv / test_set.csv
```

## Run
```bash
python -m pip install -r requirements.txt
python main.py
streamlit run app.py
```

## Limitations
Synthetic data is designed from illustrative assumptions; metrics can be optimistic. No claim of fairness, causality, calibration, or real-world validity is made. The 50% classification cutoff in the dashboard is illustrative, not a validated policy.
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    print("[8/8] Verifying outputs...")
    verification = {
        "rows": len(df), "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()),
        "models_trained": list(fitted.keys()), "best_model_by_roc_auc": best_name,
        "plots": len(list(PLOTS.glob("*.png"))), "csv_outputs": len(list(OUT.glob("*.csv"))),
        "model_files": len(list(MODELS.glob("*.joblib"))), "simulated_default_rate": float(df.loan_default.mean())
    }
    (REPORTS / "build_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    print("\\nBUILD FINISHED. Dashboard command: streamlit run app.py")
    print("Reminder: synthetic-data educational demo only.")


if __name__ == "__main__":
    main()
