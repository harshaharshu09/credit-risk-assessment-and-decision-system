
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "credit_risk_dataset.csv"
MODEL = ROOT / "models" / "best_model.joblib"
METRICS = ROOT / "outputs" / "model_comparison.csv"
OUTPUTS = ROOT / "outputs"
PLOTS = OUTPUTS / "plots"

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Credit Risk – Explainable Financial Decision Making")
st.caption(
    "Educational demo using synthetic data only. "
    "Not validated for real lending decisions."
)

# ---------- Load project files ----------
if not DATA.exists():
    st.error(f"Dataset not found: {DATA}")
    st.stop()

df = pd.read_csv(DATA)

metrics = pd.read_csv(METRICS) if METRICS.exists() else pd.DataFrame()

model = None
if MODEL.exists():
    try:
        model = joblib.load(MODEL)
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")

# ---------- Helpers ----------
def show_plot(filename, caption=None):
    """Display a saved plot if it exists."""
    path = PLOTS / filename
    if path.exists():
        st.image(str(path), caption=caption or filename, width="stretch")
        return True
    return False


def show_all_plots():
    """Display every PNG file currently in outputs/plots."""
    if not PLOTS.exists():
        st.info("The outputs/plots folder does not exist yet.")
        return

    images = sorted(PLOTS.glob("*.png"))

    if not images:
        st.info("No PNG plots found in outputs/plots.")
        return

    st.write(f"**{len(images)} plot images available**")

    for start in range(0, len(images), 2):
        columns = st.columns(2)

        for col, image_path in zip(columns, images[start:start + 2]):
            with col:
                st.image(
                    str(image_path),
                    caption=image_path.stem.replace("_", " ").title(),
                    width="stretch",
                )


def show_csv_downloads():
    """Provide downloads for available CSV files."""
    csv_files = []

    if DATA.exists():
        csv_files.append(DATA)

    if OUTPUTS.exists():
        csv_files.extend(sorted(OUTPUTS.glob("*.csv")))

    # Remove duplicate paths while preserving order.
    unique_files = list(dict.fromkeys(csv_files))

    if not unique_files:
        st.info("No CSV files found.")
        return

    for file_path in unique_files:
        with st.expander(file_path.name):
            st.download_button(
                label=f"Download {file_path.name}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                mime="text/csv",
                key=f"download_{file_path.name}",
            )


# ---------- Dashboard navigation ----------
overview, explore, predict, visuals, performance, reports = st.tabs(
    [
        "Portfolio Overview",
        "Explore Data",
        "Applicant Estimate",
        "Data Visualizations",
        "Model Performance",
        "Reports & Downloads",
    ]
)

# ---------- 1. Portfolio Overview ----------
with overview:
    st.subheader("Portfolio Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Synthetic Applicants", f"{len(df):,}")

    if "loan_default" in df.columns:
        default_rate = df["loan_default"].mean()
        c2.metric("Simulated Default Rate", f"{default_rate:.1%}")
    else:
        c2.metric("Simulated Default Rate", "N/A")

    if "annual_income" in df.columns:
        c3.metric(
            "Median Annual Income",
            f"₹{df['annual_income'].median():,.0f}",
        )
    else:
        c3.metric("Median Annual Income", "N/A")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), width="stretch")

    st.markdown("### Default Distribution")

    if "loan_default" in df.columns:
        distribution = (
            df["loan_default"]
            .value_counts()
            .sort_index()
            .rename(index={0: "Non-default", 1: "Default"})
        )
        st.bar_chart(distribution)
    else:
        st.info("The loan_default target column was not found.")

    st.markdown("### Key Visualizations")
    for filename in [
        "01_target_distribution.png",
        "02_income_distribution.png",
        "03_correlation_heatmap.png",
        "06_model_metrics_comparison.png",
    ]:
        show_plot(filename)

# ---------- 2. Explore Data ----------
with explore:
    st.subheader("Explore the Synthetic Dataset")

    if "loan_purpose" in df.columns:
        purposes = ["All"] + sorted(
            df["loan_purpose"].dropna().astype(str).unique().tolist()
        )
        selected_purpose = st.selectbox("Loan Purpose", purposes)

        if selected_purpose == "All":
            view = df
        else:
            view = df[df["loan_purpose"] == selected_purpose]
    else:
        view = df

    st.write(f"Rows displayed: {len(view):,}")
    st.dataframe(view, width="stretch")

    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, width="stretch")

    if (
        "loan_purpose" in view.columns
        and "loan_default" in view.columns
    ):
        purpose_rates = view.groupby("loan_purpose")[
            "loan_default"
        ].mean()
        st.markdown("### Default Rate by Loan Purpose")
        st.bar_chart(purpose_rates)

# ---------- 3. Applicant Estimate ----------
with predict:
    st.subheader("Simulated Applicant Risk Estimate")

    st.warning(
        "This is an educational estimate from a synthetic-data model. "
        "It is not a loan approval or rejection recommendation."
    )

    if model is None:
        st.error("The saved best model could not be loaded.")
    else:
        with st.form("applicant_form"):
            left, right = st.columns(2)

            with left:
                age = st.slider("Age", 21, 65, 32)
                income = st.number_input(
                    "Annual Income (₹)",
                    min_value=120000,
                    max_value=3000000,
                    value=600000,
                    step=25000,
                )
                years = st.slider(
                    "Employment Years", 0.0, 40.0, 4.0, step=0.5
                )
                employment = st.selectbox(
                    "Employment Type",
                    ["Salaried", "Self-employed", "Contract", "Business"],
                )
                purpose = st.selectbox(
                    "Loan Purpose",
                    ["Home", "Vehicle", "Education", "Personal", "Business"],
                )

            with right:
                loan = st.number_input(
                    "Loan Amount (₹)",
                    min_value=25000,
                    max_value=2000000,
                    value=300000,
                    step=10000,
                )
                term = st.selectbox(
                    "Loan Term (Months)", [12, 24, 36, 48, 60], index=2
                )
                interest = st.slider(
                    "Interest Rate (%)", 5.0, 25.0, 12.0, step=0.25
                )
                history = st.slider("Credit History (Years)", 0, 40, 8)
                existing = st.slider("Existing Loans", 0, 10, 1)
                missed = st.slider("Missed Payments", 0, 10, 0)

            submitted = st.form_submit_button(
                "Estimate Simulated Risk",
                width="stretch",
            )

        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "age": age,
                        "annual_income": income,
                        "employment_years": years,
                        "employment_type": employment,
                        "loan_amount": loan,
                        "loan_term_months": term,
                        "interest_rate": interest,
                        "credit_history_years": history,
                        "existing_loans": existing,
                        "missed_payments": missed,
                        "loan_purpose": purpose,
                    }
                ]
            )

            try:
                probability = float(model.predict_proba(row)[0, 1])
                prediction = int(probability >= 0.5)

                st.metric(
                    "Estimated Simulated Default Probability",
                    f"{probability:.1%}",
                )

                st.write(
                    "Model class:",
                    "Simulated default" if prediction else "Simulated non-default",
                )

                st.caption(
                    "The 50% cutoff is illustrative, not a validated "
                    "business threshold. Model output is not a real-world "
                    "creditworthiness assessment."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "Check that the form fields match the features used "
                    "when training the saved model."
                )

# ---------- 4. Data Visualizations ----------
with visuals:
    st.subheader("Exploratory Data Analysis & Charts")
    st.write(
        "Browse the PNG plots generated by the project pipeline."
    )

    show_all_plots()

# ---------- 5. Model Performance ----------
with performance:
    st.subheader("Model Comparison")

    if metrics.empty:
        st.warning(
            "Model metrics file is missing or empty. "
            "Check outputs/model_comparison.csv."
        )
    else:
        st.dataframe(metrics, width="stretch")

        metric_columns = [
            col
            for col in [
                "accuracy",
                "precision_default",
                "recall_default",
                "f1_default",
                "roc_auc",
            ]
            if col in metrics.columns
        ]

        if "model" in metrics.columns and metric_columns:
            chart_data = metrics.set_index("model")[metric_columns]
            st.markdown("### Metrics Comparison")
            st.bar_chart(chart_data)

    st.divider()
    st.subheader("Confusion Matrix")
    show_plot("07_best_model_confusion_matrix.png")

    st.subheader("ROC Curves")
    show_plot("08_roc_curves.png")

    st.subheader("Precision-Recall Curves")
    show_plot("09_precision_recall_curves.png")

    st.subheader("Feature Importance")
    show_plot("10_permutation_feature_importance.png")

    st.subheader("SHAP Explainability")
    if not show_plot("11_shap_summary.png"):
        st.info(
            "SHAP summary image was not found. "
            "Check the SHAP generation report."
        )

# ---------- 6. Reports & Downloads ----------
with reports:
    st.subheader("Download Project Outputs")
    st.write(
        "Download the dataset and generated CSV reports for further analysis."
    )

    show_csv_downloads()

    st.divider()
    st.subheader("Available PNG Images")

    if PLOTS.exists():
        images = sorted(PLOTS.glob("*.png"))

        if images:
            for image_path in images:
                st.download_button(
                    label=f"Download {image_path.name}",
                    data=image_path.read_bytes(),
                    file_name=image_path.name,
                    mime="image/png",
                    key=f"image_download_{image_path.name}",
                )
        else:
            st.info("No plot images are available to download.")
    else:
        st.info("The plots folder does not exist.")

st.divider()
st.caption(
    "Credit Risk Analytics | Synthetic educational project | "
    "Results are illustrative and not validated for real lending."
)
