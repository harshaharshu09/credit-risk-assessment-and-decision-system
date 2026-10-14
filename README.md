# Credit Risk Assessment & Decision Support System

An Explainable Machine Learning system for credit-risk analysis, simulated loan-default prediction, model evaluation, and interactive financial decision support.

> **Responsible Use Notice:** This project uses a fully synthetic dataset and simulated default labels. The generated probabilities, metrics, and risk estimates are intended for analytical and software-demonstration purposes only and are not validated for real-world lending, underwriting, or financial decisions.

---

## Overview

The **Credit Risk Assessment & Decision Support System** is an end-to-end machine learning solution that analyzes synthetic borrower and loan information to identify patterns associated with simulated loan default.

The system combines data validation, exploratory analysis, feature preparation, classification modelling, model comparison, explainable AI, visualization, reporting, and an interactive Streamlit dashboard.

The implementation is designed around a reproducible workflow in which trained model artifacts and analytical outputs are generated separately and then consumed by the dashboard for portfolio exploration and applicant-level what-if analysis.

---

## Core Capabilities

- Synthetic credit-risk dataset generation and validation
- Data-quality inspection and exploratory analysis
- Feature preparation and train/test separation
- Multi-model classification and performance comparison
- Default-probability estimation
- Holdout model evaluation
- Confusion-matrix analysis
- ROC and Precision-Recall analysis
- Permutation feature importance
- SHAP-based model explainability
- Reusable serialized model artifacts
- Interactive Streamlit credit-risk dashboard
- CSV report generation and downloadable outputs

---

## Project Workflow

```text
Synthetic Credit Dataset
        ↓
Data Validation & Quality Checks
        ↓
Exploratory Data Analysis
        ↓
Feature Preparation
        ↓
Train / Test Split
        ↓
Classification Model Training
        ↓
Model Performance Comparison
        ↓
Best Model Selection
        ↓
Default Probability Prediction
        ↓
Model Evaluation
        ↓
Feature Importance Analysis
        ↓
SHAP Explainability
        ↓
Reports & Visualizations
        ↓
Streamlit Credit Risk Dashboard
```
## Dataset
The system operates on a synthetic credit-risk dataset containing fictional applicant and loan records.
The primary target variable is:
loan_default

Target encoding:
0 → Simulated Non-Default
1 → Simulated Default

The dataset contains financial and applicant-level attributes used by the modelling pipeline, including variables such as age, annual income, employment information, loan amount, loan term, interest rate, credit history, existing loans, missed payments, and loan purpose.
All records and labels are synthetic and are generated for project demonstration purposes.

## Data Quality & Preprocessing
The data pipeline performs structured validation and preparation before model training.
The workflow includes:
- Dataset inspection
- Data-type validation
- Missing-value checks
- Duplicate checks
- Target validation
- Feature preparation
- Categorical-data handling
- Train/test separation
- Model-ready feature preparation
The training workflow keeps the test data separate so that model evaluation is performed on unseen records.
Exploratory Data Analysis
Exploratory Data Analysis is used to understand the structure and behaviour of the synthetic credit portfolio before modelling.
The analysis covers:
- Default distribution
- Income distribution
- Loan amount behaviour
- Correlation analysis
- Loan-purpose default patterns
- Applicant and loan characteristics
- Model-performance visualizations
- Feature-importance analysis
The Streamlit dashboard also provides interactive dataset exploration, descriptive statistics, loan-purpose filtering, and default-rate analysis. 

## Machine Learning Models
The project implements and compares four classification approaches:
Model	Role
Logistic Regression	Interpretable baseline classifier
Random Forest	Ensemble tree-based classifier
XGBoost	Gradient-boosting classifier
CatBoost	Gradient-boosting classifier for structured/tabular data


The models are evaluated using the same prepared modelling workflow so that their performance can be compared consistently.
The implementation saves individual trained model artifacts together with a selected best_model.joblib artifact for downstream application use

## Model Evaluation
Model performance is evaluated using multiple classification metrics rather than relying on accuracy alone.
The evaluation framework includes:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
The Streamlit Model Performance workspace loads the generated outputs/model_comparison.csv file and presents the available model metrics interactively.

## Model Selection
After model training and evaluation, the selected model is persisted as:
models/best_model.joblib

The dashboard loads this saved artifact for applicant-level what-if prediction rather than retraining the model during every application session.

## Risk Prediction
The dashboard provides an Applicant Estimate workflow for hypothetical applicant profiles.
The prediction interface accepts model features including:
- Age
- Annual Income
- Employment Years
- Employment Type
- Loan Purpose
- Loan Amount
- Loan Term
- Interest Rate
- Credit History
- Existing Loans
- Missed Payments
The trained model produces an estimated simulated default probability.
The application then applies an illustrative 50% classification cutoff to produce a simulated default/non-default class. This cutoff is explicitly treated as a demonstration threshold rather than a validated lending policy.

## Explainable AI
SHAP Analysis
The project integrates SHAP (SHapley Additive exPlanations) to provide model-level interpretability.
SHAP is used to examine how input features contribute to model predictions and to provide a clearer view of model behaviour.
The generated SHAP summary is surfaced by the dashboard when the corresponding output is available.

## Permutation Feature Importance
Permutation feature importance is used to assess the effect of individual features on model performance.
This provides an additional model-agnostic perspective alongside SHAP-based analysis.

## Visual Analysis
The project generates PNG visualizations during exploratory analysis and model evaluation.
The dashboard automatically discovers and displays PNG files available inside:
outputs/plots/

The current application includes visual analysis for:
- Target distribution
- Income distribution
- Correlation analysis
- Model metrics comparison
- Best-model confusion matrix
- ROC curves
- Precision-Recall curves
- Permutation feature importance
- SHAP summary
The same generated PNG outputs are also made available for download through the dashboard.

Target Distribution

Income Distribution

Correlation Heatmap

Model Metrics Comparison

Best Model Confusion Matrix

ROC Curves

Precision-Recall Curves

Permutation Feature Importance

SHAP Summary

## Streamlit Credit Risk Dashboard
The project includes an interactive Credit Risk Analytics application developed with Streamlit.
The dashboard is organized into six functional workspaces:
Workspace	Function
Portfolio Overview	Portfolio-level KPIs, dataset preview, and default distribution
Explore Data	Dataset filtering, descriptive statistics, and segment analysis
Applicant Estimate	Hypothetical applicant risk estimation
Data Visualizations	Interactive access to generated PNG analysis
Model Performance	Model metrics, confusion matrix, ROC and Precision-Recall analysis
Reports & Downloads	Downloadable CSV and PNG project outputs
The application loads the saved dataset, model artifact, model-comparison metrics, and generated visual outputs directly from the project directories.

## Dashboard Workflow
Applicant Input
      ↓
Feature Preparation
      ↓
Saved Best Model
      ↓
Default Probability
      ↓
Simulated Classification
      ↓
Dashboard Result

The dashboard also provides access to portfolio-level analysis and generated project artifacts.

## Project Outputs
The project produces machine-learning and analytical artifacts under the outputs/ directory.
Typical outputs include:
outputs/
├── model_comparison.csv
├── predictions_*.csv
├── train_set.csv
├── test_set.csv
└── plots/
    ├── 01_target_distribution.png
    ├── 02_income_distribution.png
    ├── 03_correlation_heatmap.png
    ├── 06_model_metrics_comparison.png
    ├── 07_best_model_confusion_matrix.png
    ├── 08_roc_curves.png
    ├── 09_precision_recall_curves.png
    ├── 10_permutation_feature_importance.png
    └── 11_shap_summary.png

The training workflow also generates data-quality and model-related reports, while the dashboard exposes available CSV and PNG artifacts for download.

## Model Artifacts
Trained models are stored under:
models/

The project maintains reusable model artifacts for the trained classifiers and a selected best-model artifact:
models/
├── logistic_regression.joblib
├── random_forest.joblib
├── xgboost.joblib
├── catboost.joblib
└── best_model.joblib

These artifacts allow the Streamlit application to perform inference using the saved model without retraining the complete workflow on every application launch.

## Technology Stack
Area	Technology
Programming Language	Python
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn
Gradient Boosting	XGBoost, CatBoost
Explainable AI	SHAP
Visualization	Matplotlib, Seaborn
Model Persistence	Joblib
Interactive Application	Streamlit
Development Environment	VS Code
Version Control	Git
Repository Hosting	GitHub


The project dependency configuration includes the core data-science, machine-learning, explainability, visualization, model-persistence, and Streamlit components used by the implementation.

## Repository Structure
Credit_risk/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── catboost.joblib
│   └── best_model.joblib
│
├── outputs/
│   ├── model_comparison.csv
│   ├── predictions_*.csv
│   ├── train_set.csv
│   ├── test_set.csv
│   ├── plots/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_income_distribution.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 06_model_metrics_comparison.png
│   │   ├── 07_best_model_confusion_matrix.png
│   │   ├── 08_roc_curves.png
│   │   ├── 09_precision_recall_curves.png
│   │   ├── 10_permutation_feature_importance.png
│   │   └── 11_shap_summary.png
│   │
│   └── reports/
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
## Installation
Clone the repository:
git clone https://github.com/harshaharshu09/credit-risk-assessment-and-decision-system.git

Move into the project directory:
cd credit-risk-assessment-and-decision-system

Install the required dependencies:
python -m pip install -r requirements.txt

## Run the Machine Learning Pipeline
Execute the main project workflow:
python main.py

The pipeline generates the dataset-related outputs, model artifacts, evaluation results, visualizations, reports, and Streamlit application resources.

## Launch the Dashboard
Start the interactive application with:
streamlit run app.py

The Streamlit interface provides portfolio analysis, dataset exploration, applicant-level estimation, visualization, model-performance analysis, and downloadable project outputs.

## End-to-End Architecture
                    ┌──────────────────────────┐
                    │ Synthetic Credit Dataset │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Quality Validation   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Exploratory Data Analysis│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Feature Preparation      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Train / Test Split       │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       Logistic Regression   Random Forest      XGBoost
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                             CatBoost
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Model Comparison         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Best Model Selection     │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
          Risk Prediction              Explainable AI
                  │                             │
                  │                         SHAP Analysis
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Reports & Visualizations │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Streamlit Dashboard      │
                    └──────────────────────────┘

## Evaluation Metrics
The project uses the following evaluation measures:
Accuracy
Overall proportion of correctly classified test records.
Precision
Proportion of predicted default cases that correspond to simulated default labels.
Recall
Proportion of simulated default cases identified by the model.
F1 Score
Harmonic mean of precision and recall.
ROC-AUC
Measures the model's ability to rank positive cases above negative cases across classification thresholds.
Confusion Matrix
Provides a direct view of true positives, true negatives, false positives, and false negatives.

## Reproducibility
The project separates data preparation, model training, evaluation, output generation, and application presentation.
The generated model artifacts and analytical outputs are saved to dedicated directories so that the Streamlit dashboard can reuse them without retraining the complete pipeline.
The verification stage also records information such as:
- Dataset row count
- Dataset column count
- Missing-cell count
- Trained model names
- Best model by ROC-AUC
- Number of generated plots
- Number of generated CSV outputs
- Number of saved model files
- Simulated default rate
These verification details are generated by the project build workflow.

## Responsible Use & Limitations
This project is intentionally based on synthetic data.
Therefore:
- Model metrics describe performance against simulated labels.
- Results do not represent real borrower behaviour.
- Predictions are estimates produced by the trained model.
- SHAP explanations describe model behaviour and do not establish causality.
- The illustrative 50% classification cutoff is not a validated credit policy.
- No production lending, regulatory, fairness, calibration, or real-world performance claim is made.
The system is designed as a technical demonstration of an explainable credit-risk analytics workflow rather than an automated loan-approval engine.

## Conclusion
The Credit Risk Assessment & Decision Support System provides a complete machine-learning workflow for synthetic credit-risk analysis.
It integrates:
Data
  ↓
Validation
  ↓
EDA
  ↓
Feature Preparation
  ↓
Model Training
  ↓
Model Comparison
  ↓
Risk Prediction
  ↓
Model Evaluation
  ↓
Feature Importance
  ↓
SHAP Explainability
  ↓
Reporting
  ↓
Interactive Streamlit Dashboard

The result is a structured, explainable, and reproducible credit-risk analytics implementation that demonstrates how machine learning outputs can be transformed into an interactive decision-support experience.

## Author
Harsha Vardhan kumar
B.Tech — Computer Science & Engineering (AIML)
GitHub: https://github.com/harshaharshu09