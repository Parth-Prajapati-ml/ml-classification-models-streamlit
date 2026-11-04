import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

# ---------------- Page setup ----------------
st.set_page_config(page_title="Online Shopper Purchase Prediction", layout="wide")
st.title("Online Shopper Purchase Prediction")
st.write(
    "Will a website visitor make a purchase? Upload the test data to see how "
    "five pre-trained classifiers perform on it. The models are loaded from "
    "saved files and are not retrained here."
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl",
}
TARGET = "Revenue"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def to_binary(series):
    """Convert the target to 0/1, whether it is stored as 0/1 or True/False."""
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)
    return series.astype(int)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return scores, y_pred


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("About the data")
st.sidebar.write(
    "UCI Online Shoppers Purchasing Intention dataset: one row per browsing "
    "session, 17 features, and the target `Revenue` (1 = purchase). "
    "Only about 15% of sessions end in a purchase, so look at Recall, F1 "
    "and MCC, not just Accuracy."
)

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button(
            "Download test_data.csv", f, file_name="test_data.csv", mime="text/csv"
        )

# ---------------- Main area ----------------
uploaded = st.file_uploader(
    f"Upload test CSV (all feature columns plus the '{TARGET}' column)", type="csv"
)

if uploaded is None:
    st.info("Upload test_data.csv to see results. You can download it from the sidebar.")
    st.stop()

df = pd.read_csv(uploaded)

if TARGET not in df.columns:
    st.error(f"The CSV is missing the '{TARGET}' column. Add it and upload again.")
    st.stop()

model = load_model(MODEL_FILES[model_name])

# Check the uploaded file has every column the model was trained on
expected_cols = list(getattr(model, "feature_names_in_", []))
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error("These feature columns are missing from the CSV: " + ", ".join(missing))
    st.stop()

X = df[expected_cols] if expected_cols else df.drop(columns=[TARGET])
y = to_binary(df[TARGET])

st.caption(f"Loaded {len(df):,} rows. Purchases in this file: {y.mean():.1%}")

# ---- Selected model ----
st.subheader(f"Results: {model_name}")
scores, y_pred = compute_metrics(model, X, y)

cols = st.columns(6)
for col, (name, value) in zip(cols, scores.items()):
    col.metric(name, f"{value:.3f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y, y_pred, target_names=["No purchase", "Purchase"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(3))

# ---- All models ----
st.markdown("---")
st.subheader("Compare all five models on this file")

rows = {}
for name, path in MODEL_FILES.items():
    m = load_model(path)
    rows[name], _ = compute_metrics(m, X, y)

comparison = pd.DataFrame(rows).T.round(3)
st.dataframe(comparison.style.highlight_max(axis=0, color="#cfe8ff"))

metric_to_plot = st.selectbox("Metric to chart", list(comparison.columns), index=5)
st.bar_chart(comparison[metric_to_plot])