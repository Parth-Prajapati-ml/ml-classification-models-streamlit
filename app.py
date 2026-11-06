
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Online Shopper Prediction", layout="wide")
st.title("Online Shopper Purchase Prediction")
st.write("Upload your test dataset and check the performance of different ML models.")

models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl"
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    results = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions)
    }
    return results, predictions

st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Select ML Model", list(models.keys()))
st.sidebar.markdown("---")
st.sidebar.write("Dataset: UCI Online Shoppers Purchasing Intention")
st.sidebar.write("Target: Revenue (1 = Purchase, 0 = No Purchase)")

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as file:
        st.sidebar.download_button("Download Test Data", file, file_name="test_data.csv")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Revenue" not in df.columns:
        st.error("Revenue column is missing.")
        st.stop()

    y = df["Revenue"]

    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)
    else:
        y = y.astype(int)

    model = load_model(models[selected_model])
    features = list(model.feature_names_in_)
    missing_columns = [col for col in features if col not in df.columns]

    if missing_columns:
        st.error("Missing columns: " + ", ".join(missing_columns))
        st.stop()

    X = df[features]

    st.write(f"Total Sessions: {len(df)}")
    st.write(f"Purchase Sessions: {y.sum()}")

    st.subheader(f"Results: {selected_model}")
    scores, predictions = evaluate_model(model, X, y)

    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, round(value, 3))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Purchase", "Purchase"], yticklabels=["No Purchase", "Purchase"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Classification Report")
        report = classification_report(y, predictions, target_names=["No Purchase", "Purchase"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

    st.markdown("---")
    st.subheader("Model Comparison")
    comparison = {}

    for name, path in models.items():
        ml_model = load_model(path)
        results, _ = evaluate_model(ml_model, X, y)
        comparison[name] = results

    comparison_df = pd.DataFrame(comparison).T.round(3)
    st.dataframe(comparison_df.style.highlight_max(axis=0, color="#cfe8ff"))

    selected_metric = st.selectbox("Select Metric", comparison_df.columns, index=5)
    st.bar_chart(comparison_df[selected_metric])

else:
    st.info("Please upload your test_data.csv file to view the results.")