# Online Shopper Purchase Prediction

## a. Problem Statement

Most visitors to an online store browse and leave without buying anything. If a business can tell, from how a visitor behaves during a session, whether that session is likely to end in a purchase, it can act on it: show a discount, offer live chat, or retarget the visitor later.

The goal of this project is to predict whether an online shopping session ends in a purchase (`Revenue` = 1) or not (`Revenue` = 0), using information about the pages visited, time spent, Google Analytics metrics, and visitor details. This is a **binary classification** problem. Five classification models are trained on the same data, compared on six evaluation metrics, and demonstrated through an interactive Streamlit web app.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository – Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018)
- **Instances:** 12,330 sessions (12,205 after removing 125 duplicate rows)
- **Features:** 17 input features
  - 10 numerical: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
  - 7 categorical: `Month`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`
- **Target:** `Revenue` (1 = purchase, 0 = no purchase)
- **Class balance:** about 84.5% no purchase and 15.5% purchase, so the data is imbalanced
- **Missing values:** none

### Preprocessing

- Removed 125 duplicate rows so identical sessions could not appear in both the training and test sets.
- Treated `OperatingSystems`, `Browser`, `Region` and `TrafficType` as categorical, because they are ID codes rather than quantities.
- One-hot encoded all categorical columns and standardised the numerical columns with `StandardScaler`.
- Split the data 80/20 into training (9,764 rows) and test (2,441 rows) sets, stratified on `Revenue` to keep the same class balance in both.
- Built the preprocessing and each model into a single scikit-learn `Pipeline`, so the saved model files include the preprocessing and the app can use raw test data directly.
- Saved the test split as `test_data.csv` for use in the Streamlit app.

## c. Links

- **GitHub Repository:** https://github.com/Parth-Prajapati-ml/ml-classification-models-streamlit
- **Live Streamlit App:** [paste your .streamlit.app link here]

## d. Models Used

1. Logistic Regression (`max_iter=1000`)
2. Decision Tree Classifier (`max_depth=6`)
3. K-Nearest Neighbors (`n_neighbors=7`)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, `probability=True`)

All models were trained on the same training set and evaluated on the same 2,441-row test set.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.887 | [AUC] | 0.752 | 0.414 | 0.534 | 0.503 |
| Decision Tree | 0.900 | 0.916 | 0.709 | 0.607 | 0.654 | 0.599 |
| KNN | 0.876 | [AUC] | 0.687 | 0.385 | 0.493 | 0.453 |
| Naive Bayes | 0.277 | [AUC] | 0.176 | 0.979 | 0.298 | 0.138 |
| Support Vector Machine | 0.896 | [AUC] | 0.736 | 0.518 | 0.608 | 0.561 |

Precision, Recall, F1 and MCC refer to the positive class (purchase).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High accuracy (0.887) and the best precision (0.752), so when it predicts a purchase it is usually right. But recall is only 0.414: it misses more than half of the real buyers. As a linear model it cannot easily capture threshold effects such as a sharp jump in purchase rate once `PageValues` is above zero, and the class imbalance pushes it towards predicting "no purchase". |
| Decision Tree | Best overall balance: highest recall (0.607) among the reasonable models, highest F1 (0.654) and highest MCC (0.599), while keeping precision at 0.709. Trees split on thresholds, which suits features like `PageValues` and `ExitRates`, and they are not affected by skewed distributions. Limiting depth to 6 prevented overfitting. |
| KNN | Weakest of the non-Naive-Bayes models (F1 0.493, MCC 0.453) and the lowest recall (0.385). With 85% of neighbours belonging to the "no purchase" class, a buyer's nearest neighbours are often non-buyers, so the majority vote favours the majority class. The many one-hot columns also make distances less meaningful. |
| Naive Bayes | Predicts almost every session as a purchase: recall is 0.979 but precision is only 0.176 and accuracy drops to 0.277. Its assumptions do not hold here: the duration and page features are heavily skewed rather than normally distributed, the one-hot columns are binary, and features such as `BounceRates` and `ExitRates` are strongly correlated rather than independent. Its MCC (0.138) is close to random. |
| Support Vector Machine | Second best overall (F1 0.608, MCC 0.561) with good precision (0.736) and moderate recall (0.518). The RBF kernel captures non-linear patterns better than Logistic Regression, but like the other models it is still biased towards the majority class. |
| **Overall winner** | **Decision Tree.** Because only about 15% of sessions are purchases, accuracy is misleading: a model predicting "no purchase" every time would still score about 84%. Judged on MCC and F1, which account for both classes, the Decision Tree performs best, catching the most buyers while keeping false alarms low. |

## Repository Structure

```
ml-classification-models-streamlit/
├── app.py                  # Streamlit web app
├── Model_Training.ipynb    # EDA, preprocessing, training and evaluation
├── requirements.txt        # Python dependencies
├── README.md
├── test_data.csv           # Test split used for evaluation in the app
├── data/
│   └── online_shoppers_intention.csv   # Original dataset
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── model_comparison.csv
```

## How to Run Locally

1. Clone the repository:
   ```
   git clone https://github.com/Parth-Prajapati-ml/ml-classification-models-streamlit.git
   cd ml-classification-models-streamlit
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```
4. In the browser, upload `test_data.csv`, choose a model from the sidebar, and view the metrics, confusion matrix, classification report and model comparison.

To retrain the models, open `Model_Training.ipynb` and run all cells. This regenerates the files in `model/` and `test_data.csv`.