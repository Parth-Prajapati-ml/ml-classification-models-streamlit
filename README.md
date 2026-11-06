# Online Shopper Purchase Prediction

## a. Problem Statement
Many people visit online shopping websites, browse different products, and leave without making a purchase. If we can predict whether a visitor is likely to buy something based on their browsing activity, businesses can take steps to encourage them to complete the purchase. For example, they can offer discounts, provide customer support through live chat, or show targeted advertisements.

The main objective of this project is to predict whether a customer will make a purchase during an online shopping session. The prediction is based on details such as pages visited, time spent on the website, Google Analytics metrics, and visitor information.

The target variable is Revenue, where 1 represents a purchase and 0 represents no purchase. This makes it a binary classification problem.

In this project, five different machine learning classification models are trained and evaluated using six performance metrics. An interactive Streamlit application is also developed to demonstrate the predictions of these models.

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
- **Live Streamlit App:** https://online-shopper-purchase-prediction.streamlit.app/

## d. Models Used

1. Logistic Regression (`max_iter=1000`)
2. Decision Tree Classifier (`max_depth=5`)
3. K-Nearest Neighbors (`n_neighbors=7`)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, `probability=True`)

All models were trained on the same training set and evaluated on the same 2,441-row test set.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8894 | 0.8843 | 0.7857 | 0.4031 | 0.5329 | 0.5117 |
| Decision Tree | 0.9005 | 0.9077 | 0.7683 | 0.5209 | 0.6209 | 0.5802 |
| KNN | 0.8771 | 0.8046 | 0.7030 | 0.3717 | 0.4863 | 0.4518 |
| Naive Bayes | 0.2835 | 0.7705 | 0.1777 | 0.9869 | 0.3012 | 0.1506 |
| Support Vector Machine | 0.8927 | 0.8656 | 0.7500 | 0.4712 | 0.5788 | 0.5394 |

Precision, Recall, F1 and MCC refer to the positive class (purchase).

### Observations

Logistic Regression - 

Logistic Regression achieved accuracy (0.8894) and the precision (0.7857). This means that when the model predicts a customer will make a purchase, it is usually correct. However, its recall is only 0.4031, meaning it misses many actual buyers. This could be because the model struggles to identify complex patterns in the data.

Decision Tree- 

Decision Tree performed better than the other models overall. It achieved the recall (0.5209), F1-score (0.6209) and MCC (0.5802), with a good precision of 0.7683. It can identify patterns based on conditions and values, which works well for features like PageValues and ExitRates. Limiting the tree depth to 5 also helped control overfitting.

KNN- 

KNN showed comparatively weaker performance, with an F1-score of 0.4863 and MCC of 0.4518. One possible reason is that most customers in the dataset did not make a purchase. Since KNN makes predictions based on nearby data points, it often classifies potential buyers as non-buyers. The large number of encoded categorical columns may also affect its performance.

Naive Bayes- 

Naive Bayes achieved very high recall (0.9869), but its precision was very low (0.1777). This means it identified almost all actual buyers but also incorrectly classified many non-buyers as buyers. Its overall accuracy was only 0.2835. The model may not perform well because some features are highly skewed and several variables are related to each other, whereas Naive Bayes assumes that features are independent.

Support Vector Machine (SVM)- 

SVM delivered the second-best performance after Decision Tree, with an F1-score of 0.5788 and MCC of 0.5394. Its precision was 0.7500, while recall was 0.4712. The RBF kernel helped the model capture non-linear patterns in the data, although it still missed some actual buyers.

Overall Winner – Decision Tree- 

Based on the results, Decision Tree is the best-performing model for this dataset.Decision Tree provides a better balance between identifying actual buyers and avoiding incorrect predictions. Therefore, considering F1-score, recall and MCC, Decision Tree is the most suitable model among the five.


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