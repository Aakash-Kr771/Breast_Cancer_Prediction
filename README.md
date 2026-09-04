# Breast Cancer Prediction – Machine Learning Project

This project performs exploratory data analysis (EDA) on a breast cancer dataset and trains multiple machine learning classification models, comparing their performance to predict whether a patient has cancer.

## 📌 Project Overview

- **Goal:** Predict breast cancer based on patient health & lifestyle features (Age, BMI, Blood Pressure, Family History, etc.).
- **Notebook:** `Breast_Cancer_ML.ipynb`
- **Type:** Binary Classification Problem

## 🗂️ Workflow

1. **Data Loading** – Loaded the `breast_cancer_prediction.csv` dataset.
2. **EDA (Exploratory Data Analysis)**
   - Separated categorical and numerical columns and examined their value counts.
   - Checked for missing values (`isnull().sum()`).
3. **Data Cleaning**
   - Filled missing values using mean (for numeric columns like BMI, Tumor Size) and mode (for categorical columns).
4. **Feature Engineering**
   - Converted `BMI`, `Blood Pressure`, `Cholesterol`, and `Tumor Size` into categories (e.g. Underweight/Normal/Overweight, Low/Normal/High).
5. **Visualization**
   - Used Plotly pie charts and Seaborn count plots to view distributions.
   - Built a correlation heatmap.
6. **Preprocessing**
   - Dropped `Patient_ID` and the raw numeric columns that had already been converted into categories.
   - Used `OneHotEncoder` for categorical features and `StandardScaler` for numerical features (via `ColumnTransformer`).
7. **Model Training & Comparison**
   The following models were trained inside an sklearn `Pipeline`:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)

   Each model was compared on **Accuracy, Precision, Recall, and F1-score**.
8. **Prediction on New Data**
   - Demonstrated a prediction on a sample new patient record using the trained pipeline.

## 🛠️ Tech Stack / Libraries Used

- Python 3
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn (preprocessing, models, metrics)

## 📁 Project Structure

```
Breast_Cancer_ML/
│
├── Breast_Cancer_ML.ipynb       # Main notebook (EDA + ML models)
├── breast_cancer_prediction.csv # Dataset (must also be uploaded)
└── README.md                    # Project documentation
```

> ⚠️ Note: The notebook expects `breast_cancer_prediction.csv` to be in the same folder. Make sure to also upload this dataset CSV to the repo, otherwise the notebook won't run.

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Breast_Cancer_ML.ipynb
   ```

## 📊 Results

The accuracy/precision/recall/F1 comparison of all 5 models can be found in the `results_df` table at the end of the notebook.

## 📌 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation
- Feature importance analysis
- Model deployment (Flask/Streamlit app)

## 📄 License

This project is intended for educational/learning purposes only.
