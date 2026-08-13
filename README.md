# Credit-Card-Default-Prediction
ML project predicting credit card default risk using classification models (Business Intelligence capstone)

This is my capstone project for the Data Science & ML course, where I built a model to predict 
whether a credit card client is likely to default on their payment next month.

## About the project

Banks and credit card companies deal with this problem all the time — they need to figure out, 
based on a customer's history, whether they're at risk of not paying. I picked this as my 
Business Intelligence topic because it's a real-world classification problem that's actually 
used in the finance industry.

## Dataset

I used the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

- 30,000 rows, 25 columns
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- Target variable: whether the client defaulted on payment next month (1 = yes, 0 = no)
- Features include credit limit, age, education, marital status, past payment history, and bill amounts over 6 months

## What I did

**1. Exploratory Data Analysis (EDA)**
Went through the data to understand its structure — checked for missing values, duplicates, 
distributions, and how features relate to the target. Found that EDUCATION and MARRIAGE had some 
undocumented category values that needed cleaning later, and that the target variable is imbalanced 
(most customers didn't default).

**2. Preprocessing**
- Removed 35 duplicate rows
- Fixed invalid EDUCATION/MARRIAGE categories (grouped undocumented values into "others")
- Capped outliers in key numeric columns using the IQR method
- Scaled features using StandardScaler
- Split into train/test sets (80/20, stratified to keep the class balance consistent)

**3. Model Building**
Trained and compared four models:
- Logistic Regression
- Decision Tree
- KNN
- Random Forest

Since the target is imbalanced, I focused on **F1-score** rather than just accuracy — a model that 
just predicts "no default" for everyone would get ~78% accuracy without being useful at all.

Random Forest performed best, and I tuned it further using GridSearchCV (testing different 
`n_estimators`, `max_depth`, and `class_weight='balanced'` to handle the imbalance).

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.81 | 0.70 | 0.25 | 0.37 |
| KNN | 0.79 | 0.54 | 0.34 | 0.42 |
| Random Forest | 0.81 | 0.63 | 0.36 | 0.46 |
| Decision Tree | 0.82 | 0.65 | 0.36 | 0.47 |
| **Tuned Random Forest (final)** | 0.77 | 0.49 | **0.58** | **0.53** |

The tuned Random Forest ended up with slightly lower accuracy but a much better recall and F1-score. 
In this project, I prioritized recall because identifying more potential defaulters was considered more important than maximizing overall accuracy.

**Top predictive feature:** PAY_0 (the customer's most recent repayment status) — by far the 
strongest signal for default.

## Tools used

- Python (Pandas, NumPy)
- Matplotlib & Seaborn for visualization
- Scikit-learn for modeling and evaluation
- Jupyter Notebook

## Project structure
├── 01_EDA.ipynb # Exploratory data analysis
├── 02_Preprocessing.ipynb # Data cleaning and preparation
├── 03_Model_Building.ipynb # Model training, tuning, evaluation
├── cleaned_credit_default.csv # Cleaned dataset
├── final_model_random_forest.pkl # Saved final model
└── README.md

# What I'd try next

- Try SMOTE or other resampling techniques to handle the class imbalance differently
- Test XGBoost/LightGBM to see if they beat Random Forest
- Do more feature engineering on the payment history columns instead of using them raw
