# Predictive Modeling of Obesity Levels

This project focuses on building **machine learning models** to predict obesity levels in individuals based on their **lifestyle, eating habits, and physical conditions**.

It demonstrates the full end-to-end process — from **data preprocessing** and **EDA** to **model training**, **evaluation**, and **visualization** — using Python and popular data science libraries.

---

## Features

🔍 Exploratory Data Analysis (EDA) — visualizes distributions and correlations.

⚙️ Data Preprocessing — handles missing values, encodes categories, and normalizes features.

🤖 Model Training — uses multiple algorithms:

  Random Forest Classifier

  Gradient Boosting / HistGradientBoosting

  Logistic Regression

📈 Evaluation Metrics — accuracy, confusion matrix, and classification report.

🔎 Feature Importance — identifies key predictors of obesity level.

## Workflow Overview
1️⃣**Load Dataset**
import pandas as pd

df = pd.read_csv("obesity_dataset.csv")
print(df.head())

2️⃣ **Preprocess Data**
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['family_history_with_overweight'] = encoder.fit_transform(df['family_history_with_overweight'])

3️⃣ **Train-Test Split**
from sklearn.model_selection import train_test_split

X = df.drop('Obesity_Level', axis=1)
y = df['Obesity_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

4️⃣ **Train a Model**
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

5️⃣ **Evaluate Model**
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

6️⃣ **Feature Importance Visualization**
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance = results.importances_mean

plt.barh(X.columns, importance)
plt.xlabel("Feature Importance")
plt.title("Permutation Feature Importance")
plt.show()

## Dataset

Dataset: Obesity Levels based on Eating Habits and Physical Condition

Source: UCI Machine Learning Repository
License: Public Domain

## Model Summary

| Model                    | Key Params                         | Accuracy | Notes                                |
| ------------------------ | ---------------------------------- | -------- | ------------------------------------ |
| **HistGradientBoosting** | `max_depth=5`, `learning_rate=0.1` | ⭐ 90%    | Best balance between bias & variance |
| **Random Forest**        | `n_estimators=100`                 | 88%      | Stable and interpretable             |
| **Logistic Regression**  | default                            | 75%      | Baseline for comparison              |

