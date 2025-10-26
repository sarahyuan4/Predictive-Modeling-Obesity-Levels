# Predictive Modeling of Obesity Levels

This project focuses on building machine learning models to predict obesity levels in individuals based on their lifestyle, eating habits, and physical conditions.

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

2️⃣ **Preprocess Data**

3️⃣ **Train-Test Split**

4️⃣ **Train a Model**

5️⃣ **Evaluate Model**

6️⃣ **Feature Importance Visualization**


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

