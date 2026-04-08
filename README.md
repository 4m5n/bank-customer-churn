# Bank Customer Churn Prediction

## Overview
Churn is one of the most expensive problems in banking, where losing a customer means losing years of potential revenue. This project builds a machine learning pipeline to predict whether a customer is likely to churn, and it is deployed in an interactive Streamlit dashboard so anyone can use it without running the code.

## Dataset
I worked with a banking customer churn dataset from Kaggle that contains features such as:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited`

The goal was to predict a binary outcome: did the customer leave (`Exited=1`) or stay (`Exited=0`)?

**Source:**  
[Bank Customer Churn Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset?resource=download)

## Data Exploration
Before modeling, I visualized the distributions of key features to understand the data. This revealed patterns such as:
- how `Tenure` influences churn
- how `Age` impacts churn
- whether account `Balance` influences churn rate

I also examined `NumOfProducts`, which I expected to strongly influence churn, but it turned out to have less impact than expected.

## Preprocessing
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, and `Surname`
- Label encoded categorical features: `Geography` and `Gender`
- Split the data into an 80/20 training/test split
- Applied SMOTE to address class imbalance, generating synthetic samples for the minority churn class

## Modeling
I trained an XGBoost classifier, which is well suited for structured tabular data. The chosen parameters were:
- `n_estimators=100`
- `max_depth=5`
- `learning_rate=0.1`

This setup provides a strong baseline while avoiding heavy overfitting and keeping training efficient.

## Results
The model achieved approximately **81% overall accuracy** on the test set.

However, the confusion matrix showed a more nuanced picture:
- the model is stronger at identifying customers who stay than those who leave
- among actual churners, recall was around **68%**
- when the model predicted churn, precision was about **53%**

In a banking context:
- missing a churner is more costly because you lose the customer
- a false alarm usually means an unnecessary retention offer

So, improving recall for the churn class should be the priority in future iterations.

## Streamlit Dashboard
I built an interactive dashboard where users can input customer attributes such as:
- age
- credit score
- account balance

The app returns an instant churn prediction, making the model accessible to non-technical users.

## What’s Next
Future improvements could include:
- more rigorous hyperparameter tuning with tools like Optuna
- adding SHAP values for model explainability
- adjusting the classification threshold to prioritize recall
- exploring alternative evaluation metrics like F1 score
