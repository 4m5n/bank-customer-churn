Overview
Churn is considered to be one of the most expensive problems in banking, where losing a customer means losing years of potential revenue. This project builds a machine learning pipeline to predict whether a customer is likely to churn, and is deployed in an interactive Streamlit dashboard so anyone can use it without running the code.
The Dataset
I worked with a Banking Customer Churn Dataset found on Kaggle that contained features such as Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary, and Exited (Churned). The goal was to predict a binary outcome – did the customer leave or stay?
Data Exploration
Before modeling, I visualized the distributions of key features to develop a better understanding of the data. This helped surface patterns like how Tenure influences Churn, if Age impacts churn, and if an individual's balance influences churn rate. I also selected a numerical feature I thought would influence churn rate drastically — the number of products — which turned out to have less influence than expected.
Preprocessing
Categorical features such as geography and gender were label encoded to make them model-ready. I also dropped RowNumber, CustomerId, and Surname, as they don't have an impact on deciding whether a customer will churn or not. I then split the data 80/20 into training and test sets. To address the class imbalance, I applied SMOTE – a technique that synthetically generates new samples of the minority class so the model doesn't just learn to always predict no churn.
Modeling
I trained an XGBoost classifier, which is well suited for structured tabular data like this. It handles non-linear relationships well and is robust out of the box without heavy tuning. I set n_estimators to 100, max_depth to 5, and learning rate to 0.1 — chosen as a baseline to avoid overfitting while keeping training efficient.
Results
The model achieved 81% overall accuracy on the test set. However, the confusion matrix tells a more nuanced story — the model is stronger at identifying customers who stay than those who leave.
Of the customers who actually churned, the model caught 68% of them (recall), but when it predicted churn, it was only right about 53% of the time (precision). In practical terms, that means a meaningful number of churners are still being missed, and about half the churn alerts are false alarms.
In a real banking context, missing a churner is the more costly mistake — you lose the customer entirely. A false alarm at worst means an unnecessary retention offer. That tradeoff suggests that improving recall on the churn class should be the priority in future iterations, whether through further tuning, a lower classification threshold, or a different evaluation metric like F1 score.
Streamlit Dashboard
I built an interactive dashboard where a user can input their own customer attributes — such as age, credit score, and account balance — and get an instant churn prediction. It makes the model usable by anyone, not just someone who can develop and read code.
What's Next
There are a few directions I'd explore to improve this project further. Tuning hyperparameters more rigorously with Optuna could squeeze out better performance, and adding SHAP values would make the model explainable — showing not just whether a customer will churn, but why. I'd also experiment with lowering the classification threshold to prioritize recall and catch more churners, even at the cost of some additional false alarms.
