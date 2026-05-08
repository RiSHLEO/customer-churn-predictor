# Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to cancel their subscription, enabling businesses to intervene before losing them.

**Live App:** [link coming after deployment]

---

## The Business Problem

Customer churn is one of the most costly problems for subscription businesses. Acquiring a new customer costs 5x more than retaining an existing one. This project builds a predictive model to identify at-risk customers before they leave.

---

## Key Findings from Exploratory Analysis

- **Contract type is the strongest predictor** — month-to-month customers churn at 42.7% vs just 2.8% for two-year contracts, a 15x difference
- **New customers are highest risk** — churn rate drops sharply after the first 12 months, making early retention efforts critical
- **High paying customers churn more** — customers with above-median monthly charges leave at a higher rate, suggesting a perceived value problem rather than a price problem

---

## Model Performance

| Model | Recall (Churners) | Precision | Accuracy |
|---|---|---|---|
| Random Forest | 49% | 62% | 79% |
| XGBoost (threshold=0.5) | 66% | 53% | 75% |
| XGBoost (threshold=0.4) | 74% | 50% | 74% |

**Final model:** XGBoost with decision threshold tuned to 0.4, achieving **74% recall on churners.**

Recall was prioritised over accuracy because missing a churner (false negative) is more costly than a false alarm (false positive) — a missed churner is a lost customer, while a false alarm results in an unnecessary but cheap retention offer.

---

## Technical Stack

- **Data:** Telco Customer Churn dataset (Kaggle)
- **Libraries:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud

---

## How to Run Locally

```bash
git clone https://github.com/yourusername/customer-churn-predictor
cd customer-churn-predictor
pip install -r requirements.txt
cd app
streamlit run app.py
```

---

## What I Would Improve With More Time

- Add SHAP values to explain individual predictions inside the app
- Test additional models — LightGBM, Logistic Regression
- Add a batch prediction feature — upload a CSV of customers and get predictions for all of them
- Collect real customer feedback to retrain the model over time
