# Financial-Loan-Risk-Analysis
End-to-end financial loan risk analysis using EDA, machine learning, and deep learning to analyze borrower risk and loan approval outcomes.

# Financial Loan Approval Prediction & Risk Analysis

## Project Overview

This project uses exploratory data analysis (EDA), statistical analysis, and machine learning to investigate the factors associated with loan approval and develop a predictive classification model.

Using a dataset of **20,000 loan applications**, the project examines borrower demographics, financial characteristics, credit history, and loan attributes to understand patterns associated with loan approval.

The project also evaluates multiple machine learning and deep learning approaches to identify a model that can support data-driven lending decisions.

**The primary objective is to develop a reliable loan approval prediction model that helps financial institutions assess applications, understand borrower risk patterns, and support more informed credit decisions.**

## Business Problem

Financial institutions need to assess loan applications consistently while balancing lending opportunities against potential credit risk.

However, evaluating applications involves several interconnected factors, including income, existing financial obligations, credit history, requested loan amounts, and interest rates.

This project addresses the following questions:

- Which borrower and loan characteristics are most strongly associated with loan approval?
- How do financial capacity and credit history relate to approval outcomes?
- Can machine learning accurately distinguish between approved and rejected applications?
- Which predictive model provides the most effective balance between precision and recall?
- How can model interpretation support more transparent lending decisions?

## Project Objectives

1. Explore borrower characteristics and loan application patterns.
2. Identify relationships between financial indicators, credit history, and loan approval.
3. Investigate data quality issues, missing values, outliers, and potentially redundant features.
4. Develop and evaluate machine learning classification models.
5. Apply hyperparameter tuning and classification threshold optimization.
6. Interpret model predictions using SHAP.
7. Translate analytical findings into practical lending recommendations.

## Dataset Overview

The dataset contains **20,000 observations** and an original set of 35 predictor features.

The target variable is:

| Variable | Description |
|---|---|
| `loan_approved` | Binary target indicating whether a loan application was approved or rejected |
| `0` | Rejected |
| `1` | Approved |

### Target Distribution

- Rejected applications: 15,220 (76.1%)
- Approved applications: 4,780 (23.9%)

The target distribution indicates class imbalance, which was considered during model development and evaluation.

The dataset includes variables relating to:

- **Borrower demographics:** Age, marital status, education, and dependents.
- **Employment:** Employment status, experience, and job tenure.
- **Financial profile:** Annual income, savings, checking balances, assets, liabilities, and net worth.
- **Credit history:** Credit score, previous defaults, payment history, and length of credit history.
- **Loan characteristics:** Loan amount, duration, interest rates, and loan purpose.
- **Existing financial obligations:** Monthly debt payments and debt-to-income measures.

*Dataset source and licensing information should be added here using the verified original source.*

## Analytical Workflow

### 1. Data Cleaning and Preparation

- Standardized column names.
- Inspected and corrected data types.
- Investigated and treated missing values.
- Classified numerical, categorical, and ordinal features.
- Assessed extreme observations using statistical methods.
- Conducted logical data-validity checks.

### 2. Exploratory Data Analysis (EDA)

The exploratory analysis examined borrower profiles, financial characteristics, credit history, and loan approval patterns.

It included:

- Univariate and bivariate analysis.
- Target distribution analysis.
- Numerical and categorical feature exploration.
- Correlation analysis.
- Statistical examination of relationships with loan approval.

### 3. Feature Selection and Leakage Assessment

Several features were investigated for redundancy and potential target leakage.

The following variables were excluded from the primary modeling feature set:

- `monthly_income`: largely duplicated `annual_income`.
- `monthly_loan_payment`: mathematically derived from other loan variables.
- `total_debt_to_income_ratio`: reproducible from its underlying financial components.
- `risk_score`: excluded as a precaution because of its unusually strong association with loan approval and potential to act as a target proxy.
- `log_annual_income`: removed because it duplicates information already represented by annual income.

The final processed feature matrix contained **45 features**.

### 4. Model Development

Four classification approaches were evaluated:

| Model | Purpose |
|---|---|
| Dummy Classifier | Establish a majority-class baseline |
| Logistic Regression | Establish an interpretable classification benchmark |
| Random Forest | Capture nonlinear relationships and feature interactions |
| XGBoost | Evaluate gradient-boosted decision trees |
| Multilayer Perceptron (MLP) | Assess the contribution of deep learning |

Class weighting was used to address the imbalance between approved and rejected applications.

### 5. Model Optimization

XGBoost was optimized using:

- Randomized hyperparameter search.
- Five-fold stratified cross-validation.
- Average precision as the primary tuning metric.
- Classification threshold optimization using training-set cross-validation.

The final XGBoost classification threshold was set to **0.64**, improving the precision-recall balance.

### 6. Model Interpretability

SHAP (SHapley Additive Explanations) was used to identify the features that contributed most strongly to the final XGBoost predictions.

The most influential features included:

- Annual income.
- Interest rate.
- Loan amount.
- Length of credit history.
- Net worth.
- Credit score.
- Education level.
- Monthly debt payments.

The direction analysis indicated that higher income, longer credit histories, and higher net worth generally pushed predictions toward approval, while higher interest rates and larger loan amounts generally pushed predictions toward rejection.

These represent model-learned associations rather than causal relationships.

## Model Evaluation and Results

All final models were evaluated on the same held-out test set of 4,000 observations.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 95.00% | 84.55% | 96.76% | 90.24% | 99.15% | 97.20% |
| Random Forest | 91.73% | 91.95% | 71.65% | 80.54% | 97.83% | 93.95% |
| **Tuned XGBoost** | **96.20%** | **91.87%** | **92.26%** | **92.07%** | 99.29% | **97.89%** |
| MLP Neural Network | 95.43% | 86.09% | 96.44% | 90.97% | **99.33%** | 97.79% |

*XGBoost results use the optimized 0.64 classification threshold. The other models use the default 0.50 threshold.*

### Selected Model: Tuned XGBoost

Tuned XGBoost was selected as the primary model because it achieved the strongest overall balance between precision, recall, F1-score, and PR-AUC.

Its final test performance was:

- Accuracy: **96.20%**
- Precision: **91.87%**
- Recall: **92.26%**
- F1-score: **92.07%**
- ROC-AUC: **99.29%**
- PR-AUC: **97.89%**

### Final Confusion Matrix

Using the optimized threshold of 0.64, the model produced the following results:

| | Predicted Rejected | Predicted Approved |
|---|---:|---:|
| Actual Rejected | 2,966 | 78 |
| Actual Approved | 74 | 882 |

The model correctly classified 3,848 of the 4,000 test applications.

## Key Business Insights

The analysis highlights several important patterns:

1. **Income matters:** Higher annual income generally supports loan approval predictions.
2. **Loan affordability matters:** Larger requested loan amounts tend to push predictions toward rejection.
3. **Interest rates matter:** Higher interest rates are associated with lower approval predictions.
4. **Credit history matters:** Longer credit histories generally contribute positively to approval predictions.
5. **Overall financial position matters:** Net worth, assets, and credit score provide additional information about borrower financial capacity.

### Business Recommendations

- Assess requested loan amounts relative to borrowers' repayment capacity.
- Consider affordability when assigning interest rates and structuring loans.
- Incorporate credit history and broader financial indicators into borrower assessments.
- Use predictive models as decision-support tools alongside human review and established lending policies.
- Monitor model performance and reassess it as borrower profiles and lending conditions change.

## Limitations and Future Improvements

The analysis is based on historical borrower data and may not fully represent future lending conditions.

The availability and timing of certain predictors, particularly `interest_rate`, must be verified before real-world deployment to rule out post-decision information leakage.

Future improvements could include:

- External validation using new loan application data.
- Fairness and bias assessment across borrower groups.
- Probability calibration and further threshold evaluation.
- Model monitoring and periodic retraining.
- Deployment through an interactive loan-risk assessment application.

## Tools and Technologies

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn

**Machine Learning:** Scikit-learn, XGBoost

**Deep Learning:** TensorFlow / Keras

**Model Interpretability:** SHAP

**Development Environment:** Jupyter Notebook / Google Colab

**Version Control:** Git and GitHub

## Final Takeaway

This project demonstrates an end-to-end data science workflow that combines exploratory analysis, statistical investigation, feature selection, predictive modeling, hyperparameter optimization, and model interpretability.

The findings highlight the value of financial and credit-related characteristics in predicting loan approval and demonstrate how machine learning can support more informed, transparent, and data-driven lending decisions.

**The final XGBoost model provides a strong predictive benchmark, while its use in real lending decisions requires further validation, fairness assessment, and verification of feature availability at decision time.**
