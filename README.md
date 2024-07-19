![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)	![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
# Healthcare Provider Fraud Detection Using Machine Learning

## Project Overview
This project addresses the critical issue of healthcare provider fraud using machine learning techniques. By analyzing Medicare claims data, it aims to identify potentially fraudulent claims and providers, helping to reduce financial losses and improve the integrity of the healthcare system.

## Problem Statement
Healthcare provider fraud, including false claims, unnecessary treatments, and billing for unrendered services, costs billions of dollars annually. Using Machine Learning algorithms the project classifies claims as fraudulent or legitimate and identifies key features that contribute to accurate fraud prediction.

## Data Source
The [dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data), from Kaggle, includes:
- Inpatient and outpatient claims data
- Beneficiary details (demographics, enrollment info, chronic conditions)
- Provider information

## Methodology
1. Exploratory Data Analysis
2. Data Preprocessing and Transformation
3. Feature Engineering
4. Data Normalization
5. Feature Selection
6. Model Training and Testing
   - Algorithms: Logistic Regression, Random Forest, Decision Trees, XGBoost
8. Hyperparameter Tuning
9. Model Evaluation

## Evaluation Metrics
| AUC (Area Under the Curve)  |F1-Score | Accuracy  |
| ------------- | ------------- | ------------------

## Key Findings
### Procedure and Diagnosis Codes
- Inpatients: Most common procedure code is 4019 (general surgery), most frequent diagnostic code is 4019 (unspecified essential hypertension).

![](/images/inpatient_procedure_distribution.png)

![](/images/inpatient_diagnosis_distribution.png)

- Outpatients: Most common procedure code is 9904 (general medical procedures), with diagnostic code 4019 also being the most frequent.

![](/images/outpatient_procedure_distribution.png)

![](/images/outpatient_diagnosis_distribution.png)

### Claim Reimbursement Patterns
- For hospital stays (inpatient claims): Most reimbursements are between $0 and $10,000. A few claims have much higher amounts, but these are less common.
- For outpatient visits (no overnight stay): Almost all claims (99.9%) are $3,500 or less. Any claims above $3,500 are unusual and might be for very expensive procedures or could potentially be fraudulent.
> This pattern helps us understand what typical medical costs look like and identify any unusually high claims that might need closer inspection.

<p float="left">
  <img src="/images/Inpatient_claim_reimbursement.png" width="400" />
  <img src="/images/Outpatient_Claim_Reimbursement.png" width="400" />
</p>

**Figure:** Distribution of Claim Amount Reimbursement for Inpatient (left) and Outpatient (right) services.

### Financial Impact
- In 2009, approximately $290 million was lost to fraud.
- $241 million was lost in the inpatient setting, and $54 million in the outpatient setting.

### Age Distribution
- Higher concentration of potential fraud cases among patients over 65.
  
![](/images/fraud_based_on_age.png)

These insights highlight the complexity of healthcare fraud detection and the importance of thorough data analysis and preprocessing in developing effective machine learning models for fraud identification.

## Results
### Using All Features

| Model               | Hyperparameters                                                    | Accuracy | F1 Score | AUC    |
|---------------------|---------------------------------------------------------------------|----------|----------|--------|
| Logistic Regression | Penalty 'l2', C = 10.0                                              | 0.6298   | 0.4829   | 0.5875 |
| Decision Tree       | max_depth: 50, min_samples_split: 270                               | 0.7522   | 0.6951   | 0.8227 |
| Random Forest       | criterion: 'gini', max_depth: 8, max_features: 'auto', n_estimators: 300 | 0.6387   | 0.5495   | 0.6576 |
| XGBoost             | n_estimators: 100, eta: 0.3                                         | 0.7623   | 0.6929   | 0.8177 |

### Using Important Features

| Model               | Hyperparameters                                                    | Accuracy | F1 Score | AUC    |
|---------------------|---------------------------------------------------------------------|----------|----------|--------|
| Logistic Regression | C: 1000.0, penalty: 'l2'                                            | 0.6287   | 0.48406  | 0.5846 |
| Decision Tree       | max_depth: 50, min_samples_split: 270                               | 0.7525   | 0.6954   | 0.8227 |
| Random Forest       | n_estimators: 500, max_features: 'auto', max_depth: 8, criterion: 'entropy' | 0.6352   | 0.5529   | 0.6615 |
| XGBoost             | n_estimators: 50, eta: 0.3                                          | 0.7519   | 0.6786   | 0.8063 |

- The Decision Tree model performed consistently well across both feature sets, achieving the highest AUC of 0.8227.
- XGBoost also showed strong performance, particularly when using all features.
- Feature selection slightly improved the performance of the Decision Tree model but had mixed effects on other models.
- Logistic Regression had the lowest performance among the models tested.

> ## Tools and Technologies
  - Python
  - Scikit-learn
  - Pandas
  - Matplotlib/Seaborn

