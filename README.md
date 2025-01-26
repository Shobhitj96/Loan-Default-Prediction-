# Loan Default Prediction and Analysis

## Project Overview

This project focuses on predicting loan default based on various customer features. By analyzing a variety of attributes, including demographic information, financial history, and loan-related details, we aim to develop a model that can accurately predict whether a loan applicant will default on their loan.

The project incorporates advanced data visualization techniques, statistical analysis, feature engineering, and model evaluation using machine learning algorithms. It also includes anomaly detection, data preprocessing, and model interpretability methods.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Data Analysis](#data-analysis)
3. [Data Visualization](#data-visualization)
4. [Feature Engineering](#feature-engineering)
5. [Statistical Testing](#statistical-testing)
6. [Model Development](#model-development)
7. [Model Evaluation](#model-evaluation)
8. [Anomaly Detection](#anomaly-detection)
9. [Conclusion](#conclusion)

---

## 1. Introduction

The goal of this project is to understand the factors influencing loan defaults and develop a predictive model that can be used to forecast loan defaults in the future. The dataset contains various features like **Age**, **Income**, **CreditScore**, and **LoanAmount**, among others. These features will be analyzed to understand the relationship with the target variable, **Default**, which indicates whether a borrower has defaulted on their loan (1 for default, 0 for no default).

---

## 2. Data Analysis

### Data Cleaning and Preprocessing

1. **Handling Missing Data**: 
   - Missing values are either imputed using appropriate techniques or dropped if necessary.
   
2. **Encoding Categorical Variables**:
   - Categorical features such as **Education**, **EmploymentType**, and **LoanPurpose** are transformed using **Label Encoding** to convert them into numerical representations for model training.

3. **Outlier Detection**:
   - Outliers in continuous variables are detected and handled to prevent them from skewing the results.

4. **Data Normalization**:
   - Continuous variables are scaled appropriately to bring all features onto a similar range.

---

## 3. Data Visualization

### Key Visualizations

1. **Loan Purpose Distribution by Age Group**:
   - An **area plot** is used to visualize the distribution of loan purposes across different age groups, providing insight into how loan purposes vary across demographics.

2. **Credit Score Distribution for Defaulted vs Non-defaulted Customers**:
   - A **violin plot** is used to compare the **Credit Score** distribution for defaulted and non-defaulted customers. This helps in understanding if there’s a significant difference in the credit scores of defaulters and non-defaulters.

3. **Density Plot Between Age and Income**:
   - A **2D density plot** helps visualize the relationship between **Age** and **Income**, highlighting regions with higher concentrations of data points.

4. **Correlation Heatmap**:
   - A **heatmap** of the correlation matrix is used to understand the relationships between numerical features in the dataset. This is helpful in identifying highly correlated features that can be used in model building.

5. **Partial Dependence Plots (PDP)**:
   - PDPs are used to analyze the impact of specific features (e.g., **Income**, **Interest Rate**) on the predicted probability of default. These plots provide valuable insights into how changes in feature values affect the target variable.

---

## 4. Feature Engineering

1. **Binning**:
   - Numerical features such as **Age**, **Income**, and **LoanAmount** are binned into categories to facilitate better visualization and analysis.

2. **Interaction Features**:
   - Interaction terms between features are created to capture relationships that may not be evident when looking at individual features in isolation.

3. **Feature Importance**:
   - **Random Forest** model is used to rank the importance of features in predicting loan defaults. This helps in selecting the most relevant features for the predictive model.

---

## 5. Statistical Testing

### Chi-Square Test

To assess the relationship between categorical features and the target variable (**Default**), the **Chi-Square Test of Independence** is used. Features with a **p-value** less than 0.05 are considered statistically significant and potentially valuable predictors.

### Variance Inflation Factor (VIF)

To detect multicollinearity between features, **Variance Inflation Factor (VIF)** is calculated. Features with high VIF values (greater than 5) are removed or transformed to ensure the stability of the predictive model.

---

## 6. Model Development

Several machine learning models are explored to predict loan defaults:

1. **Logistic Regression**:
   - A baseline **Logistic Regression** model is trained and evaluated using cross-validation.

2. **Random Forest Classifier**:
   - A more complex **Random Forest Classifier** is trained to capture non-linear relationships between the features and the target variable. Feature importance is also evaluated using this model.

3. **Support Vector Machines (SVM)**:
   - SVMs are tested for their ability to classify customers as default or non-default based on the provided features.

4. **Isolation Forest**:
   - **Isolation Forest** is used for anomaly detection to identify unusual patterns that might indicate fraudulent activity or data quality issues.

---

## 7. Model Evaluation

1. **Classification Report**:
   - The **classification report** provides key metrics such as precision, recall, and F1-score, which help in evaluating the model's performance, especially in imbalanced datasets.

2. **Confusion Matrix**:
   - The **confusion matrix** visualizes the performance of the model, helping to understand the number of false positives, false negatives, true positives, and true negatives.

3. **ROC-AUC Score**:
   - The **ROC-AUC score** is calculated to assess the model's ability to discriminate between default and non-default customers.

---

## 8. Anomaly Detection

The **Isolation Forest** algorithm is used to detect anomalies in the dataset. Anomalies are identified and classified, and a **confusion matrix** and **classification report** are generated to assess the performance of the anomaly detection process.

---

## 9. Conclusion

This project demonstrates the process of analyzing loan default data and building predictive models. The following key insights were gained:

1. **Significant Predictors**: Features such as **Credit Score**, **Income**, and **DTI Ratio** are strong predictors of loan default.
2. **Data Imbalance**: The dataset exhibits a class imbalance, which was addressed using **SMOTE** to resample the minority class (default).
3. **Model Performance**: The **Random Forest** model performed well in predicting loan defaults, with significant feature importance contributing to the accuracy of the model.

---

## Future Work

- Further fine-tuning of machine learning models using hyperparameter optimization techniques.
- Exploration of other algorithms such as **XGBoost** and **LightGBM** for better performance.
- Implementation of time series analysis if loan data includes temporal information.

---

