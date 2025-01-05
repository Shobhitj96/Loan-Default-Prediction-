# Loan-Default-Prediction-

This repository provides a comprehensive analysis of loan default prediction using machine learning techniques. The dataset contains information about loan applicants, including financial and demographic features. The goal is to predict whether a loan applicant will default on their loan.

Dataset

The dataset consists of the following features:

1. LoanID: Unique identifier for the loan applicant.


2. Age: Age of the loan applicant (numerical).


3. Income: Annual income of the applicant (numerical).


4. LoanAmount: Amount of the loan applied for (numerical).


5. CreditScore: Applicant's credit score (numerical).


6. MonthsEmployed: Number of months the applicant has been employed (numerical).


7. NumCreditLines: Number of credit lines the applicant has (numerical).


8. InterestRate: Interest rate of the loan (numerical).


9. LoanTerm: Term duration of the loan (numerical).


10. DTIRatio: Debt-to-income ratio of the applicant (numerical).


11. Education: Education level of the applicant (categorical).


12. EmploymentType: Employment status (categorical).


13. MaritalStatus: Marital status of the applicant (categorical).


14. HasMortgage: Whether the applicant has a mortgage (categorical).


15. HasDependents: Whether the applicant has dependents (categorical).


16. LoanPurpose: Purpose of the loan (categorical).


17. HasCoSigner: Whether the applicant has a co-signer for the loan (categorical).


18. Default: Target variable (binary). Indicates whether the applicant defaults on the loan (1) or not (0).



Data Preprocessing

The dataset has been preprocessed, including:

Binning of Numerical Features: For better visualization and understanding of distributions, numerical features like Age, Income, LoanAmount, CreditScore, etc., have been binned into intervals.

Categorical Encoding: Categorical features like Education, EmploymentType, etc., are encoded using LabelEncoder for machine learning compatibility.

Feature Selection: Random Forest and other statistical methods (like Chi-Square test) have been used to identify the most important features affecting loan default prediction.


Key Insights

Visualizations

Various visualizations have been created to understand the distribution of features and their relationships with the target variable (Default):

Loan Amount vs Default: As loan amount increases, the count of defaults increases.

Credit Score vs Default: Higher credit scores are associated with fewer defaults.

Interest Rate vs Default: As interest rate increases, the probability of default also increases.

Employment Type vs Default: Unemployed individuals tend to have higher default rates.


Statistical Tests

Chi-Square Test: Categorical features like EmploymentType, Education, LoanPurpose, etc., were tested for significance in relation to the target variable (Default). Features such as Education, EmploymentType, MaritalStatus, and others showed significant associations with Default.


Feature Importance

Random Forest: A Random Forest model was used to determine feature importance. The top 5 features influencing loan default prediction are:

Income

Interest Rate

Loan Amount

Credit Score

Age



Partial Dependence Plots (PDP)

Income vs Default Probability: A decreasing trend was observed, where higher income correlates with a lower probability of default.

Interest Rate vs Default Probability: Higher interest rates were associated with a higher probability of default.


Anomaly Detection

Isolation Forest: An anomaly detection technique was applied to identify unusual loan applications. The model identified several anomalies, although many were false positives (non-default cases flagged as anomalies).


Resampling Techniques

SMOTE (Synthetic Minority Over-sampling Technique): SMOTE was applied to address class imbalance in the dataset. After resampling, the number of default cases increased to balance the data.


Model Evaluation

Logistic Regression: After resampling, the Logistic Regression model achieved a satisfactory ROC-AUC score. A classification report was generated, highlighting precision, recall, and F1-score for predicting loan default.

Random Forest: With hyperparameter tuning (pruning), the Random Forest model showed good performance in predicting loan defaults after SMOTE resampling.


Conclusion

This analysis has provided valuable insights into loan default prediction. 

--
