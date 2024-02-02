# Predicting-Hospital-Readmissions
This is an open-source project repo which deals with predicting if the patients will get readmitted to the hospital or not in the next 30 Days.

## Introduction
The primary goal of this project is to build a predictive model that can identify patients who are at high risk of hospital readmission within 30 days after their initial discharge.  
This project aims at ETL of data from the csv file, Cleaning, Performing EDA, Data Visualization, Feature Extraction and Classification Model Building using the Machine Learning algorithms available.

## Table of Contents
1. Pre-requisites
2. Technology Stacks 
3. Usage
4. Column Details
5. Data Cleaning and Transformation
6. Model Building
7. Model Evaluation
8. EDA - Data Visualization
9. Feature Engineering
10. Final Thoughts & Insights

## Pre-requsites
Install the following packages to run the project. 
```
pip install pandas 
pip install numpy 
pip install seaborn 
pip install matplotlib 
pip install plotly 
pip install scikit-learn 
pip install imbalanced-learn 
pip install yellowbrick
```

## Technology Stack
- Python scripting 
- Pandas
- Jupyter Notebook
- ETL from CSV
- Data Pre-processing
- Encoding of Categorical to Numerical data
- Sklearn - Model Building
- EDA and Feature Engineering
- Plotly, Seaborn, Matplotlib

## Usage
Clone the repo from the below mentioned link.  
[Predicting-Hospital-Readmissions](https://github.com/Chindhu-Alagappan/Predicting-Hospital-Readmissions.git)   
Install packages from "requirement.txt" 
Run the "Hospital_Readmissions.ipynb" using jupyter notebook.

## Column Details 
**Table : Hospital Readmissions**
| Column Name | Description | Options |
| :---------- | :---------- | :------- |
| Patient_ID | Unique identifier of the patient | - |
| Age | Age of the patient | - |
| Gender | Gender of the patient | Male, Female, Others | - |
| Admission_Type | Admission_type of the patient | Emergency, Elective, Urgent |
| Diagnosis | Diagnosis for which the patient is visiting the hospital | Diabetes, Heart Disease, Infection, Injury |
| Num_Lab_Procedures | No. of lab procedures undergone by the patient | - |
| Num_Medications | No. of medications given to the patient | - |
| Num_Outpatient_Visits | No. of outpatient visits of the patient | - |
| Num_Inpatient_Visits | No. of inpatient visits of the patient | - |
| Num_Emergency_Visits | No. of emergency visits of the patient | - |
| Num_Diagnoses | No. of diagnoses done by the patient | - |
| A1C_Result | Measures the amount of hemoglobin with attached glucose and reflects your average blood glucose levels over the past 3 months | Abnormal, Normal, Nan |
| Readmitted | Is the patient readmitted or not | Yes, No |

## Data Cleaning and Transformation 
- Load the dataset from CSV file to a pandas dataframe.
- Handle missing values.
- Remove outliers.
- Transform the categorical variables to numerical.

## Model Building
- Train_test_split of the dataset. Independent Columns = 17, Target Columns = 1.  
- SMOTE - Synthetic Minority Oversampling Technique - to handle the imbalanced classes in the target column.   
- Scaling the dataset, by applying fit_transform(X_train_smote) and fit(X_test).  
- PCA - Principal Component Analysis - no. of components = 5. Reduce the curse of dimensionality from 17 to 5 by applying PCA.  
- Models Used for Comparison :
  - Logistic Regression
  - KNN Classifier
  - SVC
  - Random Forest
  - Ada Boosting
  - Gradient Boosting

## Model Evaluation
- All the models mentioned above are evaluated to predict the train and test metrics.
![Training Evalustion Metrics](https://github.com/Chindhu-Alagappan/Predicting-Hospital-Readmissions/blob/09572a33c94a6c093dd261d59e31e809501e0d16/Metrics/Training_Eval_Metrics.png)  
  
![Testing Evalustion Metrics](https://github.com/Chindhu-Alagappan/Predicting-Hospital-Readmissions/blob/09572a33c94a6c093dd261d59e31e809501e0d16/Metrics/Testing_Eval_Metrics.png)  

- From the above images, it is clear that KNN provides the highest accuracy and precision score when compared to others.
- Hyper Parameter Tuning - Tuning the params of KNN has a posibility to increase the accuracy rate, so applied it. Unfortunately, the model has undergone an overfitting problem, which in turn REDUCED the testing accuracy. So, neglecting it.  
  
## EDA - Data Visualization
The following are the categories in which the plots / charts have been drawn.
- Age, Readmitted
- Gender, Age_Group, Readmitted
- Admission_Type, Readmitted
- Diagnosis, Readmitted
- Num_Lab_Procedures, Readmitted
- Num_Medications, Readmitted
- Num_Outpatient_visits, Readmitted
- Num_Inpatient_visits, Readmitted
- Num_Emergency_visits, Readmitted
- Num_Diagnoses, Readmitted

## Feature Engineering
The below image depicts the correlation between the columns used for model building. The correlation coefficient between the columns are low - Values are between the range 0 and -0.5 (Approx), so features can't be eliminated.  
![Correlation Matrix](https://github.com/Chindhu-Alagappan/Predicting-Hospital-Readmissions/blob/2539b0733821e64e731f3fa2d1155f7a9f93236a/Metrics/Correlation_Matrix.png)  

Also, feature importance is derived using the Logistic Regression and Random Forest classifiers.

## Final Thoughts & Insights
- Male are getting readmitted more than female and others.
- If the Num_Medications is more than 30, then all "Male" are likely to get readmitted.
- If the Num_Medications is more than 30, then all "Other"(Gender) are NOT likely to get readmitted.
- If the Num_Medications is more than 30 and the age group of a person is between 20 and 30, then there is a high chance of 66.6% for him/ her to get readmitted.
- If the Num_Outpatient_Visits increases, the chance of getting readmitted also increases (ie) Num_Outpatient_Visits are directly proportional to Readmitted_Yes columns.
- There is a high probability of 16.59% for people of age 40-50 to get readmitted (Count=81) than any other age groups.
- There is a high probability of 16.13% for people of age 70-80 to NOT get readmitted (Count=81) than any other age groups.
- After applying EDA and feature importance, it is clear that the columns - "Gender", "Diagnosis", "Admission_Type" - majorly contributes to the model built.
- It is evident that the hospital readmissions focuses primarily on these 3 fields. Other fields have a minor impact on the target column (Readmission_Yes).
