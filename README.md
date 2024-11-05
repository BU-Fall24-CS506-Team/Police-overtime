# Police Overtime Project Report

## 1. Description of the Project

This project aims to analyze the overtime budget and expenditure patterns within the Boston Police Department (BPD). The BPD operates on an annual budget exceeding $400 million, a portion of which goes toward overtime pay for officers. Our goal is to investigate how the overtime budget is allocated and whether discrepancies exist between overtime hours worked and paid. We will also explore whether certain demographic groups of officers, such as those with longer tenure or specific ranks, have a higher likelihood of overtime discrepancies. By performing this analysis, we hope to provide insights into potential areas of inefficiency or bias in police spending, contributing to ongoing conversations around police accountability and financial transparency.

## 2. Clear Goal(s)

The main objectives of this project are:

- To analyze year-over-year changes in BPD overtime budgets and spending both intra and inter departments.
- To compare overtime hours worked versus overtime hours paid and identify any discrepancies in its amount and ratio.
- To identify the amount of injury pay and the percentage of officers taking them.
- To identify outliers in the distribution of ratios of overtime worked vs. overtime paid.
- To explore demographic factors (e.g., officer rank, tenure, gender, and race) and their influence on overtime pay discrepancies.
- To create visualizations that illustrate trends and anomalies in overtime payments.

## 3. Preliminary Visualizations of Data

To gain initial insights into the data, we conducted several preliminary visualizations:

1. Year-over-Year Overtime Spending

![image](https://github.com/user-attachments/assets/5cc0a3ca-03b2-4fdc-b405-4338b9e5dd3b)


- Overtime spending has increased from ~$57.9M in 2014 to ~$88.5M in 2023.
- Significant increases in 2018 and 2023 suggest growing operational demands or increased overtime rates.

2. Spending on Salary Categories

![image](https://github.com/user-attachments/assets/1375e63d-7332-4d65-a767-f7847a33c966)


- **Regular Pay** is the largest cost driver, followed by **Overtime**.
- Total spending shows an upward trend, with notable increases around 2018-2019 and 2023.

3. Overtime Paid vs. Regular Hours

![image](https://github.com/user-attachments/assets/839ef134-6483-4a09-b3b9-2b01121ed31c)


- Most employees have low regular hours and overtime pay.
- Outliers show high overtime payments with minimal regular hours, suggesting special assignments or inefficiencies.

4. Overtime Discrepancy Ratios

![image](https://github.com/user-attachments/assets/b1041e99-2f2c-419d-8876-a9a4168d8363)


- Most ratios are low, but some outliers indicate disproportionately high overtime pay relative to regular hours.


## 4. Data Processing

### Data Collection

- **Data Sources**: Salary data from 2014 to 2023 was downloaded from the [Analyze Boston](https://data.boston.gov/) portal.
- **Data Directory**: All datasets were stored in a `data` directory for organized access.

### Data Cleaning and Standardization

- **Header Standardization**: Column names varied across datasets. A mapping was created to standardize headers to a common format.
- **Year Column Addition**: A `year` column was added to each dataset to track the data over time.
- **Data Concatenation**: All yearly datasets were combined into a single DataFrame for comprehensive analysis.
- **Department Filtering**: Extracted records where `DEPARTMENT_NAME` is "Boston Police Department" to focus on BPD employees.
- **Currency Formatting**: Removed dollar signs (`$`) and parentheses from salary columns to convert them to numeric types.

### Handling Missing Values

- **Missing Value Analysis**: Identified missing values in key salary columns.
- **Imputation**: Filled missing values in income-related columns with zero, assuming no earnings in those categories.
- **Column Dropping**: Removed irrelevant columns such as `NAME`, `_id`, and `DEPARTMENT_NAME` to streamline the dataset.

### Feature Scaling and Encoding

- **Standardization**: Applied `StandardScaler` to numeric features to normalize the data.
- **Feature Selection**: Calculated correlations and Variance Inflation Factor (VIF) to identify and remove multicollinear features.

### Outlier Detection

- **Z-Score Method**: Calculated Z-scores for `OVERTIME` pay to identify outliers beyond a threshold of 5.
- **Outlier Analysis**: Listed employees with unusually high overtime pay for further investigation.

## 5. Data Modeling Methods

### Linear Regression

- **Objective**: To predict overtime pay based on other salary components.
- **Features Used**: `REGULAR`, `RETRO`, `DETAIL`, `QUINN_EDUCATION`, `TOTAL_GROSS`, `INJURED`.
- **Results**:
  - **R² Score**: 0.823
  - **Mean Absolute Error**: Approximately \$6,429
- **Interpretation**: The model explains about 82% of the variance in overtime pay, indicating a strong linear relationship.

### Quantile Regression

- **Objective**: To model the median of the overtime pay distribution.
- **Results**:
  - **R² Score**: Negative, indicating poor model fit.
- **Interpretation**: Quantile regression did not perform well, suggesting that the median is not a good predictor for this dataset.

### XGBoost Regression

- **Objective**: To improve prediction accuracy using a boosted tree model.
- **Results**:
  - **R² Score**: 0.948
  - **Mean Absolute Error**: Approximately \$3,121
- **Feature Importance**:
  - `TOTAL_GROSS` was the most significant feature.
- **Interpretation**: XGBoost provided the best predictive performance, capturing nonlinear relationships in the data.

### Regularized Regression (Lasso)

- **Objective**: To prevent overfitting by penalizing large coefficients.
- **Results**:
  - **R² Score**: 0.823
- **Interpretation**: Similar performance to linear regression, indicating that regularization did not significantly impact model accuracy.

### K-Means Clustering

- **Objective**: To identify clusters of employees with similar earning patterns and detect outliers.
- **Method**:
  - Used the Elbow Method to determine the optimal number of clusters (K=3).
  - Applied K-Means clustering to standardized features.
- **Results**:
  - Successfully identified groups and outliers.
- **Interpretation**: Clustering revealed distinct patterns in earnings and highlighted individuals with exceptionally high overtime pay.

## 6. Preliminary Results

- **Overtime Trend**: There is a clear upward trend in overtime spending within the BPD over the past decade.
- **Model Performance**:
  - **Linear Regression**: Provided a good fit with an R² of 0.823, suggesting linear relationships between features and overtime pay.
  - **XGBoost Regression**: Achieved superior performance with an R² of 0.948, indicating that nonlinear models capture the data patterns more effectively.
- **Feature Importance**:
  - `TOTAL_GROSS` is the most significant predictor of overtime pay.
  - `REGULAR` pay and `DETAIL` earnings also contribute meaningfully.
- **Outlier Detection**: Identified several employees with overtime pay significantly higher than the norm, warranting further investigation.

## Conclusion

The analysis reveals that overtime pay constitutes a significant and growing portion of the BPD's budget. Predictive models, especially nonlinear ones like XGBoost, effectively capture the factors influencing overtime spending. The identification of outliers suggests potential areas for policy review or audit to ensure fiscal responsibility.

---
