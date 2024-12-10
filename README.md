# Police Overtime Project Report
## 1. Description of the Project

This project aims to analyze the overtime budget and expenditure patterns within the Boston Police Department (BPD). The BPD operates on an annual budget exceeding $400 million, a portion of which goes toward overtime pay for officers. Our goal is to investigate how the overtime budget is allocated and whether discrepancies exist between overtime hours worked and paid. We will also explore whether certain demographic groups of officers, such as those with longer tenure or specific ranks, have a higher likelihood of overtime discrepancies. By performing this analysis, we hope to provide insights into potential areas of inefficiency or bias in police spending, contributing to ongoing conversations around police accountability and financial transparency.

## 2. Clear Goal(s)(what we will do)

The main objectives of this project are:

- To analyze year-over-year changes in BPD overtime budgets and spending both intra and inter departments.
- To compare overtime hours worked versus overtime hours paid and identify any discrepancies in its amount and ratio.
- To identify the amount of injury pay and the percentage of officers taking them.
- To identify outliers in the distribution of ratios of overtime worked vs. overtime paid.
- To explore demographic factors (e.g., officer rank, tenure, gender, and race) and their influence on overtime pay discrepancies.
- To create visualizations that illustrate trends and anomalies in overtime payments.



## 3.Visualizations of Data

To gain insights into the data, we conducted several visualizations:

1. Average Overtime by Ethnic Group

This bar chart shows the average overtime pay grouped by ethnic group categories. Each bar represents a specific ethnic group, and the height of the bar corresponds to the mean overtime earnings for that group.
![average_overtime](https://github.com/user-attachments/assets/fce4c8ae-4f01-4a62-ad77-c34359e64da4)


Insights:

The data reveals significant disparities in overtime earnings among different ethnic groups.
Black employees have the highest average overtime pay, followed closely by Hispanic and White employees.
Asian employees appear to earn lower average overtime pay compared to other groups.

2. Average Overtime by Sex

This bar chart compares the average overtime pay between male and female employees. The height of each bar indicates the mean overtime pay for each gender.
![average_overtime_sex](https://github.com/user-attachments/assets/5b89ecaa-aef8-461d-bf88-b763809e12bb)

Insights:

Male employees earn significantly higher average overtime pay compared to female employees.
This disparity raises questions about workload distribution, availability, or access to overtime opportunities among genders.


3. Average Regular Pay by Sex

This bar chart compares the average regular pay between male and female employees. Similar to the overtime chart, the height of each bar indicates the mean regular pay.
![average_regular_sex](https://github.com/user-attachments/assets/d992f6b2-7e30-4d74-ba1d-03e96dc0a1d4)

Insights:

The average regular pay for males and females is relatively similar, suggesting parity in base salaries.
The disparity observed in overtime pay is not reflected in regular pay, hinting at differences in work assignments or additional pay structures.

4. Average Regular Pay by Ethnic Group

This bar chart shows the average regular pay grouped by ethnic categories. Each bar represents a specific ethnic group, and its height corresponds to the mean regular earnings.
![average_regular_ethnic](https://github.com/user-attachments/assets/623cf19f-c8bf-4268-99a3-389f7a809929)

Insights:

There is minimal variation in regular pay among different ethnic groups, indicating equitable distribution of base salaries.
This contrasts with the disparities seen in overtime pay, suggesting potential inequities in work assignments or overtime opportunities.

5. Total Amount Spent on Each Salary Category per Year

This stacked bar chart visualizes the total yearly expenditure for various salary components, including regular pay, overtime, detail pay, injured pay, and Quinn education incentives. Each bar represents a year, and the different colors correspond to different salary categories.
![toal_spent_each](https://github.com/user-attachments/assets/b4b5dcf9-b056-44a6-b3e3-1d061d34d2e4)
Insights:

Regular pay consistently constitutes the largest share of salary spending.
Overtime spending has increased over time, reflecting operational demands or potential inefficiencies.
Other salary components, such as Quinn education incentives and detail pay, have remained relatively stable.
Spikes in total spending, such as those seen in 2018 and 2023, suggest extraordinary circumstances or policy changes.

6. Total Amount Spent on Overtime per Year

This line chart illustrates the annual total expenditure on overtime pay from 2014 to 2023. A table below the chart lists the exact overtime amounts for each year.
![total_spent_peryear](https://github.com/user-attachments/assets/854fa122-04f9-427b-baf7-5a47466c15c9)
Insights:

The overtime expenditure shows a steady upward trend, starting at approximately $57.9 million in 2014 and reaching $88.5 million in 2023.
Significant increases are observed in 2018 and 2023, which may correlate with operational or policy changes.
A dip around 2020 might be attributable to pandemic-related disruptions or other unique circumstances.

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
outlier information:

|   _id | NAME              | DEPARTMENT_NAME          | TITLE                   |   REGULAR |   RETRO |    OTHER |   OVERTIME |   INJURED |   DETAIL |   QUINN_EDUCATION |   TOTAL_GROSS |   year |   Overtime_ZScore |
|------:|:------------------|:-------------------------|:------------------------|----------:|--------:|---------:|-----------:|----------:|---------:|------------------:|--------------:|-------:|------------------:|
|     7 | Demesmin,Stanley  | Boston Police Department | Police Lieutenant (Det) |    142466 |       0 | 15820.5  |     167510 |         0 |    24695 |           28198.5 |        378690 |   2021 |           5.68317 |
|    10 | Barrett,Thomas E. | Boston Police Department | Police Sergeant (Det)   |    130930 |       0 | 16724    |     166042 |         0 |        0 |           32732.7 |        346429 |   2021 |           5.62543 |
|     4 | Demesmin,Stanley  | Boston Police Department | Police Lieutenant (Det) |    145775 |       0 | 13932.9  |     196515 |         0 |    11880 |           29155.3 |        397259 |   2022 |           6.82449 |
|    15 | Barrett,Thomas E. | Boston Police Department | Police Sergeant (Det)   |    130930 |       0 | 16724    |     163495 |         0 |        0 |           32732.7 |        343881 |   2022 |           5.52519 |
|    32 | Downey,Paul J     | Boston Police Department | Police Sergeant         |    136589 |       0 |  3187.71 |     163269 |         0 |    12402 |               0   |        315448 |   2022 |           5.51633 |
|     1 | Demesmin,Stanley  | Boston Police Department | Police Lieutenant (Det) |    145775 |       0 |  6053.17 |     221579 |         0 |    23862 |           29155.3 |        426425 |   2023 |           7.81072 |
|    10 | Barrett,Thomas E. | Boston Police Department | Police Sergeant (Det)   |    130930 |       0 | 19672.8  |     180548 |         0 |        0 |           32732.7 |        363884 |   2023 |           6.19622 |
|    12 | Brown,Michael A   | Boston Police Department | Police Sergeant (Det)   |    130930 |       0 | 13775.1  |     172605 |         0 |        0 |           32732.7 |        350043 |   2023 |           5.88368 |
|    17 | Johnson,Rick E    | Boston Police Department | Police Sergeant (Det)   |    127627 |       0 | 19246.7  |     163204 |         0 |        0 |           31906.9 |        341985 |   2023 |           5.51377 |
|    40 | Medina,Richard L  | Boston Police Department | Police Sergeant (Det)   |    136699 |       0 |  5776.11 |     174080 |         0 |        0 |               0   |        316555 |   2023 |           5.94169 |
|    80 | Acosta,Jose L     | Boston Police Department | Police Officer          |    109502 |       0 | 10665.1  |     174379 |         0 |        0 |               0   |        294546 |   2023 |           5.95346 |
|    24 | Brown,Gregory     | Boston Police Department | Police Detective        |    111584 |       0 | 16762.8  |     179948 |         0 |        0 |               0   |        308294 |   2019 |           6.17259 |

## Conclusion

The analysis reveals that overtime pay constitutes a significant and growing portion of the BPD's budget. Predictive models, especially nonlinear ones like XGBoost, effectively capture the factors influencing overtime spending. The identification of outliers suggests potential areas for policy review or audit to ensure fiscal responsibility.

---
