# Police Overtime Project Report

## Using the Makefile

Commands

- Install Dependencies

`make install`

- Run Notebook

`make run`


## 1. Description of the Project

This project aims to analyze the overtime budget and expenditure patterns within the Boston Police Department (BPD). The BPD operates on an annual budget exceeding $400 million, a portion of which goes toward overtime pay for officers. Our goal is to investigate how the overtime budget is allocated and whether discrepancies exist between overtime hours worked and paid. We will also explore whether certain demographic groups of officers, such as those with longer tenure or specific ranks, have a higher likelihood of overtime discrepancies. By performing this analysis, we hope to provide insights into potential areas of inefficiency or bias in police spending, contributing to ongoing conversations around police accountability and financial transparency.

## 2. Goal

The main objectives of this project are:

- To analyze year-over-year changes in BPD overtime budgets and spending.
- To analyze BPD overtime budgets and spending on different police saraly plan.
- To identify the amount of injury pay and the percentage of officers taking them.
- To identify outliers in the distribution of ratios of overtime worked vs. overtime paid.
- To explore demographic factors (e.g., officer rank, tenure, gender, and race) and their influence on overtime pay discrepancies.
- To create visualizations that illustrate trends and anomalies in overtime payments.



## 3.Visualizations and analyze of Data

To gain insights into the data, we conducted several visualizations:

1. **Average Overtime by Ethnic Group**

This bar chart shows the average overtime pay grouped by ethnic group categories. Each bar represents a specific ethnic group, and the height of the bar corresponds to the mean overtime earnings for that group.
![average_overtime](https://github.com/user-attachments/assets/fce4c8ae-4f01-4a62-ad77-c34359e64da4)


Insights:

The data reveals significant disparities in overtime earnings among different ethnic groups.
Black employees have the highest average overtime pay, followed closely by Hispanic and White employees.
Asian employees appear to earn lower average overtime pay compared to other groups.
Black employees earn 38% more on overtime than Asian employees.

2. **Average Overtime by Sex**

This bar chart compares the average overtime pay between male and female employees. The height of each bar indicates the mean overtime pay for each gender.
![average_overtime_sex](https://github.com/user-attachments/assets/5b89ecaa-aef8-461d-bf88-b763809e12bb)

Insights:

Male employees earn significantly higher average overtime pay compared to female employees.
This disparity raises questions about workload distribution, availability, or access to overtime opportunities among genders.


3. **Average Regular Pay by Sex**

This bar chart compares the average regular pay between male and female employees. Similar to the overtime chart, the height of each bar indicates the mean regular pay.
![average_regular_sex](https://github.com/user-attachments/assets/d992f6b2-7e30-4d74-ba1d-03e96dc0a1d4)

Insights:

The average regular pay for males and females is relatively similar, suggesting parity in base salaries.
The disparity observed in overtime pay is not reflected in regular pay, hinting at differences in work assignments or additional pay structures.

4. **Average Regular Pay by Ethnic Group**

This bar chart shows the average regular pay grouped by ethnic categories. Each bar represents a specific ethnic group, and its height corresponds to the mean regular earnings.
![average_regular_ethnic](https://github.com/user-attachments/assets/623cf19f-c8bf-4268-99a3-389f7a809929)

Insights:

There is minimal variation in regular pay among different ethnic groups, indicating equitable distribution of base salaries.
This contrasts with the disparities seen in overtime pay, suggesting potential inequities in work assignments or overtime opportunities.

5. **Total Amount Spent on Each Salary Category per Year**

This stacked bar chart visualizes the total yearly expenditure for various salary components, including regular pay, overtime, detail pay, injured pay, and Quinn education incentives. Each bar represents a year, and the different colors correspond to different salary categories.
![toal_spent_each](https://github.com/user-attachments/assets/b4b5dcf9-b056-44a6-b3e3-1d061d34d2e4)
Insights:

Regular pay consistently constitutes the largest share of salary spending.
Overtime spending has increased over time, reflecting operational demands or potential inefficiencies.
Other salary components, such as Quinn education incentives and detail pay, have remained relatively stable.
Spikes in total spending, such as those seen in 2018 and 2023, suggest extraordinary circumstances or policy changes.


6. **Total Amount Spent on Each Salary Category per Year**

This stacked bar chart visualizes the average expenditure for various salary components (e.g., regular pay, overtime, detail pay, injured pay, Quinn education incentives) across different 'Sal Plan' categories. Each bar represents a specific 'Sal Plan', and the stacked sections correspond to different salary components. This allows for a direct comparison of how each category contributes to the overall compensation structure.

**Key Salary Plans:**
- **PC:** Police Captain (senior leadership roles within the department)
- **PD:** Police Detective (officers assigned to investigative duties)
- **PO:** Police Officer (base-level patrol and general duty officers)
- **PSD:** Police Sergeant Detective (sergeants serving in investigative capacities)
- **PSO:** Police Superior Officer (supervisory ranks such as sergeants and lieutenants)

![total_spent sca](https://github.com/user-attachments/assets/f2e98c60-0b39-46df-b4e9-bb1038ca1394)

Insights:
- Overtime and Injured Pay Significance: Overtime and injured pay, though smaller than regular pay, still constitute substantial portions. This may indicate that staffing decisions, scheduling, or departmental policies influence these earnings.
- Variations by Salary Plan: Different salary plans (PC, PD, PO, PSD, PSO) display variations in the composition of pay categories, reflecting differences in roles, responsibilities, and departmental hierarchy.


7. **Total Amount Spent on Overtime per Year**

This line chart illustrates the annual total expenditure on overtime pay from 2014 to 2023. A table below the chart lists the exact overtime amounts for each year.
![total_spent_peryear](https://github.com/user-attachments/assets/854fa122-04f9-427b-baf7-5a47466c15c9)
Insights:

The overtime expenditure shows a steady upward trend, starting at approximately $57.9 million in 2014 and reaching $88.5 million in 2023.
Significant increases are observed in 2018 and 2023, which may correlate with operational or policy changes.
A dip around 2020 might be attributable to pandemic-related disruptions or other unique circumstances.

8. **Injury Pay Analysis**
This analysis examines the proportion of officers receiving injury pay over the last decade and the average total amount paid per officer during that time frame, rather than per individual case. The pie chart below shows that approximately 41% of officers received injury pay from 2014 to 2023. Among those recipients, the average total injury pay over 10 years was about \$84,408.16 per person.
![injury_pay_distribution (1)](https://github.com/user-attachments/assets/d7ad5605-7370-482f-ae43-994b06872407)
Insights:
- A significant portion (41%) of officers received injury pay.
- Each recipient received an average of \$84,408.16 in total over 10 years.



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

### Feature Selection

Method:
Feature selection was conducted through correlation analysis to identify the most relevant features for predicting overtime pay. The correlation matrix below highlights the relationships between different features and overtime pay.

![correlation_matrix](https://github.com/user-attachments/assets/5a96f6d8-e923-4328-9fe0-4db85584f181)

Analysis:

- The feature most strongly correlated with overtime pay is total_gross (correlation = 0.66), which reflects the overall compensation of an employee, including overtime and other pay components.
- Other moderately correlated features include:
regular (correlation = 0.39): Indicates that regular pay has a positive relationship with overtime pay.
quinn_education (correlation = 0.25): Suggests that educational incentives contribute modestly to overtime earnings.
year (correlation = 0.17): Indicates slight temporal variation in overtime patterns.
injured (correlation = 0.15): Suggests a minor relationship between injury compensation and overtime.
- Features with low correlations (e.g., detail, postal) were deemed less relevant for predicting overtime and excluded from further analysis to simplify the model.

### Scaling
Method:

Numerical features were standardized using StandardScaler to ensure that all variables were on the same scale. This step was essential for improving model performance, especially for algorithms sensitive to feature magnitude, such as regression models and clustering algorithms.

Implementation:

- Training data was scaled using scaler.fit_transform() to calculate the scaling parameters (mean and variance).
- Test data was scaled using scaler.transform() to ensure consistency with the training data.

Benefits:

- Scaling prevented features with larger ranges, such as total_gross, from dominating the model during training.
- Improved convergence for optimization algorithms and enhanced model interpretability.

The combination of feature selection through correlation analysis and scaling of numerical features ensured that the models were trained on the most relevant and normalized data. This approach minimized noise, reduced computation complexity, and maximized predictive performance.


### Outlier Detection

- **Z-Score Method**: Calculated Z-scores for `OVERTIME` pay to identify outliers beyond a threshold of 5.
- **Outlier Analysis**: Listed employees with unusually high overtime pay for further investigation.

## 5. Data Modeling Methods

### XGBoost 

- **Objective**: To improve prediction accuracy using a boosted tree model with additional features.
- **Features Used**: TOTAL_GROSS, YEAR, REGULAR, SAL_PLAN, INJURED, SEX, QUINN_EDUCATION, RETRO.
- **Results**:
  - **MAE**: 9,566.74695456772
  - **RMSE**: 13,985.146443003146
  - **R²**: 0.7185522316150666
  - **Adjusted R²**: 0.7178490516815628
- **Interpretation**: XGBoost with an expanded feature set significantly improved prediction performance, explaining about 71.86% of the variance. The inclusion of additional variables like REGULAR, INJURED, QUINN_EDUCATION, and RETRO helped capture more complex patterns in the data.

### CatBoost 

- **Objective**: To handle categorical data effectively and further leverage the additional features.
- **Features Used**: TOTAL_GROSS, YEAR, REGULAR, SAL_PLAN, INJURED, SEX, QUINN_EDUCATION, RETRO.
- **Results**:
  - **MAE**: 9,380.004708631892
  - **RMSE**: 13,530.674605750635
  - **R²**: 0.7365472885490286
  - **Adjusted R²**: 0.7358890681581455
- **Interpretation**: CatBoost, with its adept handling of categorical variables and the expanded feature set, achieved the best performance. It explained approximately 73.65% of the variance, indicating that incorporating these additional features provided substantial improvements in prediction accuracy.

#### we consider the situation that the detail income information maybe hard to get, so we conduct other model only based on slaraly plan, ETHNIC Group, SEX, year, and total income.

### Linear Regression

- **Objective**: To predict overtime pay based on selected features.
- **Features Used**: YEAR, TOTAL_GROSS, SAL_PLAN, ETHNIC_GRP, SEX.
- **Results**:
  - **MAE**: 14,406.1290
  - **RMSE**: 19,305.8103
  - **R²**: 0.4637
  - **Adjusted R²**: 0.4628
- **Interpretation**: Linear regression explained approximately 46.37% of the variance in overtime pay, showing a moderate linear relationship.

### Ridge Regression

- **Objective**: To reduce overfitting and improve upon linear regression using L2 regularization.
- **Features Used**: YEAR, TOTAL_GROSS, SAL_PLAN, ETHNIC_GRP, SEX.
- **Results**:
  - **MAE**: 14,406.2427
  - **RMSE**: 19,305.9194
  - **R²**: 0.4637
  - **Adjusted R²**: 0.4628
- **Interpretation**: Ridge regression yielded nearly identical results to linear regression, indicating minimal benefit from regularization in this case.

### Random Forest

- **Objective**: To capture nonlinear relationships and interactions between features.
- **Features Used**: YEAR, TOTAL_GROSS, SAL_PLAN, ETHNIC_GRP, SEX.
- **Results**:
  - **MAE**: 12,143.8824
  - **RMSE**: 16,659.7777
  - **R²**: 0.6006
  - **Adjusted R²**: 0.6000
- **Interpretation**: Random Forest significantly outperformed the linear models, explaining about 60.06% of the variance, suggesting that nonlinear patterns in the data are important for prediction.

### XGBoost

- **Objective**: To improve predictive accuracy through gradient boosting.
- **Features Used**: YEAR, TOTAL_GROSS, SAL_PLAN, ETHNIC_GRP, SEX.
- **Results**:
  - **MAE**: 11,595.0642
  - **RMSE**: 16,009.6721
  - **R²**: 0.6312
  - **Adjusted R²**: 0.6306
- **Interpretation**: XGBoost delivered better performance than Random Forest and linear models, capturing more complex interactions and achieving a higher R².

### CatBoost

- **Objective**: To handle categorical data effectively and further improve prediction performance.
- **Features Used**: YEAR, TOTAL_GROSS, SAL_PLAN, ETHNIC_GRP, SEX.
- **Results**:
  - **MAE**: 11,391.8519
  - **RMSE**: 15,530.4487
  - **R²**: 0.6529
  - **Adjusted R²**: 0.6524
- **Interpretation**: CatBoost achieved the best overall performance. Its ability to handle categorical variables without extensive preprocessing allowed for the most accurate predictions, explaining about 65.29% of the variance in overtime pay.

### Neural Network

- **Objective**: To capture complex relationships in the data using deep learning.
- **Results**:
  - **MAE**: 16,413.28
  - **RMSE**: 22,018.26
  - **R²**: 0.2342
- **Interpretation**: The neural network underperformed compared to tree-based models, likely due to the limited dataset size and lack of feature engineering specific to deep learning.


## 6. Results

Predictive Modeling: 

Successfully built and evaluated multiple models to predict overtime pay, with CatBoost providing the most accurate results (R² = 0.73654 for all features, 0.6529 for restricted features).

**Outlier Detection**: Identified several employees with overtime pay significantly higher than the norm, warranting further investigation.
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


This project successfully met its objectives by providing a comprehensive analysis of the Boston Police Department's (BPD) overtime budget and spending patterns. The key findings and insights align with the stated goals, offering valuable contributions to the understanding of potential inefficiencies, discrepancies, and biases in overtime allocation. Here is how the project addressed each goal:

1. **Investigating Overtime Budget Allocation and Discrepancies**  
   The analysis revealed significant disparities in overtime pay among demographic groups, as well as outliers with disproportionately high overtime earnings. These findings were achieved through:
   - **Visualizations**: Highlighting overtime trends, such as the steady increase in overtime spending from $57.9 million in 2014 to $88.5 million in 2023, and the disproportionate distribution of overtime pay across ethnic groups and genders.
   - **Outlier Analysis**: Using Z-scores, we identified specific employees with unusually high overtime pay relative to their peers, indicating potential inefficiencies or discrepancies in the allocation process.

2. **Exploring Demographic Disparities in Overtime Pay**  
   The project successfully identified correlations between demographic factors (e.g., ethnicity, gender) and overtime pay:
   - **Ethnic Disparities**: Black employees earned 38% more in overtime than Asian employees, raising questions about workload distribution or access to overtime opportunities.
   - **Gender Disparities**: Male employees consistently earned more overtime pay than their female counterparts, despite similar regular pay, suggesting possible inequities in overtime assignments.
   - **Rank and Tenure**: Outliers were often associated with higher-ranking positions (e.g., Lieutenants and Sergeants), indicating that rank may play a significant role in overtime distribution.

3. **Insights into Potential Inefficiencies and Bias**  
   The combination of predictive modeling and statistical analysis uncovered potential inefficiencies and biases in the allocation of overtime:
   - Models like CatBoost and LightGBM provided strong predictive performance (R² = 0.66 and 0.6573, respectively), emphasizing the importance of regular pay (`REGULAR`) and demographic factors (`SEX`, `ETHNIC_GRP`) in predicting overtime pay.
   - Identified outliers suggest areas where policy revisions or further investigations are necessary to ensure fair and transparent practices.

4. **Contributing to Accountability and Transparency**  
   By creating a detailed dataset, performing advanced modeling, and presenting clear visualizations, this project contributes to the ongoing conversation around police accountability and financial transparency:
   - The findings underscore the need for data-driven approaches to manage payroll inefficiencies and address demographic disparities.
   - The identification of specific outliers provides actionable insights for policy-makers to review and optimize overtime allocation processes.

### Key Findings
- Overtime spending has grown significantly over the past decade, with notable spikes in 2018 and 2023.
- Disparities exist in overtime pay across demographic groups, with males and certain ethnic groups receiving higher overtime compensation.
- High-ranking officers, such as Lieutenants and Sergeants, are often outliers with disproportionately high overtime pay.
- Predictive models highlight regular pay and demographic factors as critical predictors of overtime pay, suggesting systemic patterns in allocation.

### Conclusion
This analysis offers a strong foundation for understanding and addressing inefficiencies and potential biases in overtime allocation within the BPD. By shedding light on these issues, the project paves the way for informed decision-making and policy reforms to promote equity, accountability, and financial transparency in public spending.

---
