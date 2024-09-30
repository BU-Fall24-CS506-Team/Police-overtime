# Police Overtime Project Proposal

## 1. Description of the Project

This project aims to analyze the overtime budget and expenditure patterns within the Boston Police Department (BPD). The BPD operates on an annual budget exceeding $400 million, a portion of which goes toward overtime pay for officers. Our goal is to investigate how the overtime budget is allocated and whether discrepancies exist between overtime hours worked and paid. We will also explore whether certain demographic groups of officers, such as those with longer tenure or specific ranks, have a higher likelihood of overtime discrepancies. By performing this analysis, we hope to provide insights into potential areas of inefficiency or bias in police spending, contributing to ongoing conversations around police accountability and financial transparency.

## 2. Clear Goal(s)

The main objectives of this project are:

- To analyze year-over-year changes in BPD overtime budgets and spending.
- To compare overtime hours worked versus overtime hours paid and identify any discrepancies.
- To explore demographic factors (e.g., officer rank, tenure, gender, and race) and their influence on overtime pay discrepancies.
- To create visualizations that illustrate trends and anomalies in overtime payments.

## 3. Data Collection

The project will leverage several datasets:

- **Employee Earnings Data (2011-2023)**: This dataset will provide detailed information about police earnings, including regular pay, overtime pay, and injury pay.
- **BPD Overtime Data (2012-2022)**: This dataset includes the breakdown of hours worked versus hours paid for overtime, which will allow us to calculate discrepancies and identify outliers.
- **BPD Field Activity Data & Roster**: This dataset will help contextualize overtime data by providing information about officer assignments, ranks, and tenure.
- **Payroll Definitions**: This resource will clarify the payroll categories used in the BPD earnings dataset, ensuring the data is accurately interpreted.

## 4. Data Modeling Plan

We plan to use several modeling approaches:

- **Linear Regression**: This will be used to assess how factors such as tenure, rank, and demographic variables (age, gender, etc.) influence the discrepancy between hours worked and hours paid.
- **Clustering (K-Means)**: To explore groups of officers with similar overtime patterns, clustering techniques will help identify officers who are outliers in terms of hours worked versus hours paid.
- **Outlier Detection**: To identify extreme cases in overtime pay (e.g., officers who are consistently overpaid compared to the hours they work), statistical outlier detection methods will be applied.

## 5. Data Visualization Plan

The data will be visualized using the following methods:

- **Line charts**: To display year-over-year trends in overtime budgets and spending.
- **Scatter plots**: To illustrate the relationship between overtime hours worked versus paid for individual officers.
- **Box plots**: To show the distribution of overtime discrepancies, highlighting officers who are outliers in terms of hours worked versus hours paid.

## 6. Test Plan

To validate the models and findings:

- **Data Splitting**: We will split the data into training (80%) and testing (20%) sets. The training set will be used to build models, and the test set will be reserved for evaluating model performance.
- **Outlier Testing**: As outlier analysis is crucial to this project, we will implement statistical tests (e.g., Z-scores) to identify officers with unusually high or low hours worked vs. hours paid ratios.
- **Time-Based Analysis**: Year-over-year data will be used to test whether patterns are consistent over time, allowing us to explore how factors such as political changes or operational shifts affect overtime pay.
