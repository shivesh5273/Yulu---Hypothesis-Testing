# Yulu Bike Sharing Data Analysis & Predictive Modeling

<div align="center">
  <b>End-to-End Exploratory Data Analysis, Hypothesis Testing & Machine Learning</b>  
</div>


## Project Overview
This project explores and predicts bike rental demand for the Yulu bike sharing system using real-world data.
It covers data cleaning, EDA, statistical hypothesis testing, feature engineering, regression & Random Forest modeling, model interpretability (SHAP), and actionable business insights.

## Files in this Repository
Yulu - Hypothesis Testing.py — Main Python analysis code (well-commented)
bike_sharing.csv — The original dataset (11K+ rows, 12 columns)
Yulu Bike Sharing Data Analysis & Predictive Modeling.pdf — Final business report (full results, plots, and recommendations)

## Project Steps

1. Data Cleaning & Preparation
	•	Loaded, inspected, and converted columns to correct types
	•	Checked for missing values and outliers

2. Exploratory Data Analysis (EDA)
	•	Distribution plots for all numeric & categorical features
	•	Correlation heatmaps and pairplots for insights

3. Statistical Hypothesis Testing
	•	Compared bike rental demand across working days vs. non-working days (t-test)
	•	Assessed the effect of weather and season (ANOVA, chi-square)

4. Predictive Modeling
	•	Linear Regression: Identified significant predictors of rental count
	•	Random Forest: Ranked features by importance; improved predictive accuracy

5. Model Interpretability & Visualization
	•	SHAP values: Interpreted feature impact on predictions
	•	Partial dependence plots: Visualized marginal effect of top features

6. Time Series Analysis
	•	Decomposed rentals into trend, seasonality, and residuals

##  Key Results
	•	Top features: Humidity, temperature, and windspeed drive most demand
	•	Statistical findings: Weather & season significantly affect rentals
	•	Model performance: Random Forest explained ~31% of variance (R² ≈ 0.31)
	•	Business insight: Data-driven optimization can improve fleet and pricing strategy


## Final Report

All findings, plots, code snippets, and recommendations are included in Yulu Bike Sharing Data Analysis & Predictive Modeling.pdf.

##  Business Recommendations
Use weather forecasts and seasonality for dynamic pricing
Prioritize high-demand periods for maintenance & fleet availability
Collect additional features (event data, location, time-of-day) for even better predictions


##  Libraries Used
Python 3.13, pandas, numpy, matplotlib, seaborn
scikit-learn (Random Forest, SHAP, etc.)
statsmodels (linear regression, time series decomposition)
scipy (statistical tests)



## Contact

Author: Shivesh Raj Sahu (Ethan)
