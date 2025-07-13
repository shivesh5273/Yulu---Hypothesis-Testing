import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels

print("All libraries loaded successfully!")

# STEP 1: Load the Data
# Loading the Yulu dataset
df = pd.read_csv("bike_sharing.csv")

# Display the top 5 rows, shape, info
print("\nFirst 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nInfo:\n")
print(df.info())

# STEP 2: Convert Categorical Columns
# Convert columns to 'category' dtype where appropriate
cat_cols = ['season', 'holiday', 'workingday', 'weather']
for col in cat_cols:
    df[col] = df[col].astype('category')

print("\nUpdated info (after category conversion):\n")
print(df.info())

# STEP 3: Check for Missing Values
print("\nMissing values per column:\n", df.isnull().sum())

# STEP 4: Statistical Summary
print("\nStatistical Summary (numerics):\n", df.describe())
print("\nStatistical Summary (categoricals):\n", df.describe(include='category'))

# STEP 5: Univariate Analysis - Continuous Variables
# Plotting Continuous Variables
num_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    print(f"\nObservation: Check for skewness, outliers, and range in {col}.")

# STEP 6: Univariate Analysis - Categorical Variables
# Plotting Categorical Variables
cat_cols = ['season', 'holiday', 'workingday', 'weather']

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f'Countplot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
    print(f"\nObservation: Check balance/distribution in {col}.")

# STEP 7: Bivariate Analysis - Count vs. Categorical Variables

cat_cols = ['season', 'holiday', 'workingday', 'weather']

for col in cat_cols:
    plt.figure(figsize=(7,5))
    sns.boxplot(x=col, y='count', data=df)
    plt.title(f'Bike Rentals by {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel('Total Rentals (count)')
    plt.show()
    print(f"\nObservation: How does demand (count) vary across {col}? Look for higher medians, "
          f"wider ranges, or outliers in each category.\n")

# STEP 8: Mean Rentals by Group
for col in cat_cols:
    group_means = df.groupby(col)['count'].mean()
    print(f"\nMean Rentals by {col.capitalize()}:\n{group_means}\n")

# STEP 9: HYPOTHESIS TESTING
from scipy.stats import ttest_ind, shapiro, levene

# Split data
working = df[df['workingday'] == 1]['count']
non_working = df[df['workingday'] == 0]['count']

# Normality test (Shapiro-Wilk)
print("Shapiro-Wilk Test (Working Day):", shapiro(working))
print("Shapiro-Wilk Test (Non-Working Day):", shapiro(non_working))

# Variance test (Levene)
print("Levene Test for equal variances:", levene(working, non_working))

# T-test (Welch's if variances unequal)
t_stat, p_val = ttest_ind(working, non_working, equal_var=False)
print(f"\nT-Test Result: t-stat={t_stat:.2f}, p-value={p_val:.4f}")

# STEP 10: Hypothesis Testing – ANOVA for Season and Weather

from scipy.stats import f_oneway

# ANOVA for Season
groups_season = [df[df['season'] == s]['count'] for s in df['season'].cat.categories]
anova_season = f_oneway(*groups_season)
print(f"\nANOVA (Season): F-stat={anova_season.statistic:.2f}, p-value={anova_season.pvalue:.4f}")

# ANOVA for Weather
groups_weather = [df[df['weather'] == w]['count'] for w in df['weather'].cat.categories]
anova_weather = f_oneway(*groups_weather)
print(f"\nANOVA (Weather): F-stat={anova_weather.statistic:.2f}, p-value={anova_weather.pvalue:.4f}")

# Interpretation:
print("\nInterpretation: If p-value < 0.05, at least one group's mean is different, "
      "meaning season or weather affects rentals.")

# STEP 11: Hypothesis Testing – Chi-Square Test (Season vs Weather)

from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['season'], df['weather'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Test (Season vs Weather): chi2={chi2:.2f}, p-value={p:.4f}, dof={dof}")
print("\nInterpretation: If p-value < 0.05, weather and season are related (not independent).")

# STEP 12: Correlation Heatmap (Advanced)
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# STEP 13: Pairplot (Optional)
sns.pairplot(df[num_cols])
plt.show()

# STEP 14: Outlier Detection in 'count'
Q1 = df['count'].quantile(0.25)
Q3 = df['count'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['count'] < Q1 - 1.5*IQR) | (df['count'] > Q3 + 1.5*IQR)]
print(f"\nNumber of outliers in count: {len(outliers)}")
print("Recommendation: For business insight, keep all data;"
      " but you may mention robust methods for predictive modeling.")

# STEP 15: Simple Linear Regression (Bonus)
import statsmodels.formula.api as smf

model = smf.ols('count ~ temp + atemp + humidity + windspeed + C(season) + C(weather) + C(workingday)',
                data=df).fit()
print(model.summary())

# ADVANCED STEP: Random Forest Feature Importance

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Prepare data
X = df[['temp', 'atemp', 'humidity', 'windspeed', 'season', 'holiday', 'workingday', 'weather']]
y = df['count']
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest Feature Importances:\n", importances)
importances.plot(kind='bar', figsize=(10,5))
plt.title("Random Forest Feature Importances")
plt.show()

# ADVANCED STEP: SHAP for Model Interpretability

import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# ADVANCED STEP: Residual Analysis

y_pred = rf.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution (y_test - y_pred)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.4f}")

# ADVANCED STEP: Time Series Decomposition (Seasonality/Trend)
import statsmodels.api as sm

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

decomposition = sm.tsa.seasonal_decompose(df['count'], model='additive', period=24*30) # ~monthly
decomposition.plot()
plt.show()

from sklearn.inspection import PartialDependenceDisplay

features = ['temp', 'atemp', 'humidity', 'windspeed']
fig, ax = plt.subplots(figsize=(12,8))
PartialDependenceDisplay.from_estimator(rf, X_test, features, ax=ax)
plt.tight_layout()
plt.show()
