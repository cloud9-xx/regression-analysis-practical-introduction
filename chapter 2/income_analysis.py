import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the income dataset
df = pd.read_csv('../data/income.csv')

# Display first few rows
print("Data Summary:")
print(df[['income', 'educ', 'afqt', 'age']].head())
print(f"\nDataset shape: {df.shape}\n")

# Prepare data
income = df['income'].values.reshape(-1, 1)

# Model 1: Income ~ Education (educ)
educ = df['educ'].values.reshape(-1, 1)
model_educ = LinearRegression()
model_educ.fit(educ, income)
r2_educ = r2_score(income, model_educ.predict(educ))

print("=" * 50)
print("MODEL 1: Income ~ Education")
print("=" * 50)
print(f"Coefficient (slope): {model_educ.coef_[0][0]:.2f}")
print(f"Intercept: {model_educ.intercept_[0]:.2f}")
print(f"R² (Coefficient of Determination): {r2_educ:.4f}")
print(f"Interpretation: Education explains {r2_educ*100:.2f}% of variation in income\n")

# Model 2: Income ~ AFQT (afqt)
afqt = df['afqt'].values.reshape(-1, 1)
model_afqt = LinearRegression()
model_afqt.fit(afqt, income)
r2_afqt = r2_score(income, model_afqt.predict(afqt))

print("=" * 50)
print("MODEL 2: Income ~ AFQT")
print("=" * 50)
print(f"Coefficient (slope): {model_afqt.coef_[0][0]:.2f}")
print(f"Intercept: {model_afqt.intercept_[0]:.2f}")
print(f"R² (Coefficient of Determination): {r2_afqt:.4f}")
print(f"Interpretation: AFQT explains {r2_afqt*100:.2f}% of variation in income\n")

# Model 3: Income ~ Age
age = df['age'].values.reshape(-1, 1)
model_age = LinearRegression()
model_age.fit(age, income)
r2_age = r2_score(income, model_age.predict(age))

print("=" * 50)
print("MODEL 3: Income ~ Age")
print("=" * 50)
print(f"Coefficient (slope): {model_age.coef_[0][0]:.2f}")
print(f"Intercept: {model_age.intercept_[0]:.2f}")
print(f"R² (Coefficient of Determination): {r2_age:.4f}")
print(f"Interpretation: Age explains {r2_age*100:.2f}% of variation in income\n")

# Summary and comparison
print("=" * 50)
print("SUMMARY: Which Variable Explains Most Variation?")
print("=" * 50)
results = {
    'Education (educ)': r2_educ,
    'AFQT (afqt)': r2_afqt,
    'Age': r2_age
}

# Sort by R² value
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for rank, (variable, r2) in enumerate(sorted_results, 1):
    print(f"{rank}. {variable}: R² = {r2:.4f} ({r2*100:.2f}%)")

best_variable = sorted_results[0][0]
best_r2 = sorted_results[0][1]
print(f"\n✓ {best_variable} explains the greatest amount of variation in income")
print(f"  with R² = {best_r2:.4f}")
