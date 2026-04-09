import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('../data/home-prices.csv')

# Select relevant columns
X = data[['sqft', 'waterfront']]
y = data['logprice']

# Add constant for intercept
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Interpretation
print("\n=== Coefficient Interpretation ===")
print(f"Intercept (β₀): {model.params['const']:.4f}")
print("  - Expected natural log of price when sqft=0 and waterfront=0")
print(f"Sqft coefficient (β₁): {model.params['sqft']:.6f}")
print("  - Approximate percentage change in price for each additional square foot")
print("  - For a 1 sq ft increase, price changes by approximately", f"{model.params['sqft']*100:.2f}%")
print(f"Waterfront coefficient (β₂): {model.params['waterfront']:.4f}")
print("  - Difference in natural log of price for waterfront properties vs non-waterfront")
print("  - Waterfront properties have prices approximately", f"{(model.params['waterfront']*100):.2f}% higher than non-waterfront")