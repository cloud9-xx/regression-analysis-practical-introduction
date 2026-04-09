import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([10, 12, 14, 16], dtype=float)
Y = np.array([40, 45, 40, 55], dtype=float)
weights = np.array([1, 2, 3, 4], dtype=float)

ols = LinearRegression()
ols.fit(X.reshape(-1, 1), Y, sample_weight=weights)
print("\n=== scikit-learn WLS ===")
print(f"Intercept (β₀): {ols.intercept_:.4f}")
print(f"Slope     (β₁): {ols.coef_[0]:.4f}")
