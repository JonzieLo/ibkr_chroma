import numpy as np
import pandas as pd
from ibkr_chroma.statarb.pca_factors import PCAFactorModel

# Simulate 252 days x 50 stocks with 4 real hidden factors
np.random.seed(42)
T, N = 252, 50
hidden_factors = np.random.normal(0, 0.015, (T, 4))
betas = np.random.uniform(0.5, 1.5, (4, N))
noise = np.random.normal(0, 0.01, (T, N))
returns = pd.DataFrame(hidden_factors @ betas + noise, columns=[f"STOCK_{i}" for i in range(N)])

model = PCAFactorModel(use_rmt=True)
results = model.fit_transform(returns)

print(f"Universe: {N} Stocks over {T} Days")
print(f"Marchenko-Pastur Noise Ceiling (lambda_max): {results.marchenko_pastur_limit:.2f}")
print(f"Top 6 Empirical Eigenvalues: {np.round(results.eigenvalues[:6], 2)}")
print(f"Dynamically Selected Factors K: {results.n_factors} factors")