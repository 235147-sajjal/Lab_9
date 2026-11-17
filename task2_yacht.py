import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

# Load headerless dataset
yachts = pd.read_csv("yacht_hydrodynamics.data", sep=r"\s+", header=None)

X = yachts[[5]]   # Froude number
y = yachts[6]     # Residuary resistance

# (a) Linear baseline and degree-2 polynomial pipeline
linear_model = LinearRegression()
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression()),
])

# (b) Same five shuffled folds, negative RMSE scoring
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(linear_model, X, y, cv=folds, scoring=scoring)
quadratic_rmse = -cross_val_score(quadratic_model, X, y, cv=folds, scoring=scoring)

print("Linear fold RMSE:", linear_rmse.round(2))
print("Linear mean RMSE:", round(linear_rmse.mean(), 2))
print("Quadratic fold RMSE:", quadratic_rmse.round(2))
print("Quadratic mean RMSE:", round(quadratic_rmse.mean(), 2))

# (c) Choose better model
if quadratic_rmse.mean() < linear_rmse.mean():
    print("Quadratic Regression is the better model.")
else:
    print("Linear Regression is the better model.")