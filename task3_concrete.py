import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data and remove identical rows
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

# (a) 20% test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

# Linear baseline
linear_rmse = -cross_val_score(
    LinearRegression(), X_train, y_train, cv=folds, scoring=scoring
)
print("Linear fold RMSE:", linear_rmse.round(2))
print("Linear mean CV RMSE:", round(linear_rmse.mean(), 2))

# (b) Polynomial + scaling + Ridge, grid search
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])

grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}

search = GridSearchCV(ridge_model, grid, cv=folds, scoring=scoring, refit=True)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best mean CV RMSE:", round(-search.best_score_, 2))

results = pd.DataFrame(search.cv_results_)
top_three = results.sort_values("rank_test_score").head(3)
print("Top three settings:")
print(top_three[[
    "param_poly__degree", "param_ridge__alpha",
    "rank_test_score", "mean_test_score",
]].to_string(index=False))

# (c) Test RMSE and MAE
test_predictions = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)
print("Test RMSE:", round(test_rmse, 2))
print("Test MAE:", round(test_mae, 2))

# Plot actual vs predicted with y = x line
plt.scatter(y_test, test_predictions, label="Predictions")
low = min(y_test.min(), test_predictions.min())
high = max(y_test.max(), test_predictions.max())
plt.plot([low, high], [low, high], color="red", label="y = x")
plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.legend()
plt.grid(True)

plt.savefig("task3_concrete.png")
plt.show()