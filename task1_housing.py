import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance into units of 100 meters
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

# Select input and target
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# (a) Slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# (b) Predict price at 500 m
new_house = pd.DataFrame({"distance_100m": [5]})
prediction = model.predict(new_house)
print("Predicted price at 500 meters:", prediction[0])

# (c) Plot data and fitted line
plt.scatter(X["distance_100m"], y, label="Actual Data")
plt.plot(X["distance_100m"], model.predict(X), color="red", label="Regression Line")
plt.xlabel("Distance to Nearest MRT Station (100 meters)")
plt.ylabel("House Price per Unit Area")
plt.title("Housing Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)

plt.savefig("task1_housing.png")
plt.show()