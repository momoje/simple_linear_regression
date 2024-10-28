import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
print(df.head())
print(df.info())


# Set up the plot
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
# axes = axes.flatten()

# # Generate scatter plots for each feature against PRICE
# for idx, feature in enumerate(housing.feature_names):
#     axes[idx].scatter(df[feature], df['Price'], alpha=0.5)
#     axes[idx].set_xlabel(feature)
#     axes[idx].set_ylabel('Price')
#     axes[idx].set_title(f'{feature} vs Price')

# # Remove any extra subplot (since we have 8 features, not 9)
# for i in range(len(housing.feature_names), len(axes)):
#     fig.delaxes(axes[i])

#plt.tight_layout()
#plt.show()

X = df[['MedInc']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 12)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")

plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('MedInc')
plt.ylabel('Price')
plt.legend()
plt.show()

