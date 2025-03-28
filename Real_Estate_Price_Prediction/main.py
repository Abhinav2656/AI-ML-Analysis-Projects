import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Read the CSV file into a DataFrame called real_estate_data
real_estate_data = pd.read_csv("Real_Estate.csv")
real_estate_data = real_estate_data.dropna()
sns.set_style("whitegrid")


# Create histograms for numerical problems
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig. suptitle('Histograms of Real Estate Data', fontsize=16)

cols=['House age', 'Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude',
      'House price of unit area']

for i, col in enumerate(cols):
    sns.histplot(real_estate_data[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(col)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()


# Create scatter plots to observe relationship with house price
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
fig.suptitle('Scatter Plots with House Price of unit Area', fontsize=16)

sns.scatterplot(data=real_estate_data, x='House age', y='House price of unit area', ax=axes[0,0])
sns.scatterplot(data=real_estate_data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0,1])
sns.scatterplot(data=real_estate_data, x='Number of convenience stores', y='House price of unit area', ax=axes[1,0])
sns.scatterplot(data=real_estate_data, x='Latitude', y='House price of unit area', ax=axes[1,1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Correlation Matrix
correlation_matrix = real_estate_data.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
print(correlation_matrix)


# Linear Regression Model
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

X = real_estate_data[features]
y = real_estate_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Making Predictions
y_pred_lr = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Actual Vs Predicted House Prices')
plt.show()

