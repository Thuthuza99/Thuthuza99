import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = r"C:\Users\w8874882\Downloads\Avocado_HassAvocadoBoard.csv"
df = pd.read_csv(file_path)

print("Head of the dataset:")
print(df.head())

print("\nShape of dataset:", df.shape)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Iterate over missing rows
missing_rows = df[df.isnull().any(axis=1)]
print("\nMissing rows:")
print(missing_rows)

df = df.drop_duplicates()

df.fillna(0, inplace=True)

print("\nNumber of unique values per column:")
print(df.nunique())

# Bar plot of average price per type
plt.figure(figsize=(8, 5))
sns.barplot(x="type", y="AveragePrice", data=df, estimator=sum)
plt.title("Total Average Price per Type")
plt.xlabel("Type")
plt.ylabel("Total Average Price")

# Add data values on top of bars
for index, row in enumerate(df.groupby("type")["AveragePrice"].sum()):
    plt.text(index, row, round(row, 2), ha='center', fontsize=12)

plt.show()

columns_to_sum = ["TotalVolume", "TotalBags", "SmallBags", "LargeBags", "XLargeBags"]
column_sums = df[columns_to_sum].sum()
print("\nSum of selected columns:")
print(column_sums)

type_distribution = df["type"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(type_distribution, labels=type_distribution.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title("Distribution of Avocado Type")
plt.show()


# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
df['region'] = le.fit_transform(df['region'])

# Select Features and Target Variable
features = ['TotalVolume', 'TotalBags', 'SmallBags', 'LargeBags', 'XLargeBags', 'type', 'region', 'Year', 'Month', 'Week']
target = 'AveragePrice'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42, shuffle=False)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Square Error (RMSE): {rmse}")

# Plot actual vs predicted prices
plt.figure(figsize=(10,5))
plt.plot(df['Date'][-len(y_test):], y_test, label="Actual Prices", marker='o')
plt.plot(df['Date'][-len(y_test):], y_pred, label="Predicted Prices", linestyle='dashed', marker='x')
plt.xlabel("Date")
plt.ylabel("Average Price")
plt.title("Actual vs Predicted Avocado Prices")
plt.legend()
plt.show()

