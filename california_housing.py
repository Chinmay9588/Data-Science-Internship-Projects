# 📌 Simple Linear Regression on Housing Prices (California Housing Dataset)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ----- 1. Load the dataset -----
df = pd.read_csv(r"C:\Users\deshm\OneDrive\Pictures\All Program\Desktop\Data Science Internship\housing.csv")
  # Make sure the file is in the same folder
print("📌 First 5 rows of the dataset:")
print(df.head())

# ----- 2. Check for missing values -----
print("\n📌 Missing values:")
print(df.isnull().sum())

# ----- 3. Basic Statistics -----
print("\n📌 Summary Statistics:")
print(df.describe())

# ----- 4. Feature Selection -----
# Use 'median_income' to predict 'median_house_value'
X = df[['median_income']]   # Feature
y = df['median_house_value']  # Target


# ----- 5. Data Normalization -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- 6. Split Data -----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----- 7. Train Linear Regression Model -----
model = LinearRegression()
model.fit(X_train, y_train)

# ----- 8. Make Predictions -----
y_pred = model.predict(X_test)




# ----- 9. Evaluate the Model -----
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Mean Squared Error: {mse:.4f}")
print(f"📊 R² Score: {r2:.4f}")

# ----- 10. Visualization -----
plt.figure(figsize=(8,5))
sns.scatterplot(x=X_test.flatten(), y=y_test, color='blue', label='Actual')
sns.lineplot(x=X_test.flatten(), y=y_pred, color='red', label='Predicted')
plt.xlabel('Median Income (Standardized)')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression - Housing Prices')
plt.legend()
plt.show()
