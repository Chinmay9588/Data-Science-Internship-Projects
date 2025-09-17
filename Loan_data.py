import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample loan eligibility dataset
np.random.seed(42)
rows = 100

loan_data = pd.DataFrame({
    "Loan_ID": [f"LP00{i:03d}" for i in range(1, rows+1)],
    "Gender": np.random.choice(["Male", "Female"], size=rows),
    "Married": np.random.choice(["Yes", "No"], size=rows),
    "Dependents": np.random.choice(["0", "1", "2", "3+"], size=rows),
    "Education": np.random.choice(["Graduate", "Not Graduate"], size=rows),
    "Self_Employed": np.random.choice(["Yes", "No"], size=rows),
    "ApplicantIncome": np.random.randint(1500, 25000, size=rows),
    "CoapplicantIncome": np.random.randint(0, 10000, size=rows),
    "LoanAmount": np.random.randint(50, 700, size=rows),
    "Loan_Amount_Term": np.random.choice([360, 180, 120, 60], size=rows),
    "Credit_History": np.random.choice([1.0, 0.0], size=rows, p=[0.8, 0.2]),
    "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], size=rows),
    "Loan_Status": np.random.choice(["Y", "N"], size=rows, p=[0.7, 0.3])
})

# Save the dataset
file_path = "loan_data.csv"   # saves in same folder as script
loan_data.to_csv(file_path, index=False)

# ---------------- EDA OUTPUTS ----------------
print("✅ loan_data.csv saved successfully at:", file_path)

# 1. Show first 5 rows
print("\n🔹 First 5 rows of dataset:\n", loan_data.head())

# 2. Print shape
print("\n🔹 Dataset Shape:", loan_data.shape)

# 3. Show column names & data types
print("\n🔹 Dataset Info:")
print(loan_data.info())

# 4. Show missing values count
print("\n🔹 Missing Values:\n", loan_data.isnull().sum())

# 5. Summary statistics for numerical columns
print("\n🔹 Summary Statistics:\n", loan_data.describe())

# 6. Value counts of target variable
print("\n🔹 Loan_Status Value Counts:\n", loan_data['Loan_Status'].value_counts())

# ---------------- VISUALIZATIONS ----------------
plt.figure(figsize=(14,8))

# Histogram of ApplicantIncome
plt.subplot(2,2,1)
plt.hist(loan_data["ApplicantIncome"], bins=20, color="skyblue", edgecolor="black")
plt.title("Applicant Income Distribution")

# Bar chart for Loan_Status
plt.subplot(2,2,2)
loan_data["Loan_Status"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Loan Status Count")
plt.xlabel("Loan Status")
plt.ylabel("Count")

# Boxplot for LoanAmount
plt.subplot(2,2,3)
sns.boxplot(x="Loan_Status", y="LoanAmount", data=loan_data,
            hue="Loan_Status", palette="Set2", legend=False)
plt.title("Loan Amount by Loan Status")

# Correlation Heatmap
plt.subplot(2,2,4)
sns.heatmap(loan_data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()
