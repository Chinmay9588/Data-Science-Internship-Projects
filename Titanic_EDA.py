# Titanic_EDA.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\deshm\OneDrive\Pictures\All Program\Desktop\Data Science Internship\train.csv")

# ----- 1. Basic Info -----
print("📌 First 5 rows of the dataset:")
print(df.head())
print("\n📌 Dataset Info:")
print(df.info())
print("\n📌 Missing Values:")
print(df.isnull().sum())

# ----- 2. Handle Missing Values -----
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

# ----- 3. Basic Statistics -----
print("\n📌 Summary Statistics:")
print(df.describe())

# ----- 4. Visualizations -----
sns.countplot(x='Pclass', data=df, palette='pastel', hue=None, legend=False)
plt.title('Passenger Class Distribution')
plt.show()

sns.countplot(x='Survived', data=df, palette='muted', hue=None, legend=False)
plt.title('Survival Count')
plt.show()

sns.histplot(df['Age'], kde=True, bins=30, color='teal')
plt.title('Age Distribution')
plt.show()

# ✅ Fixed: Correlation Heatmap with only numeric columns
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Gender')
plt.show()
