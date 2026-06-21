import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset directly from Kaggle/GitHub source
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first 5 rows
df.head()

# Basic information about the dataset
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Statistics:\n", df.describe())

# Graph 1: Number of passengers who survived and who did not survive
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survived vs Not Survived')
plt.xticks([0,1], ['Not Survived', 'Survived'])
plt.show()

# Graph 2: Survival comparison between males and females
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Male vs Female Survival')
plt.show()

# Graph 3: Age distribution of passengers
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.show()

# Conclusions
print("=== EDA CONCLUSIONS ===")
print(f"Total Passengers: {len(df)}")
print(f"Survived: {df['Survived'].sum()}")
print(f"Not Survived: {len(df) - df['Survived'].sum()}")
print(f"Survival Rate: {df['Survived'].mean()*100:.1f}%")
print(f"Average Age: {df['Age'].mean():.1f} years")
print(f"Male Passengers: {len(df[df['Sex']=='male'])}")
print(f"Female Passengers: {len(df[df['Sex']=='female'])}")