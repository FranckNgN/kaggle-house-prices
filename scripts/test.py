import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

sns.histplot(train['SalePrice'], kde=True)
plt.title('SalePrice distribution')
plt.show()

print("Skewness:", train['SalePrice'].skew())
print("Kurtosis:", train['SalePrice'].kurt())

missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Visualize missing data
plt.figure(figsize=(10,6))
sns.barplot(x=missing.index, y=missing)
plt.xticks(rotation=90)
plt.title('Missing values in train data')
plt.show()

corr = train.corr()
top_corr = corr['SalePrice'].abs().sort_values(ascending=False)
print(top_corr.head(10))  # Top 10 features correlated with SalePrice

# Visualize strongest correlation (for example GrLivArea)
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.show()