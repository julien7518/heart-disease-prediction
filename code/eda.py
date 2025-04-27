import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

PATH = r'/Users/julien/Desktop/ESILV/A3/Dorset/Python/Data analysis - Assignement 2/data/heart_disease_synthetic_dataset.csv'

#%% Dataset summary
df = pd.read_csv(PATH)
print("+-------------+")
print("| Sample data |")
print("+-------------+")
print(df.head(2))

print("+---------------+")
print("| Dataset infos |")
print("+---------------+")
print(df.info())

print("+--------------------+")
print("| Statistics summary |")
print("+--------------------+")
print(df.describe(include='all'))


print("+----------------+")
print("| Missing values |")
print("+----------------+")
print(df.isnull().sum())

#%% Visualisations
sb.set_theme(style="whitegrid")

plt.figure(figsize=(10, 8))
sb.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("resources/heatmap.png")
plt.close()

df.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature distributions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("resources/distributions.png")
plt.close()

target_corr = df.corr(numeric_only=True)[['target']].drop('target')
plt.figure(figsize=(6, 8))
sb.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, square=True)
plt.title("Feature Correlation with Target")
plt.tight_layout()
plt.savefig("resources/target_heatmap.png")
plt.close()