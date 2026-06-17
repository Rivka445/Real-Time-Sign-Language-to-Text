import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = 'dataset.csv'

if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found! Make sure you ran data_collection.py first.")
    exit()

print("Loading dataset...")
df = pd.read_csv(CSV_FILE, encoding='utf-8')

print(f"\nTotal rows (samples): {df.shape[0]}")
print(f"Total columns (features): {df.shape[1]}")

print("\n=== Label Distribution ===")
class_counts = df['label'].value_counts()
print(class_counts)

missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"\nWarning: {missing_values} missing values found. Dropping them...")
    df = df.dropna()
else:
    print("\nNo missing values found.")

plt.figure(figsize=(10, 5))
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Signs in Dataset', fontsize=14)
plt.xlabel('Letter / Gesture', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

graph_path = 'dataset_distribution.png'
plt.savefig(graph_path)
print(f"Graph saved as '{graph_path}'")
try:
    plt.show()
except KeyboardInterrupt:
    pass
