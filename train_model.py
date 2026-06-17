import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_FILE = 'dataset.csv'
MODEL_FILE = 'model.pkl'

print("📊 Loading dataset and preparing data...")
df = pd.read_csv(CSV_FILE)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"🔹 Training samples: {X_train.shape[0]}")
print(f"🔹 Testing samples: {X_test.shape[0]}")

print("\n🤖 Training Random Forest model (Version 1)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy on test set: {accuracy:.2%}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

print("📈 Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f'Confusion Matrix - Sign Language (Accuracy: {accuracy:.2%})', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

cm_path = 'confusion_matrix.png'
plt.savefig(cm_path)
print(f"✅ Confusion matrix successfully saved as '{cm_path}'")
plt.show()

print(f"\n💾 Saving the trained model to '{MODEL_FILE}'...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print("🎉 Model training completed successfully! The model is ready for live deployment.")