import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log", encoding="utf-8")]
)
logger = logging.getLogger(__name__)

CSV_FILE = 'data/dataset.csv'
MODEL_FILE = 'artifacts/model.pkl'

logger.info("Loading dataset from %s", CSV_FILE)
df = pd.read_csv(CSV_FILE)
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logger.info("Training samples: %d | Testing samples: %d", X_train.shape[0], X_test.shape[0])

logger.info("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info("Accuracy: %.2f%%", accuracy * 100)
logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
logger.info("Confusion matrix saved to confusion_matrix.png")
plt.show()

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
logger.info("Model saved to %s", MODEL_FILE)
