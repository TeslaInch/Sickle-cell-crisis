import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("data/sickle_cell_data.csv")

# Features and target
X = df[["pain_level", "hydration", "medication", "temperature", "fatigue"]]
y = df["crisis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("model/crisis_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/crisis_model.pkl")
