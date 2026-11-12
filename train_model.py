
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("dataset/symptoms.csv")

# 2. Separate features and target
X = df.drop("condition", axis=1)
y = df["condition"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Create models
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 5. Train and evaluate
best_model = None
best_accuracy = -1
best_name = ""

for name, model in models.items():

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# 6. Save selected model
joblib.dump(best_model, "model.pkl")

# 7. Save feature names
joblib.dump(list(X.columns), "features.pkl")

print("\n" + "=" * 45)
print("TRAINING COMPLETED")
print("Selected model:", best_name)
print("Test accuracy:", round(best_accuracy * 100, 2), "%")
print("Model saved as model.pkl")
print("=" * 45)
