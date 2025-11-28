import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from travel_preprocess_split import load_train_split

# Reload same split
X_train, X_test, y_train, y_test, id_train, id_test = load_train_split()
print("Loaded preprocessed split. Test shape:", X_test.shape)

model_name = input("Enter model to run on travel dataset (nn / logreg / svm): ").strip().lower()

if model_name == "nn":
    model = joblib.load("travel_nn_model.pkl")
    print("\nLoaded travel_nn_model.pkl")
elif model_name == "logreg":
    model = joblib.load("travel_logreg_model.pkl")
    print("\nLoaded travel_logreg_model.pkl")
elif model_name == "svm":
    model = joblib.load("travel_svm_model.pkl")
    print("\nLoaded travel_svm_model.pkl")
else:
    raise ValueError("Unknown model name. Use nn / logreg / svm.")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1w = f1_score(y_test, y_pred, average="weighted")

print(f"Test Accuracy: {acc:.4f}")
print(f"Weighted F1-score: {f1w:.4f}")
