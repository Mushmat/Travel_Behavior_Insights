import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from travel_preprocess_split import load_train_split
from sklearn.utils import resample
import pandas as pd

SEED = 42

# Load preprocessed split
X_train, X_test, y_train, y_test, id_train, id_test = load_train_split()
print("Training data shape:", X_train.shape)

# simple upsampling to balance classes (optional but helpful)
def balance_multiclass(X, y):
    Xy = pd.DataFrame(X)
    Xy["label"] = y
    max_count = Xy["label"].value_counts().max()
    resampled = []
    for cls in Xy["label"].unique():
        grp = Xy[Xy["label"] == cls]
        grp_up = resample(grp, replace=True, n_samples=max_count, random_state=SEED)
        resampled.append(grp_up)
    Xy_bal = pd.concat(resampled, axis=0).sample(frac=1.0, random_state=SEED)
    X_bal = Xy_bal.drop("label", axis=1).values
    y_bal = Xy_bal["label"].values
    return X_bal, y_bal

X_train_bal, y_train_bal = balance_multiclass(X_train, y_train)
print("Balanced train counts:\n", pd.Series(y_train_bal).value_counts())

# 1) Neural Network (multiclass)
nn = MLPClassifier(
    hidden_layer_sizes=(160, 80, 40),
    activation='relu',
    solver='adam',
    alpha=0.0025,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.18,
    n_iter_no_change=20,
    learning_rate_init=0.003,
    learning_rate='adaptive',
    random_state=SEED,
    verbose=False
)
nn.fit(X_train_bal, y_train_bal)
joblib.dump(nn, "travel_nn_model.pkl")
print("Saved travel_nn_model.pkl")

# 2) Multinomial Logistic Regression
logreg = LogisticRegression(
    C=1.5,
    max_iter=600,
    multi_class='multinomial',
    solver='lbfgs'
)
logreg.fit(X_train_bal, y_train_bal)
joblib.dump(logreg, "travel_logreg_model.pkl")
print("Saved travel_logreg_model.pkl")

# 3) Linear SVM + calibration (for multiclass probabilities)
base_svm = LinearSVC(C=1.0, max_iter=1200, random_state=SEED)
svm = CalibratedClassifierCV(base_svm, cv=3)
svm.fit(X_train_bal, y_train_bal)
joblib.dump(svm, "travel_svm_model.pkl")
print("Saved travel_svm_model.pkl")

print("All three travel models trained and saved.")
