import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from data_preprocessing import load_and_preprocess

def resample_minority(X, y):
    Xy = pd.DataFrame(X)
    Xy["label"] = y.values if hasattr(y, "values") else y
    max_count = Xy["label"].value_counts().max()
    resampled = []
    for val in Xy["label"].unique():
        group = Xy[Xy["label"] == val]
        group_up = resample(group, replace=True, n_samples=max_count, random_state=21)
        resampled.append(group_up)
    Xy_bal = pd.concat(resampled, axis=0).sample(frac=1, random_state=21)
    X_bal = Xy_bal.drop("label", axis=1).values
    y_bal = Xy_bal["label"].values.astype(int)
    return X_bal, y_bal

X_train, y_train, X_test, test_ids, _ = load_and_preprocess('train.csv', 'test.csv')
X_train_bal, y_train_bal = resample_minority(X_train, y_train)

# Base models
knn = KNeighborsClassifier(n_neighbors=25, weights='distance', n_jobs=-1)
svm = LinearSVC(C=4.0, max_iter=350, dual=False, random_state=42)

# Generate meta-features using out-of-fold predictions
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_train_knn = np.zeros((X_train_bal.shape[0], 3))
meta_train_svm = np.zeros((X_train_bal.shape[0], 3))

print("Generating meta-features for stacking...")
for train_idx, val_idx in cv.split(X_train_bal, y_train_bal):
    knn.fit(X_train_bal[train_idx], y_train_bal[train_idx])
    meta_train_knn[val_idx] = knn.predict_proba(X_train_bal[val_idx])
    
    svm.fit(X_train_bal[train_idx], y_train_bal[train_idx])
    # SVM decision function for probability-like scores
    svm_scores = svm.decision_function(X_train_bal[val_idx])
    if svm_scores.ndim == 1:
        svm_scores = svm_scores.reshape(-1, 1)
    # Normalize to 3 classes
    from scipy.special import softmax
    meta_train_svm[val_idx] = softmax(svm_scores, axis=1) if svm_scores.shape[1] > 1 else np.hstack([1-svm_scores, svm_scores, np.zeros_like(svm_scores)])

# Refit on full train
knn.fit(X_train_bal, y_train_bal)
svm.fit(X_train_bal, y_train_bal)

# Test meta-features
meta_test_knn = knn.predict_proba(X_test)
svm_scores_test = svm.decision_function(X_test)
if svm_scores_test.ndim == 1:
    svm_scores_test = svm_scores_test.reshape(-1, 1)
from scipy.special import softmax
meta_test_svm = softmax(svm_scores_test, axis=1) if svm_scores_test.shape[1] > 1 else np.hstack([1-svm_scores_test, svm_scores_test, np.zeros_like(svm_scores_test)])

# Stack features
meta_train = np.hstack([meta_train_knn, meta_train_svm])
meta_test = np.hstack([meta_test_knn, meta_test_svm])

# Meta-model: Logistic Regression
meta_model = LogisticRegression(C=1.0, max_iter=200, multi_class='ovr')
meta_model.fit(meta_train, y_train_bal)
final_preds = meta_model.predict(meta_test)

pd.DataFrame({'trip_id': test_ids, 'category': final_preds}).to_csv('submission_knn_svm_stack.csv', index=False)
print("submission_knn_svm_stack.csv written (Stacking)")
