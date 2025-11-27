import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from data_preprocessing import load_and_preprocess

def resample_minority(X, y):
    Xy = pd.DataFrame(X)
    Xy["label"] = y.values if hasattr(y, "values") else y
    max_count = Xy["label"].value_counts().max()
    resampled = []
    for val in Xy["label"].unique():
        group = Xy[Xy["label"] == val]
        group_up = resample(group, replace=True, n_samples=max_count, random_state=10)
        resampled.append(group_up)
    Xy_bal = pd.concat(resampled, axis=0).sample(frac=1, random_state=10)
    X_bal = Xy_bal.drop("label", axis=1).values
    y_bal = Xy_bal["label"].values.astype(int)
    return X_bal, y_bal

X_train, y_train, X_test, test_ids, _ = load_and_preprocess('train.csv', 'test.csv')
X_train_bal, y_train_bal = resample_minority(X_train, y_train)

# Ensemble of KNN with different k values
k_values = [15, 25, 35, 45]
test_probas = []
for k in k_values:
    print(f"Training KNN with k={k}")
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean', n_jobs=-1)
    knn.fit(X_train_bal, y_train_bal)
    test_probas.append(knn.predict_proba(X_test))

avg_probas = np.mean(test_probas, axis=0)
final_preds = np.argmax(avg_probas, axis=1)
pd.DataFrame({'trip_id': test_ids, 'category': final_preds}).to_csv('submission_knn_ensemble.csv', index=False)
print("submission_knn_ensemble.csv written")
