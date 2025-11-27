import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from data_preprocessing import load_and_preprocess

def resample_minority(X, y):
    Xy = pd.DataFrame(X)
    Xy["label"] = y.values if hasattr(y, "values") else y
    max_count = Xy["label"].value_counts().max()
    resampled = []
    for val in Xy["label"].unique():
        group = Xy[Xy["label"] == val]
        group_up = resample(group, replace=True, n_samples=max_count, random_state=41)
        resampled.append(group_up)
    Xy_bal = pd.concat(resampled, axis=0).sample(frac=1, random_state=41)
    X_bal = Xy_bal.drop("label", axis=1).values
    y_bal = Xy_bal["label"].values.astype(int)
    return X_bal, y_bal

X_train, y_train, X_test, test_ids, _ = load_and_preprocess('train.csv', 'test.csv')
X_train_bal, y_train_bal = resample_minority(X_train, y_train)
print("SVM: class counts post-resample:", pd.Series(y_train_bal).value_counts())
svc_base = LinearSVC(C=6.0, max_iter=400, dual=False, random_state=41)
svc = CalibratedClassifierCV(svc_base, cv=3)
svc.fit(X_train_bal, y_train_bal)
svc_preds = svc.predict(X_test)
pd.DataFrame({'trip_id': test_ids, 'category': svc_preds}).to_csv('submission_svm_balanced.csv', index=False)
print("submission_svm_balanced.csv written")
