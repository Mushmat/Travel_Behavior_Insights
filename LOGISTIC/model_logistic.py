import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
print("LR: class counts post-resample:", pd.Series(y_train_bal).value_counts())
lr = LogisticRegression(C=2.0, max_iter=210, multi_class='ovr', solver='liblinear')
lr.fit(X_train_bal, y_train_bal)
lr_pred = lr.predict(X_test)
pd.DataFrame({'trip_id': test_ids, 'category': lr_pred}).to_csv('submission_logreg_balanced.csv', index=False)
print("submission_logreg_balanced.csv written")
