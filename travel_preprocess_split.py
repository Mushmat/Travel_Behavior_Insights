import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

SEED = 42  # use same seed as elsewhere

from data_preprocessing import add_advanced_features, compute_power_stats


TRAIN_PATH = "train.csv"

def load_train_split(scale=True):
    """Load travel train.csv, engineer features, and create an 80/20 stratified split."""
    train = pd.read_csv(TRAIN_PATH)

    y = train["spend_category"]
    ids = train["trip_id"]
    X = train.drop(columns=["trip_id", "spend_category"])

    
    # Clean labels: drop rows where y is NaN
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    ids = ids.loc[mask].reset_index(drop=True)

    # compute target statistics from full train for target encoding
    train_power = compute_power_stats(train.copy())

    # feature engineering
    X = add_advanced_features(X, train_power)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=SEED, stratify=y
    )

    # categorical columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # label encode categoricals using train+test values
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(all_vals)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # fill numeric NAs
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    if scale:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, id_train.values, id_test.values
    else:
        return X_train.values, X_test.values, y_train.values, y_test.values, id_train.values, id_test.values

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, id_tr, id_te = load_train_split()
    print("Train shape:", X_tr.shape, "Test shape:", X_te.shape)
    print("Train target distribution:\n", pd.Series(y_tr).value_counts())
    print("Test target distribution:\n", pd.Series(y_te).value_counts())
