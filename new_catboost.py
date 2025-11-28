"""
Improved CatBoost pipeline for Kaggle competition
- Uses CatBoost's native handling of categorical features (no LabelEncoder)
- Aggressive feature engineering and interactions
- Stratified K-Fold training with out-of-fold (OOF) scoring
- Ensembled fold models (probability averaging)
- Early stopping + saving best models
- Produces submission CSV

Drop-in replace for previous model_catboost.py. Edit FILE PATHS at top as needed.
"""

import os
import sys
import gc
import joblib
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# --------------------------- CONFIG ---------------------------
TRAIN_FILE = "train.csv"      # path to train CSV
TEST_FILE = "test.csv"        # path to test CSV
SUBMISSION_FILE = "submission_catboost_improved.csv"
MODEL_DIR = "catboost_models"
SEED = 42
N_SPLITS = 5
VERBOSE = 200
RANDOM_STATE = SEED

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------- UTIL ---------------------------

def seed_everything(seed: int = 42):
    np.random.seed(seed)


seed_everything(SEED)

# --------------------------- PREPROCESS & FE ENGINEER ---------------------------

def load_and_preprocess(train_path: str, test_path: str):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop missing targets
    train = train.dropna(subset=["spend_category"])
    train["spend_category"] = train["spend_category"].astype(int)

    # Preserve IDs
    train_ids = train["trip_id"]
    test_ids = test["trip_id"]

    # Add placeholder for concatenation
    test["spend_category"] = -1
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # ---------------- Numeric cleanup ----------------
    numeric_fill_zero = [
        "num_females",
        "num_males",
        "mainland_stay_nights",
        "island_stay_nights"
    ]
    for col in numeric_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ---------------- Feature engineering ----------------
    df["total_people"] = df.get("num_females", 0) + df.get("num_males", 0)
    df["total_nights_calc"] = df.get("mainland_stay_nights", 0) + df.get("island_stay_nights", 0)

    # Binary service columns
    binary_cols = [
        "intl_transport_included", "accomodation_included", "food_included",
        "domestic_transport_included", "sightseeing_included", "guide_included",
        "insurance_included"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna("No").map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)

    df["services_count"] = df[[c for c in binary_cols if c in df.columns]].sum(axis=1)

    # Flags
    df["is_alone"] = (df["total_people"] == 1).astype(int)
    if "is_first_visit" in df.columns:
        df["is_first_visit"] = df["is_first_visit"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    df["is_long_trip"] = (df["total_nights_calc"] > 7).astype(int)
    df["is_big_group"] = (df["total_people"] >= 4).astype(int)

    # Interaction features
    df["nights_per_person"] = df["total_nights_calc"] / (df["total_people"] + 1)
    df["services_per_person"] = df["services_count"] / (df["total_people"] + 1)

    # ---------------- trip_id handling ----------------
    # Always treat trip_id as categorical (strings)
    df["trip_id"] = df["trip_id"].astype(str)
    df["trip_id_len"] = df["trip_id"].apply(len)

    # ---------------- Categorical columns ----------------
    manual_cat = [
        "country", "age_group", "travel_companions", "main_activity",
        "visit_purpose", "tour_type", "info_source", "arrival_weather",
        "days_booked_before_trip", "total_trip_days", "has_special_requirements"
    ]

    # keep only those present in df
    categorical_cols = [c for c in manual_cat if c in df.columns]

    # add trip_id as categorical
    categorical_cols.append("trip_id")

    # convert categoricals to string for CatBoost
    for col in categorical_cols:
        df[col] = df[col].fillna("Missing").astype(str)

    # ---------------- Drop unwanted columns ----------------
    drop_cols = ["num_females", "num_males", "mainland_stay_nights", "island_stay_nights"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    df_processed = df.drop(columns=drop_cols)

    # ---------------- Split back ----------------
    train_proc = df_processed[df_processed["spend_category"] != -1].reset_index(drop=True)
    test_proc = df_processed[df_processed["spend_category"] == -1].reset_index(drop=True)

    X = train_proc.drop(columns=["spend_category"])
    y = train_proc["spend_category"]

    X_test = test_proc.drop(columns=["spend_category"])

    # Ensure consistent column alignment
    X, X_test = X.align(X_test, join="left", axis=1, fill_value=np.nan)

    return X, y, X_test, test_ids, categorical_cols


# --------------------------- TRAIN + K-FOLD ---------------------------

def train_and_predict(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_features: List[str]):
    print(f"Starting StratifiedKFold with {N_SPLITS} splits...")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros((X.shape[0], len(y.unique())))
    test_preds = np.zeros((X_test.shape[0], len(y.unique())))

    fold_scores = []

    # Convert categorical names to indices for Pool when required
    cat_feature_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=4,
            random_seed=RANDOM_STATE + fold,
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            thread_count=8,
            verbose=VERBOSE
        )

        # Use Pool objects to pass categorical column indices
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_feature_indices)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_feature_indices)

        model.fit(train_pool,
                  eval_set=val_pool,
                  early_stopping_rounds=200,
                  use_best_model=True)

        # OOF predictions
        val_pred_proba = model.predict_proba(X_val)
        oof_preds[val_idx] = val_pred_proba

        # Score
        val_pred = np.argmax(val_pred_proba, axis=1)
        f1 = f1_score(y_val, val_pred, average='weighted')
        print(f"Fold {fold + 1} weighted F1: {f1:.5f}")
        fold_scores.append(f1)

        # Test predictions (probability average)
        test_pred_proba = model.predict_proba(X_test)
        test_preds += test_pred_proba

        # Save fold model
        model_path = os.path.join(MODEL_DIR, f"catboost_fold{fold + 1}.cbm")
        model.save_model(model_path)
        print(f"Saved model: {model_path}")

        # cleanup
        del model, train_pool, val_pool
        gc.collect()

    # Average test probabilities
    test_preds /= N_SPLITS

    # OOF final preds
    oof_final = np.argmax(oof_preds, axis=1)
    oof_score = f1_score(y, oof_final, average='weighted')
    print(f"\nOOF weighted F1 score: {oof_score:.5f}")
    print(f"Fold mean f1: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # Final test labels
    test_labels = np.argmax(test_preds, axis=1).astype(int)

    return oof_final, oof_score, test_labels


# --------------------------- MAIN ---------------------------

def main():
    X, y, X_test, test_ids, categorical_cols = load_and_preprocess(TRAIN_FILE, TEST_FILE)

    print("Columns:", X.columns.tolist())
    print("Categorical cols detected:", categorical_cols)

    # Train + predict
    oof_preds, oof_score, test_labels = train_and_predict(X, y, X_test, categorical_cols)

    # Save OOF predictions and sample
    oof_df = pd.DataFrame({
        'trip_id': X.get('trip_id'),
        'target_oof': oof_preds
    })
    oof_df.to_csv('oof_predictions.csv', index=False)

    # Build submission
    submission = pd.DataFrame({
        'trip_id': test_ids.values,
        'spend_category': test_labels
    })

    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")


if __name__ == '__main__':
    main()
