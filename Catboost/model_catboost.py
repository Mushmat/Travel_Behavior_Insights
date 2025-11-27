import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from data_preprocessing import load_and_preprocess  # your function
# NOTE: for CatBoost we will reload raw train/test to get categorical indices
# since load_and_preprocess returns scaled arrays, which CatBoost doesn't need.

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

print("=" * 80)
print("CATBOOST MULTICLASS - TRAVEL DATASET")
print("=" * 80)

# Load raw data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Target and IDs
y = train["spend_category"]
train_ids = train["trip_id"]
test_ids = test["trip_id"]

X = train.drop(columns=["trip_id", "spend_category"])
X_test = test.drop(columns=["trip_id"])

# Feature engineering using your advanced function
from data_preprocessing import add_advanced_features, compute_power_stats

power_stats = compute_power_stats(train.copy())
X = add_advanced_features(X, power_stats)
X_test = add_advanced_features(X_test, power_stats)

# Align columns
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0
for col in X_test.columns:
    if col not in X.columns:
        X[col] = 0
X_test = X_test[X.columns]

# Identify categorical feature indices for CatBoost (before any encoding!)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Categorical columns used by CatBoost ({len(cat_cols)}): {cat_cols}")

# Create CatBoost Pool
train_pool = Pool(X, label=y, cat_features=cat_feature_indices)
test_pool = Pool(X_test, cat_features=cat_feature_indices)

# Strong, regularized CatBoost model for multiclass
cb = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    iterations=1200,
    depth=7,
    learning_rate=0.03,
    l2_leaf_reg=6.0,
    random_state=42,
    bagging_temperature=0.7,
    border_count=128,
    od_type="Iter",
    od_wait=80,
    verbose=200,
    class_weights=None  # set manually if classes are very imbalanced
)

# Optional CV to see local performance
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
f1_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    tr_pool = Pool(X.iloc[tr_idx], label=y.iloc[tr_idx], cat_features=cat_feature_indices)
    val_pool = Pool(X.iloc[val_idx], label=y.iloc[val_idx], cat_features=cat_feature_indices)
    cb_fold = cb.clone()
    cb_fold.fit(tr_pool, eval_set=val_pool, use_best_model=True)
    val_pred = cb_fold.predict(val_pool).astype(int).ravel()
    f1 = f1_score(y.iloc[val_idx], val_pred, average="weighted")
    f1_scores.append(f1)
    print(f"Fold {fold} weighted F1: {f1:.4f}")

print("Mean CV weighted F1:", sum(f1_scores) / len(f1_scores))

# Fit final model on all data
cb.fit(train_pool)

# Predict test classes
test_pred = cb.predict(test_pool).astype(int).ravel()

submission = pd.DataFrame({
    "trip_id": test_ids,
    "category": test_pred
})
submission.to_csv("catboost_travel_submission.csv", index=False)
print("\nSaved: catboost_travel_submission.csv")
