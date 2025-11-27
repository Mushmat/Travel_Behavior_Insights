import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from data_preprocessing import load_and_preprocess
import warnings
warnings.filterwarnings('ignore')

def create_advanced_datasets(X, y, n_versions=4):
    """Create multiple balanced versions with different strategies"""
    datasets = []
    
    for seed in range(n_versions):
        Xy = pd.DataFrame(X)
        Xy["label"] = y.values if hasattr(y, "values") else y
        counts = Xy["label"].value_counts()
        
        # Different balancing strategies for diversity
        if seed == 0:
            # Aggressive balancing
            target_count = int(counts.quantile(0.90))
        elif seed == 1:
            # Moderate balancing
            target_count = int(counts.quantile(0.80))
        elif seed == 2:
            # Conservative balancing
            target_count = int(counts.quantile(0.75))
        else:
            # Mixed strategy
            target_count = int(counts.quantile(0.85))
        
        resampled = []
        for val in Xy["label"].unique():
            group = Xy[Xy["label"] == val]
            current_count = len(group)
            
            if current_count < target_count:
                # Smart upsampling with cap
                n_samples = min(target_count, int(current_count * 3.0))
                group_up = resample(group, replace=True, n_samples=n_samples, random_state=42+seed*10)
                resampled.append(group_up)
            else:
                resampled.append(group)
        
        Xy_bal = pd.concat(resampled, axis=0).sample(frac=1, random_state=42+seed*10)
        X_bal = Xy_bal.drop("label", axis=1).values
        y_bal = Xy_bal["label"].values.astype(int)
        datasets.append((X_bal, y_bal))
    
    return datasets

# Load and preprocess data
print("="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)
X_train, y_train, X_test, test_ids, _ = load_and_preprocess('train.csv', 'test.csv')

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"\nClass distribution:")
class_dist = pd.Series(y_train).value_counts().sort_index()
print(class_dist)
print(f"\nClass imbalance ratio: {class_dist.max() / class_dist.min():.2f}x")

# Prepare multiple balanced datasets
print("\n" + "="*70)
print("CREATING BALANCED DATASETS")
print("="*70)
datasets = create_advanced_datasets(X_train, y_train, n_versions=4)
for i, (X, y) in enumerate(datasets):
    print(f"\nDataset {i+1}: {X.shape[0]} samples")
    print(pd.Series(y).value_counts().sort_index())

# ==================== NEURAL NETWORK ENSEMBLE ====================

print("\n" + "="*70)
print("TRAINING NEURAL NETWORK ENSEMBLE")
print("="*70)

# Enhanced NN configurations with better diversity
nn_configs = [
    # Very deep networks
    ((512, 384, 256, 128, 64), 'relu', 0.0010, 0.0018, 42),
    ((600, 400, 300, 200, 100), 'relu', 0.0012, 0.0016, 99),
    ((450, 350, 250, 150, 75), 'relu', 0.0014, 0.0020, 123),
    
    # Wide networks
    ((700, 350, 175), 'relu', 0.0011, 0.0019, 17),
    ((550, 275, 137), 'relu', 0.0013, 0.0021, 88),
    ((480, 240, 120), 'relu', 0.0015, 0.0022, 200),
    
    # Balanced with tanh
    ((400, 300, 200, 150, 100, 50), 'tanh', 0.0018, 0.0024, 55),
    ((350, 250, 180, 120, 80, 40), 'tanh', 0.0017, 0.0023, 77),
    ((320, 240, 160, 120, 80, 40), 'tanh', 0.0019, 0.0025, 111),
    
    # Medium depth with different widths
    ((420, 280, 140, 70), 'relu', 0.0016, 0.0020, 133),
    ((380, 260, 130, 65), 'relu', 0.0014, 0.0019, 155),
    ((340, 220, 110, 55), 'relu', 0.0015, 0.0021, 177),
    
    # Additional architectures
    ((500, 400, 300, 200, 100, 50), 'relu', 0.0012, 0.0018, 222),
    ((360, 270, 180, 90, 45), 'relu', 0.0016, 0.0022, 244),
]

nn_probas = []
nn_count = 0

for dataset_idx, (X_bal, y_bal) in enumerate(datasets):
    print(f"\n{'='*60}")
    print(f"DATASET {dataset_idx + 1}/{len(datasets)}")
    print('='*60)
    
    # Train 3-4 NNs per dataset for diversity
    configs_for_dataset = nn_configs[dataset_idx * 3:(dataset_idx * 3) + 4]
    
    for hls, act, alpha, lr, seed in configs_for_dataset:
        nn_count += 1
        print(f"\nNN-{nn_count}: layers={hls}, act={act}, alpha={alpha}, lr={lr}")
        
        nn = MLPClassifier(
            hidden_layer_sizes=hls,
            activation=act,
            solver='adam',
            alpha=alpha,
            learning_rate_init=lr,
            learning_rate='adaptive',
            max_iter=700,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
            random_state=seed,
            batch_size='auto',
            shuffle=True,
            beta_1=0.9,
            beta_2=0.999,
            tol=1e-5
        )
        
        nn.fit(X_bal, y_bal)
        proba = nn.predict_proba(X_test)
        nn_probas.append(proba)
        print(f"  ✓ Loss: {nn.loss_:.4f} | Iters: {nn.n_iter_} | Val score: {nn.best_validation_score_:.4f}")

# ==================== GRADIENT BOOSTING ENSEMBLE ====================

print("\n" + "="*70)
print("TRAINING GRADIENT BOOSTING ENSEMBLE")
print("="*70)

gb_configs = [
    (250, 7, 0.08, 0.85, 42),
    (280, 8, 0.06, 0.82, 99),
    (220, 6, 0.09, 0.80, 123),
    (260, 7, 0.07, 0.83, 17),
    (240, 8, 0.075, 0.85, 88),
    (230, 6, 0.085, 0.81, 55),
    (270, 7, 0.065, 0.84, 200),
    (210, 6, 0.095, 0.80, 222),
]

gb_probas = []

for i, (n_est, depth, lr, subsample, seed) in enumerate(gb_configs):
    dataset_idx = i % len(datasets)
    X_bal, y_bal = datasets[dataset_idx]
    
    print(f"\nGB-{i+1}/{len(gb_configs)}: n_est={n_est}, depth={depth}, lr={lr}, sub={subsample}")
    
    gb = GradientBoostingClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        random_state=seed,
        max_features='sqrt',
        min_samples_split=8,
        min_samples_leaf=3
    )
    
    gb.fit(X_bal, y_bal)
    proba = gb.predict_proba(X_test)
    gb_probas.append(proba)
    print(f"  ✓ Training complete")

# ==================== RANDOM FOREST ENSEMBLE ====================

print("\n" + "="*70)
print("TRAINING RANDOM FOREST ENSEMBLE")
print("="*70)

rf_configs = [
    (450, 26, 42),
    (550, 30, 99),
    (500, 28, 123),
    (420, 25, 17),
    (480, 27, 88),
]

rf_probas = []

for i, (n_est, depth, seed) in enumerate(rf_configs):
    dataset_idx = i % len(datasets)
    X_bal, y_bal = datasets[dataset_idx]
    
    print(f"\nRF-{i+1}/{len(rf_configs)}: n_est={n_est}, depth={depth}")
    
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=seed,
        class_weight='balanced',
        bootstrap=True
    )
    
    rf.fit(X_bal, y_bal)
    proba = rf.predict_proba(X_test)
    rf_probas.append(proba)
    print(f"  ✓ Training complete")

# ==================== EXTRA TREES ENSEMBLE ====================

print("\n" + "="*70)
print("TRAINING EXTRA TREES ENSEMBLE")
print("="*70)

et_configs = [
    (400, 26, 42),
    (450, 28, 99),
    (420, 27, 123),
    (380, 25, 17),
]

et_probas = []

for i, (n_est, depth, seed) in enumerate(et_configs):
    dataset_idx = i % len(datasets)
    X_bal, y_bal = datasets[dataset_idx]
    
    print(f"\nET-{i+1}/{len(et_configs)}: n_est={n_est}, depth={depth}")
    
    et = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=seed,
        class_weight='balanced',
        bootstrap=True
    )
    
    et.fit(X_bal, y_bal)
    proba = et.predict_proba(X_test)
    et_probas.append(proba)
    print(f"  ✓ Training complete")

# ==================== ENSEMBLE COMBINATION ====================

print("\n" + "="*70)
print("COMBINING PREDICTIONS")
print("="*70)

print(f"\nModel counts:")
print(f"  Neural Networks: {len(nn_probas)}")
print(f"  Gradient Boosting: {len(gb_probas)}")
print(f"  Random Forest: {len(rf_probas)}")
print(f"  Extra Trees: {len(et_probas)}")
print(f"  Total: {len(nn_probas) + len(gb_probas) + len(rf_probas) + len(et_probas)}")

# Strategy 1: Balanced ensemble (performed best at 0.69)
balanced_proba = (np.mean(nn_probas, axis=0) * 0.40 + 
                  np.mean(gb_probas, axis=0) * 0.30 + 
                  np.mean(rf_probas, axis=0) * 0.20 +
                  np.mean(et_probas, axis=0) * 0.10)
balanced_preds = np.argmax(balanced_proba, axis=1)

# Strategy 2: NN-dominated (slightly higher NN weight)
nn_dominated_proba = (np.mean(nn_probas, axis=0) * 0.50 + 
                      np.mean(gb_probas, axis=0) * 0.25 + 
                      np.mean(rf_probas, axis=0) * 0.15 +
                      np.mean(et_probas, axis=0) * 0.10)
nn_dominated_preds = np.argmax(nn_dominated_proba, axis=1)

# Strategy 3: Tree-boosted (higher tree model weight)
tree_boosted_proba = (np.mean(nn_probas, axis=0) * 0.35 + 
                      np.mean(gb_probas, axis=0) * 0.35 + 
                      np.mean(rf_probas, axis=0) * 0.20 +
                      np.mean(et_probas, axis=0) * 0.10)
tree_boosted_preds = np.argmax(tree_boosted_proba, axis=1)

# Strategy 4: Weighted by individual weights
nn_weight = 0.45 / len(nn_probas)
gb_weight = 0.28 / len(gb_probas)
rf_weight = 0.17 / len(rf_probas)
et_weight = 0.10 / len(et_probas)

weighted_proba = np.zeros_like(nn_probas[0])
for proba in nn_probas:
    weighted_proba += proba * nn_weight
for proba in gb_probas:
    weighted_proba += proba * gb_weight
for proba in rf_probas:
    weighted_proba += proba * rf_weight
for proba in et_probas:
    weighted_proba += proba * et_weight

weighted_preds = np.argmax(weighted_proba, axis=1)

# Strategy 5: Power ensemble (exponential weighting for confidence)
power_proba = (np.mean(nn_probas, axis=0) ** 1.2) * 0.40
power_proba += (np.mean(gb_probas, axis=0) ** 1.1) * 0.30
power_proba += (np.mean(rf_probas, axis=0) ** 1.0) * 0.20
power_proba += (np.mean(et_probas, axis=0) ** 1.0) * 0.10
# Normalize
power_proba = power_proba / power_proba.sum(axis=1, keepdims=True)
power_preds = np.argmax(power_proba, axis=1)

# ==================== VOTING ENSEMBLE ====================
# Majority voting from top models
all_preds = np.vstack([balanced_preds, nn_dominated_preds, tree_boosted_preds, weighted_preds, power_preds])
voting_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

# ==================== DISPLAY RESULTS ====================

print("\n" + "="*70)
print("PREDICTION DISTRIBUTIONS")
print("="*70)

strategies = {
    'Balanced (0.69 baseline)': balanced_preds,
    'NN-Dominated': nn_dominated_preds,
    'Tree-Boosted': tree_boosted_preds,
    'Weighted': weighted_preds,
    'Power': power_preds,
    'Voting': voting_preds
}

for name, preds in strategies.items():
    print(f"\n{name}:")
    print(pd.Series(preds).value_counts().sort_index())

# ==================== SAVE ALL SUBMISSIONS ====================

print("\n" + "="*70)
print("SAVING SUBMISSIONS")
print("="*70)

submissions = {
    'submission_balanced_v3.csv': balanced_preds,
    'submission_nn_dominated.csv': nn_dominated_preds,
    'submission_tree_boosted.csv': tree_boosted_preds,
    'submission_weighted_v3.csv': weighted_preds,
    'submission_power.csv': power_preds,
    'submission_voting.csv': voting_preds,
}

for filename, preds in submissions.items():
    submission = pd.DataFrame({
        'trip_id': test_ids,
        'category': preds
    })
    submission.to_csv(filename, index=False)
    print(f"✓ {filename}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\n🎯 SUBMISSION PRIORITY (try in this order):")
print("1. submission_power.csv          - Confidence-weighted ensemble")
print("2. submission_balanced_v3.csv    - Improved balanced (0.69 baseline)")
print("3. submission_voting.csv         - Majority voting")
print("4. submission_tree_boosted.csv   - Tree models emphasis")
print("5. submission_nn_dominated.csv   - Neural network emphasis")
print("6. submission_weighted_v3.csv    - Fine-tuned weights")
print("\n💡 Expected best: submission_power.csv or submission_voting.csv")