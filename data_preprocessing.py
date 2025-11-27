import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from scipy import stats

def add_advanced_features(df, train_power=None):
    df = df.copy()
    
    # ==================== BASIC AGGREGATIONS ====================
    df['total_travelers'] = df['num_females'] + df['num_males']
    df['is_group'] = (df['total_travelers'] > 1).astype(int)
    df['is_solo'] = (df['total_travelers'] == 1).astype(int)
    df['is_large_group'] = (df['total_travelers'] >= 4).astype(int)
    df['is_couple'] = (df['total_travelers'] == 2).astype(int)
    df['is_small_group'] = (df['total_travelers'] == 3).astype(int)
    df['is_very_large_group'] = (df['total_travelers'] >= 6).astype(int)
    df['is_huge_group'] = (df['total_travelers'] >= 8).astype(int)
    
    # Gender features
    df['female_ratio'] = df['num_females'] / (df['total_travelers'] + 1e-5)
    df['male_ratio'] = df['num_males'] / (df['total_travelers'] + 1e-5)
    df['gender_balance'] = np.abs(df['num_females'] - df['num_males']) / (df['total_travelers'] + 1e-5)
    df['is_all_female'] = (df['num_females'] == df['total_travelers']).astype(int)
    df['is_all_male'] = (df['num_males'] == df['total_travelers']).astype(int)
    df['is_mixed_gender'] = ((df['num_females'] > 0) & (df['num_males'] > 0)).astype(int)
    df['gender_diversity'] = 1 - np.abs(df['female_ratio'] - 0.5) * 2
    df['female_dominant'] = (df['num_females'] > df['num_males']).astype(int)
    df['male_dominant'] = (df['num_males'] > df['num_females']).astype(int)
    df['gender_equal'] = (df['num_females'] == df['num_males']).astype(int)
    df['mostly_female'] = (df['female_ratio'] > 0.7).astype(int)
    df['mostly_male'] = (df['male_ratio'] > 0.7).astype(int)
    df['gender_balanced'] = ((df['female_ratio'] >= 0.4) & (df['female_ratio'] <= 0.6)).astype(int)
    
    # ==================== STAY PATTERNS ====================
    df['total_nights'] = df['mainland_stay_nights'].fillna(0) + df['island_stay_nights'].fillna(0)
    df['mainland_ratio'] = df['mainland_stay_nights'].fillna(0) / (df['total_nights'] + 1)
    df['island_ratio'] = df['island_stay_nights'].fillna(0) / (df['total_nights'] + 1)
    df['is_mainland_only'] = ((df['mainland_stay_nights'] > 0) & (df['island_stay_nights'].fillna(0) == 0)).astype(int)
    df['is_island_only'] = ((df['island_stay_nights'] > 0) & (df['mainland_stay_nights'].fillna(0) == 0)).astype(int)
    df['is_mixed_stay'] = ((df['mainland_stay_nights'] > 0) & (df['island_stay_nights'] > 0)).astype(int)
    df['no_stay'] = ((df['mainland_stay_nights'].fillna(0) == 0) & (df['island_stay_nights'].fillna(0) == 0)).astype(int)
    
    df['nights_per_person'] = df['total_nights'] / (df['total_travelers'] + 1)
    df['mainland_nights_per_person'] = df['mainland_stay_nights'].fillna(0) / (df['total_travelers'] + 1)
    df['island_nights_per_person'] = df['island_stay_nights'].fillna(0) / (df['total_travelers'] + 1)
    
    df['prefers_island'] = (df['island_stay_nights'].fillna(0) > df['mainland_stay_nights'].fillna(0)).astype(int)
    df['prefers_mainland'] = (df['mainland_stay_nights'].fillna(0) > df['island_stay_nights'].fillna(0)).astype(int)
    df['balanced_stay'] = (df['mainland_stay_nights'].fillna(0) == df['island_stay_nights'].fillna(0)).astype(int)
    
    df['is_short_stay'] = (df['total_nights'] <= 3).astype(int)
    df['is_medium_stay'] = ((df['total_nights'] > 3) & (df['total_nights'] <= 7)).astype(int)
    df['is_long_stay'] = ((df['total_nights'] > 7) & (df['total_nights'] <= 14)).astype(int)
    df['is_very_long_stay'] = (df['total_nights'] > 14).astype(int)
    
    df['mainland_1to3'] = ((df['mainland_stay_nights'] > 0) & (df['mainland_stay_nights'] <= 3)).astype(int)
    df['mainland_4to7'] = ((df['mainland_stay_nights'] > 3) & (df['mainland_stay_nights'] <= 7)).astype(int)
    df['mainland_8plus'] = (df['mainland_stay_nights'] > 7).astype(int)
    df['island_1to3'] = ((df['island_stay_nights'] > 0) & (df['island_stay_nights'] <= 3)).astype(int)
    df['island_4to7'] = ((df['island_stay_nights'] > 3) & (df['island_stay_nights'] <= 7)).astype(int)
    df['island_8plus'] = (df['island_stay_nights'] > 7).astype(int)
    
    # ==================== INCLUSION FEATURES ====================
    inclusion_cols = ['accomodation_included', 'food_included', 'domestic_transport_included',
                     'sightseeing_included', 'guide_included', 'insurance_included']
    
    df['total_inclusions'] = (df[inclusion_cols].fillna("No") == "Yes").sum(axis=1)
    df['has_no_inclusions'] = (df['total_inclusions'] == 0).astype(int)
    df['has_all_inclusions'] = (df['total_inclusions'] == len(inclusion_cols)).astype(int)
    df['inclusion_ratio'] = df['total_inclusions'] / len(inclusion_cols)
    df['has_few_inclusions'] = (df['total_inclusions'] <= 2).astype(int)
    df['has_many_inclusions'] = (df['total_inclusions'] >= 4).astype(int)
    
    for i in range(7):
        df[f'inclusions_eq_{i}'] = (df['total_inclusions'] == i).astype(int)
    
    for col in inclusion_cols:
        df[f'{col}_flag'] = (df[col].fillna("No") == "Yes").astype(int)
    
    df['has_guide'] = (df['guide_included'].fillna("No") == "Yes").astype(int)
    df['has_insurance'] = (df['insurance_included'].fillna("No") == "Yes").astype(int)
    df['premium_package'] = (df['has_guide'] & df['has_insurance']).astype(int)
    df['basic_package'] = (df['accomodation_included_flag'] & df['food_included_flag']).astype(int)
    df['full_service'] = (df['accomodation_included_flag'] & df['food_included_flag'] & 
                          df['domestic_transport_included_flag'] & df['sightseeing_included_flag']).astype(int)
    df['luxury_package'] = (df['full_service'] & df['has_guide'] & df['has_insurance']).astype(int)
    
    df['essential_inclusions'] = (df['accomodation_included_flag'] + df['food_included_flag'] + 
                                   df['domestic_transport_included_flag'])
    df['luxury_inclusions'] = (df['sightseeing_included_flag'] + df['has_guide'] + df['has_insurance'])
    df['only_essentials'] = ((df['essential_inclusions'] > 0) & (df['luxury_inclusions'] == 0)).astype(int)
    df['only_luxury'] = ((df['essential_inclusions'] == 0) & (df['luxury_inclusions'] > 0)).astype(int)
    df['mixed_inclusions'] = ((df['essential_inclusions'] > 0) & (df['luxury_inclusions'] > 0)).astype(int)
    
    df['inclusions_per_person'] = df['total_inclusions'] / (df['total_travelers'] + 1)
    df['inclusions_per_night'] = df['total_inclusions'] / (df['total_nights'] + 1)
    df['essential_per_person'] = df['essential_inclusions'] / (df['total_travelers'] + 1)
    df['luxury_per_person'] = df['luxury_inclusions'] / (df['total_travelers'] + 1)
    
    # ==================== SPECIAL REQUIREMENTS ====================
    df['has_special_req'] = (df['has_special_requirements'].astype(str).str.lower() != "none").astype(int)
    
    # ==================== WEATHER ====================
    if 'arrival_weather' in df.columns:
        df['weather_rainy'] = df['arrival_weather'].str.contains('rain', case=False, na=False).astype(int)
        df['weather_cloudy'] = df['arrival_weather'].str.contains('cloud', case=False, na=False).astype(int)
        df['weather_sunny'] = df['arrival_weather'].str.contains('sun|clear', case=False, na=False).astype(int)
        df['weather_bad'] = (df['weather_rainy'] | df['weather_cloudy']).astype(int)
    else:
        df['weather_rainy'] = 0
        df['weather_cloudy'] = 0
        df['weather_sunny'] = 0
        df['weather_bad'] = 0
    
    # ==================== TRIP DAYS ====================
    trip_bins = {"1-3": 1, "4-6": 2, "7-14": 3, "15-30": 4, "30+": 5}
    df['trip_days_numeric'] = df['total_trip_days'].map(trip_bins).fillna(1).astype(int)
    
    for i in range(1, 6):
        df[f'trip_days_cat_{i}'] = (df['trip_days_numeric'] == i).astype(int)
    
    df['is_short_trip'] = (df['trip_days_numeric'] <= 2).astype(int)
    df['is_medium_trip'] = (df['trip_days_numeric'] == 3).astype(int)
    df['is_long_trip'] = (df['trip_days_numeric'] >= 4).astype(int)
    df['is_very_long_trip'] = (df['trip_days_numeric'] == 5).astype(int)
    
    df['trip_intensity'] = df['total_nights'] / (df['trip_days_numeric'] + 0.1)
    df['is_intense_trip'] = (df['trip_intensity'] > 0.8).astype(int)
    df['is_relaxed_trip'] = (df['trip_intensity'] < 0.4).astype(int)
    df['is_moderate_trip'] = ((df['trip_intensity'] >= 0.4) & (df['trip_intensity'] <= 0.8)).astype(int)
    
    # ==================== KEY INTERACTIONS ====================
    df['travelers_x_nights'] = df['total_travelers'] * df['total_nights']
    df['travelers_x_inclusions'] = df['total_travelers'] * df['total_inclusions']
    df['nights_x_inclusions'] = df['total_nights'] * df['total_inclusions']
    df['travelers_x_trip_days'] = df['total_travelers'] * df['trip_days_numeric']
    df['nights_x_trip_days'] = df['total_nights'] * df['trip_days_numeric']
    df['inclusions_x_trip_days'] = df['total_inclusions'] * df['trip_days_numeric']
    df['travelers_x_nights_x_inclusions'] = df['total_travelers'] * df['total_nights'] * df['total_inclusions']
    
    df['group_x_weather'] = df['is_group'] * df['weather_bad']
    df['group_x_premium'] = df['is_group'] * df['premium_package']
    df['couple_x_inclusions'] = df['is_couple'] * df['total_inclusions']
    df['large_group_x_inclusions'] = df['is_large_group'] * df['total_inclusions']
    df['female_ratio_x_inclusions'] = df['female_ratio'] * df['total_inclusions']
    df['island_only_x_inclusions'] = df['is_island_only'] * df['total_inclusions']
    df['mixed_stay_x_inclusions'] = df['is_mixed_stay'] * df['total_inclusions']
    
    df['total_person_nights'] = df['total_travelers'] * df['total_nights']
    df['total_person_days'] = df['total_travelers'] * df['trip_days_numeric']
    df['spend_capacity'] = df['total_travelers'] * df['trip_days_numeric'] * (df['total_inclusions'] + 1)
    df['luxury_score'] = df['luxury_inclusions'] * df['total_travelers'] * df['trip_days_numeric']
    df['budget_score'] = df['total_travelers'] * df['total_nights'] / (df['total_inclusions'] + 1)
    
    df['traveler_density'] = df['total_travelers'] / (df['total_nights'] + 1)
    df['inclusion_density'] = df['total_inclusions'] / (df['total_travelers'] + 1)
    df['night_efficiency'] = df['total_nights'] / (df['trip_days_numeric'] + 1)
    
    # NEW: Cross-categorical features
    df['travelers_x_mainland_bin'] = df['total_travelers'] * (df['mainland_stay_nights'].fillna(0) > 0).astype(int)
    df['travelers_x_island_bin'] = df['total_travelers'] * (df['island_stay_nights'].fillna(0) > 0).astype(int)
    df['inclusions_x_weather'] = df['total_inclusions'] * df['weather_bad']
    df['nights_per_person_x_inclusions'] = df['nights_per_person'] * df['total_inclusions']
    
    # ==================== CATEGORICAL ENCODING ====================
    if train_power is not None and 'onehot_countries' in train_power:
        top_countries = train_power['onehot_countries']
    else:
        top_countries = df['country'].value_counts().nlargest(35).index.tolist()
    
    for c in top_countries:
        df[f'country_{c}'] = (df['country'] == c).astype(int)
    
    if train_power is not None and 'onehot_activities' in train_power:
        top_activities = train_power['onehot_activities']
    else:
        top_activities = df['main_activity'].value_counts().nlargest(15).index.tolist()
    
    for a in top_activities:
        df[f'activity_{a}'] = (df['main_activity'] == a).astype(int)
    
    df['country_activity'] = df['country'].astype(str) + "_" + df['main_activity'].astype(str)
    df['country_trip_days'] = df['country'].astype(str) + "_" + df['trip_days_numeric'].astype(str)
    df['activity_trip_days'] = df['main_activity'].astype(str) + "_" + df['trip_days_numeric'].astype(str)
    df['country_inclusions'] = df['country'].astype(str) + "_" + df['total_inclusions'].astype(str)
    
    # ==================== TARGET ENCODING ====================
    if train_power is not None:
        if 'country_spend_mean' in train_power:
            df['country_spend_mean'] = df['country'].map(train_power['country_spend_mean']).fillna(train_power['spend_global_mean'])
            df['country_spend_std'] = df['country'].map(train_power['country_spend_std']).fillna(1.0)
            df['country_spend_median'] = df['country'].map(train_power['country_spend_median']).fillna(train_power['spend_global_mean'])
        
        if 'activity_spend_mean' in train_power:
            df['activity_spend_mean'] = df['main_activity'].map(train_power['activity_spend_mean']).fillna(train_power['spend_global_mean'])
            df['activity_spend_std'] = df['main_activity'].map(train_power['activity_spend_std']).fillna(1.0)
            df['activity_spend_median'] = df['main_activity'].map(train_power['activity_spend_median']).fillna(train_power['spend_global_mean'])
        
        if 'country_activity_spend_mean' in train_power:
            df['country_activity_spend_mean'] = df['country_activity'].map(train_power['country_activity_spend_mean']).fillna(train_power['spend_global_mean'])
        
        if 'country_tripdays_spend_mean' in train_power:
            df['country_tripdays_spend_mean'] = df['country_trip_days'].map(train_power['country_tripdays_spend_mean']).fillna(train_power['spend_global_mean'])
        
        if 'activity_tripdays_spend_mean' in train_power:
            df['activity_tripdays_spend_mean'] = df['activity_trip_days'].map(train_power['activity_tripdays_spend_mean']).fillna(train_power['spend_global_mean'])
        
        if 'country_inclusions_spend_mean' in train_power:
            df['country_inclusions_spend_mean'] = df['country_inclusions'].map(train_power['country_inclusions_spend_mean']).fillna(train_power['spend_global_mean'])
        
        if 'weather_spend_mean' in train_power and 'arrival_weather' in df.columns:
            df['weather_spend_mean'] = df['arrival_weather'].map(train_power['weather_spend_mean']).fillna(train_power['spend_global_mean'])
    
    # ==================== FREQUENCY ENCODING ====================
    if train_power is not None:
        if 'country_freq' in train_power:
            df['country_frequency'] = df['country'].map(train_power['country_freq']).fillna(0)
        if 'activity_freq' in train_power:
            df['activity_frequency'] = df['main_activity'].map(train_power['activity_freq']).fillna(0)
    
    # ==================== RATIOS ====================
    df['female_to_male_ratio'] = df['num_females'] / (df['num_males'] + 1)
    df['male_to_female_ratio'] = df['num_males'] / (df['num_females'] + 1)
    df['mainland_to_island_ratio'] = df['mainland_stay_nights'].fillna(0) / (df['island_stay_nights'].fillna(0) + 1)
    df['island_to_mainland_ratio'] = df['island_stay_nights'].fillna(0) / (df['mainland_stay_nights'].fillna(0) + 1)
    df['essential_to_luxury_ratio'] = df['essential_inclusions'] / (df['luxury_inclusions'] + 1)
    df['luxury_to_essential_ratio'] = df['luxury_inclusions'] / (df['essential_inclusions'] + 1)
    df['nights_to_travelers_ratio'] = df['total_nights'] / (df['total_travelers'] + 1)
    df['travelers_to_nights_ratio'] = df['total_travelers'] / (df['total_nights'] + 1)
    
    # ==================== BINNING ====================
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(-1)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(-1)
    
    bins_nights = [-1, 0, 1, 2, 3, 4, 5, 7, 10, 15, 100]
    df['mainland_bin'] = pd.cut(df['mainland_stay_nights'], bins=bins_nights, labels=False).fillna(0).astype(int)
    df['island_bin'] = pd.cut(df['island_stay_nights'], bins=bins_nights, labels=False).fillna(0).astype(int)
    df['total_nights_bin'] = pd.cut(df['total_nights'], bins=bins_nights, labels=False).fillna(0).astype(int)
    
    df['travelers_bin'] = pd.cut(df['total_travelers'], bins=[0, 1, 2, 3, 4, 5, 7, 100], labels=False).fillna(0).astype(int)
    df['inclusion_bin'] = pd.cut(df['total_inclusions'], bins=[-1, 0, 1, 2, 3, 4, 5, 10], labels=False).fillna(0).astype(int)
    
    return df

def compute_power_stats(train):
    """Compute statistics from training data"""
    stats = {}
    
    stats['spend_global_mean'] = train['spend_category'].mean()
    stats['spend_global_median'] = train['spend_category'].median()
    
    country_stats = train.groupby('country')['spend_category'].agg(['mean', 'median', 'std', 'count'])
    stats['country_spend_mean'] = country_stats['mean'].to_dict()
    stats['country_spend_median'] = country_stats['median'].to_dict()
    stats['country_spend_std'] = country_stats['std'].fillna(1.0).to_dict()
    stats['country_freq'] = (country_stats['count'] / len(train)).to_dict()
    
    activity_stats = train.groupby('main_activity')['spend_category'].agg(['mean', 'median', 'std', 'count'])
    stats['activity_spend_mean'] = activity_stats['mean'].to_dict()
    stats['activity_spend_median'] = activity_stats['median'].to_dict()
    stats['activity_spend_std'] = activity_stats['std'].fillna(1.0).to_dict()
    stats['activity_freq'] = (activity_stats['count'] / len(train)).to_dict()
    
    train['country_activity'] = train['country'].astype(str) + "_" + train['main_activity'].astype(str)
    stats['country_activity_spend_mean'] = train.groupby('country_activity')['spend_category'].mean().to_dict()
    
    trip_bins = {"1-3": 1, "4-6": 2, "7-14": 3, "15-30": 4, "30+": 5}
    train['trip_days_numeric'] = train['total_trip_days'].map(trip_bins).fillna(1).astype(int)
    train['country_trip_days'] = train['country'].astype(str) + "_" + train['trip_days_numeric'].astype(str)
    train['activity_trip_days'] = train['main_activity'].astype(str) + "_" + train['trip_days_numeric'].astype(str)
    
    # NEW: country-inclusions target encoding
    train['total_inclusions'] = 0
    inclusion_cols = ['accomodation_included', 'food_included', 'domestic_transport_included',
                     'sightseeing_included', 'guide_included', 'insurance_included']
    train['total_inclusions'] = (train[inclusion_cols].fillna("No") == "Yes").sum(axis=1)
    train['country_inclusions'] = train['country'].astype(str) + "_" + train['total_inclusions'].astype(str)
    
    stats['country_tripdays_spend_mean'] = train.groupby('country_trip_days')['spend_category'].mean().to_dict()
    stats['activity_tripdays_spend_mean'] = train.groupby('activity_trip_days')['spend_category'].mean().to_dict()
    stats['country_inclusions_spend_mean'] = train.groupby('country_inclusions')['spend_category'].mean().to_dict()
    
    if 'arrival_weather' in train.columns:
        stats['weather_spend_mean'] = train.groupby('arrival_weather')['spend_category'].mean().to_dict()
    
    stats['onehot_countries'] = train['country'].value_counts().nlargest(35).index.tolist()
    stats['onehot_activities'] = train['main_activity'].value_counts().nlargest(15).index.tolist()
    
    return stats

def load_and_preprocess(train_path, test_path):
    """Main preprocessing pipeline"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train_ids = train['trip_id']
    test_ids = test['trip_id']
    y_train = train['spend_category'] if 'spend_category' in train else None
    
    if y_train is not None and y_train.isnull().any():
        print("Warning: Missing labels detected. Filling with mode.")
        y_train = y_train.fillna(y_train.mode()[0])
    
    train_power = compute_power_stats(train)
    
    X_train = train.drop(columns=['trip_id'] + (['spend_category'] if 'spend_category' in train else []))
    X_test = test.drop(columns=['trip_id'])
    
    X_train = add_advanced_features(X_train, train_power)
    X_test = add_advanced_features(X_test, train_power)
    
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    for col in train_cols - test_cols:
        X_test[col] = 0
    for col in test_cols - train_cols:
        X_train[col] = 0
    
    X_test = X_test[X_train.columns]
    
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, test_ids, train_ids