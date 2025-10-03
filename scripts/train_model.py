# import json
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import pickle
# import sys

# # Add the parent directory to the path so we can import the feature_extractor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_commits.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

# def train_risk_model():
#     """
#     Loads the dataset, processes it using Cyclomatic Complexity, and trains the model.
#     """
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
        
#     print("--- Loading and Preparing Dataset for Training ---")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{INPUT_FILE}' not found.")
#         print("Please run 'scripts/mine_data.py' first.")
#         return

#     data = []
#     for pair in dataset:
#         data.append({'code': pair['buggy_code'], 'label': 1})
#         data.append({'code': pair['fixed_code'], 'label': 0})
    
#     df = pd.DataFrame(data)

#     print("Extracting features with Cyclomatic Complexity... (This will take time but is worth it)")
#     features = df['code'].apply(extract_features)
    
#     # --- UPDATED: New column names for our better features ---
#     feature_columns = [
#         'lines_of_code',
#         'method_count',
#         'cyclomatic_complexity',
#         'try_statements',
#         'variable_declarators'
#     ]
#     df_features = pd.DataFrame(features.tolist(), columns=feature_columns)
    
#     X = df_features
#     y = df['label']

#     print(f"Processed {len(y)} total code samples with {len(feature_columns)} features each.")

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     print("\n--- Training High-Accuracy Risk Model ---")
#     model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#     model.fit(X_train, y_train)
#     print("Model training complete.")

#     y_pred = model.predict(X_test)
#     print("\n--- FINAL MODEL EVALUATION REPORT ---")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))

#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(model, f)
        
#     print(f"\nNew, high-accuracy model saved to {OUTPUT_MODEL_FILE}")

# if __name__ == '__main__':
#     train_risk_model()



import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import feature extractor
try:
    from code_reviewer.feature_extractor import extract_features
    print("✓ Successfully imported feature_extractor")
except ImportError as e:
    print(f"✗ Error importing feature_extractor: {e}")
    print("Make sure you've updated code_reviewer/feature_extractor.py with the new code")
    sys.exit(1)

# Configuration
INPUT_FILE = 'data/bugfix_commits.json'
MODEL_DIR = 'models'
OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'feature_scaler.pkl')

def remove_outliers(X, y, contamination=0.1):
    """Remove outliers using IQR method for better model training."""
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep only rows without outliers in any feature
    mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
    
    print(f"Removed {(~mask).sum()} outlier samples ({(~mask).sum()/len(X)*100:.2f}%)")
    
    return X[mask], y[mask]

def train_risk_model():
    """
    Enhanced training pipeline with feature engineering, scaling, and ensemble methods.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("\n" + "="*70)
    print("ENHANCED BUG PREDICTION MODEL TRAINING")
    print("="*70)
    
    # Load dataset
    print("\n[1/7] Loading dataset...")
    try:
        with open(INPUT_FILE, 'r') as f:
            dataset = json.load(f)
        print(f"✓ Loaded {len(dataset)} bugfix pairs")
    except FileNotFoundError:
        print(f"✗ Error: Dataset file '{INPUT_FILE}' not found.")
        print("Please run 'scripts/mine_data.py' first.")
        return

    # Prepare data
    print("\n[2/7] Preparing dataset...")
    data = []
    for pair in dataset:
        data.append({'code': pair['buggy_code'], 'label': 1})
        data.append({'code': pair['fixed_code'], 'label': 0})
    
    df = pd.DataFrame(data)
    print(f"✓ Total samples: {len(df)} (Buggy: {sum(df['label']==1)}, Fixed: {sum(df['label']==0)})")

    # Extract features
    print("\n[3/7] Extracting 20 comprehensive features...")
    print("This may take a few minutes...")
    features = df['code'].apply(extract_features)
    
    feature_columns = [
        'lines_of_code', 'non_empty_lines', 'method_count', 'complexity',
        'complexity_per_method', 'loc_per_method', 'nesting_depth', 'decision_density',
        'if_count', 'loop_count', 'exception_ratio', 'invocation_density',
        'variable_ratio', 'null_check_ratio', 'array_ratio', 'cast_ratio',
        'assignment_density', 'return_ratio', 'halstead_volume', 'vocabulary_size'
    ]
    
    X = pd.DataFrame(features.tolist(), columns=feature_columns)
    y = df['label']
    
    print(f"✓ Extracted {len(feature_columns)} features from {len(y)} samples")
    
    # Remove samples with all zeros (parsing failures)
    valid_mask = (X.sum(axis=1) > 0)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"✓ Removed {(~valid_mask).sum()} invalid samples, {len(X)} samples remain")
    
    # Check if we have enough data
    if len(X) < 100:
        print("\n⚠ WARNING: Very small dataset! You need at least 100 valid samples.")
        print(f"Current valid samples: {len(X)}")
        print("Consider collecting more bugfix data to improve accuracy.")
    
    # Remove outliers
    print("\n[4/7] Removing outliers...")
    X, y = remove_outliers(X, y)
    
    # Feature scaling
    print("\n[5/7] Scaling features...")
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Save scaler for later use
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Feature scaler saved to {SCALER_FILE}")
    
    # Feature selection
    print("\n[6/7] Selecting most important features...")
    selector = SelectKBest(f_classif, k=min(15, len(feature_columns)))  # Keep top 15 features or less if we have fewer
    X_selected = selector.fit_transform(X_scaled, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"✓ Selected {len(selected_features)} most predictive features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i}. {feat}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train ensemble model
    print("\n[7/7] Training ensemble model...")
    print("Building Random Forest, Gradient Boosting, and Voting Classifier...")
    
    # Individual models with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Voting classifier (ensemble)
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Train ensemble
    ensemble_model.fit(X_train, y_train)
    print("✓ Model training complete!")
    
    # Cross-validation
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    cv = StratifiedKFold(n_splits=min(5, len(X_train)//10), shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble_model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Test set evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    y_pred = ensemble_model.predict(X_test)
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nAccuracy: {test_accuracy:.4f}")
    print(f"ROC-AUC Score: {test_roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              0        1")
    print(f"Actual  0   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"        1   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Feature importance (from Random Forest)
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70)
    
    rf_final = rf_model.fit(X_train, y_train)
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(min(10, len(selected_features))).iterrows():
        print(f"{row['feature']:25s} {row['importance']:.4f}")
    
    # Save model
    model_package = {
        'model': ensemble_model,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'feature_columns': feature_columns
    }
    
    with open(OUTPUT_MODEL_FILE, 'wb') as f:
        pickle.dump(model_package, f)
    
    print("\n" + "="*70)
    print(f"✓ Model package saved to {OUTPUT_MODEL_FILE}")
    print("="*70)
    
    # Final recommendations based on results
    if test_accuracy < 0.70:
        print("\n⚠ ACCURACY BELOW TARGET (70%)")
        print("\nRECOMMENDATIONS:")
        print("1. Collect more training data (current: {} samples)".format(len(X)))
        print("2. Ensure buggy/fixed code pairs are substantially different")
        print("3. Verify your dataset quality - check some examples manually")
        print("4. Try collecting data from multiple repositories")
    elif test_accuracy >= 0.70 and test_accuracy < 0.80:
        print("\n✓ GOOD ACCURACY ACHIEVED (70-80%)")
        print("\nTo reach 80%+:")
        print("1. Add more diverse training examples")
        print("2. Consider domain-specific features")
        print("3. Experiment with deep learning models")
    else:
        print("\n✓✓ EXCELLENT ACCURACY ACHIEVED (80%+)")
        print("Your model is performing very well!")

if __name__ == '__main__':
    print("\nStarting training process...")
    print("Make sure you have updated feature_extractor.py with the new 20-feature version!\n")
    
    try:
        train_risk_model()
    except Exception as e:
        print(f"\n✗ TRAINING FAILED")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. feature_extractor.py has been updated with new code")
        print("2. data/bugfix_commits.json exists and is valid")
        print("3. All required packages are installed")