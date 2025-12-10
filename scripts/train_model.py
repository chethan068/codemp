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



# import json
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
# from sklearn.feature_selection import SelectKBest, f_classif
# import pickle
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# # Add the parent directory to the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Import feature extractor
# try:
#     from code_reviewer.feature_extractor import extract_features
#     print("✓ Successfully imported feature_extractor")
# except ImportError as e:
#     print(f"✗ Error importing feature_extractor: {e}")
#     print("Make sure you've updated code_reviewer/feature_extractor.py with the new code")
#     sys.exit(1)

# # Configuration
# INPUT_FILE = 'data/bugfix_commits.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')
# SCALER_FILE = os.path.join(MODEL_DIR, 'feature_scaler.pkl')

# def remove_outliers(X, y, contamination=0.1):
#     """Remove outliers using IQR method for better model training."""
#     Q1 = X.quantile(0.25)
#     Q3 = X.quantile(0.75)
#     IQR = Q3 - Q1
    
#     # Define outlier bounds
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     # Keep only rows without outliers in any feature
#     mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
    
#     print(f"Removed {(~mask).sum()} outlier samples ({(~mask).sum()/len(X)*100:.2f}%)")
    
#     return X[mask], y[mask]

# def train_risk_model():
#     """
#     Enhanced training pipeline with feature engineering, scaling, and ensemble methods.
#     """
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     print("\n" + "="*70)
#     print("ENHANCED BUG PREDICTION MODEL TRAINING")
#     print("="*70)
    
#     # Load dataset
#     print("\n[1/7] Loading dataset...")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#         print(f"✓ Loaded {len(dataset)} bugfix pairs")
#     except FileNotFoundError:
#         print(f"✗ Error: Dataset file '{INPUT_FILE}' not found.")
#         print("Please run 'scripts/mine_data.py' first.")
#         return

#     # Prepare data
#     print("\n[2/7] Preparing dataset...")
#     data = []
#     for pair in dataset:
#         data.append({'code': pair['buggy_code'], 'label': 1})
#         data.append({'code': pair['fixed_code'], 'label': 0})
    
#     df = pd.DataFrame(data)
#     print(f"✓ Total samples: {len(df)} (Buggy: {sum(df['label']==1)}, Fixed: {sum(df['label']==0)})")

#     # Extract features
#     print("\n[3/7] Extracting 20 comprehensive features...")
#     print("This may take a few minutes...")
#     features = df['code'].apply(extract_features)
    
#     feature_columns = [
#         'lines_of_code', 'non_empty_lines', 'method_count', 'complexity',
#         'complexity_per_method', 'loc_per_method', 'nesting_depth', 'decision_density',
#         'if_count', 'loop_count', 'exception_ratio', 'invocation_density',
#         'variable_ratio', 'null_check_ratio', 'array_ratio', 'cast_ratio',
#         'assignment_density', 'return_ratio', 'halstead_volume', 'vocabulary_size'
#     ]
    
#     X = pd.DataFrame(features.tolist(), columns=feature_columns)
#     y = df['label']
    
#     print(f"✓ Extracted {len(feature_columns)} features from {len(y)} samples")
    
#     # Remove samples with all zeros (parsing failures)
#     valid_mask = (X.sum(axis=1) > 0)
#     X = X[valid_mask]
#     y = y[valid_mask]
#     print(f"✓ Removed {(~valid_mask).sum()} invalid samples, {len(X)} samples remain")
    
#     # Check if we have enough data
#     if len(X) < 100:
#         print("\n⚠ WARNING: Very small dataset! You need at least 100 valid samples.")
#         print(f"Current valid samples: {len(X)}")
#         print("Consider collecting more bugfix data to improve accuracy.")
    
#     # Remove outliers
#     print("\n[4/7] Removing outliers...")
#     X, y = remove_outliers(X, y)
    
#     # Feature scaling
#     print("\n[5/7] Scaling features...")
#     scaler = RobustScaler()  # More robust to outliers than StandardScaler
#     X_scaled = pd.DataFrame(
#         scaler.fit_transform(X),
#         columns=X.columns,
#         index=X.index
#     )
    
#     # Save scaler for later use
#     with open(SCALER_FILE, 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"✓ Feature scaler saved to {SCALER_FILE}")
    
#     # Feature selection
#     print("\n[6/7] Selecting most important features...")
#     selector = SelectKBest(f_classif, k=min(15, len(feature_columns)))  # Keep top 15 features or less if we have fewer
#     X_selected = selector.fit_transform(X_scaled, y)
    
#     selected_features = X.columns[selector.get_support()].tolist()
#     print(f"✓ Selected {len(selected_features)} most predictive features:")
#     for i, feat in enumerate(selected_features, 1):
#         print(f"   {i}. {feat}")
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_selected, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     print(f"\n✓ Training set: {len(X_train)} samples")
#     print(f"✓ Test set: {len(X_test)} samples")
    
#     # Train ensemble model
#     print("\n[7/7] Training ensemble model...")
#     print("Building Random Forest, Gradient Boosting, and Voting Classifier...")
    
#     # Individual models with optimized parameters
#     rf_model = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=15,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         max_features='sqrt',
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     )
    
#     gb_model = GradientBoostingClassifier(
#         n_estimators=200,
#         learning_rate=0.1,
#         max_depth=5,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         subsample=0.8,
#         random_state=42
#     )
    
#     # Voting classifier (ensemble)
#     ensemble_model = VotingClassifier(
#         estimators=[
#             ('rf', rf_model),
#             ('gb', gb_model)
#         ],
#         voting='soft',
#         n_jobs=-1
#     )
    
#     # Train ensemble
#     ensemble_model.fit(X_train, y_train)
#     print("✓ Model training complete!")
    
#     # Cross-validation
#     print("\n" + "="*70)
#     print("CROSS-VALIDATION RESULTS")
#     print("="*70)
#     cv = StratifiedKFold(n_splits=min(5, len(X_train)//10), shuffle=True, random_state=42)
#     cv_scores = cross_val_score(ensemble_model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
#     print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
#     print(f"Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    
#     # Test set evaluation
#     print("\n" + "="*70)
#     print("TEST SET EVALUATION")
#     print("="*70)
    
#     y_pred = ensemble_model.predict(X_test)
#     y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
#     test_accuracy = accuracy_score(y_test, y_pred)
#     test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
#     print(f"\nAccuracy: {test_accuracy:.4f}")
#     print(f"ROC-AUC Score: {test_roc_auc:.4f}")
    
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
    
#     print("\nConfusion Matrix:")
#     cm = confusion_matrix(y_test, y_pred)
#     print(f"                Predicted")
#     print(f"              0        1")
#     print(f"Actual  0   {cm[0,0]:4d}    {cm[0,1]:4d}")
#     print(f"        1   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
#     # Feature importance (from Random Forest)
#     print("\n" + "="*70)
#     print("TOP 10 MOST IMPORTANT FEATURES")
#     print("="*70)
    
#     rf_final = rf_model.fit(X_train, y_train)
#     feature_importance = pd.DataFrame({
#         'feature': selected_features,
#         'importance': rf_final.feature_importances_
#     }).sort_values('importance', ascending=False)
    
#     for idx, row in feature_importance.head(min(10, len(selected_features))).iterrows():
#         print(f"{row['feature']:25s} {row['importance']:.4f}")
    
#     # Save model
#     model_package = {
#         'model': ensemble_model,
#         'scaler': scaler,
#         'selector': selector,
#         'selected_features': selected_features,
#         'feature_columns': feature_columns
#     }
    
#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(model_package, f)
    
#     print("\n" + "="*70)
#     print(f"✓ Model package saved to {OUTPUT_MODEL_FILE}")
#     print("="*70)
    
#     # Final recommendations based on results
#     if test_accuracy < 0.70:
#         print("\n⚠ ACCURACY BELOW TARGET (70%)")
#         print("\nRECOMMENDATIONS:")
#         print("1. Collect more training data (current: {} samples)".format(len(X)))
#         print("2. Ensure buggy/fixed code pairs are substantially different")
#         print("3. Verify your dataset quality - check some examples manually")
#         print("4. Try collecting data from multiple repositories")
#     elif test_accuracy >= 0.70 and test_accuracy < 0.80:
#         print("\n✓ GOOD ACCURACY ACHIEVED (70-80%)")
#         print("\nTo reach 80%+:")
#         print("1. Add more diverse training examples")
#         print("2. Consider domain-specific features")
#         print("3. Experiment with deep learning models")
#     else:
#         print("\n✓✓ EXCELLENT ACCURACY ACHIEVED (80%+)")
#         print("Your model is performing very well!")

# if __name__ == '__main__':
#     print("\nStarting training process...")
#     print("Make sure you have updated feature_extractor.py with the new 20-feature version!\n")
    
#     try:
#         train_risk_model()
#     except Exception as e:
#         print(f"\n✗ TRAINING FAILED")
#         print(f"Error: {str(e)}")
#         print("\nFull traceback:")
#         import traceback
#         traceback.print_exc()
#         print("\nPlease check:")
#         print("1. feature_extractor.py has been updated with new code")
#         print("2. data/bugfix_commits.json exists and is valid")
#         print("3. All required packages are installed")

# import json
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# import lightgbm as lgb
# from sklearn.metrics import classification_report, accuracy_score
# import pickle
# import sys

# # Add the parent directory to the path so we can import the feature_extractor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features_from_snippet

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_changes.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

# def train_risk_model():
#     """
#     Trains a high-accuracy model using change-level data, text-based features,
#     and a tuned LightGBM classifier.
#     """
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
        
#     print("--- Loading Change-Level Dataset ---")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{INPUT_FILE}' not found.")
#         print("Please run the new 'scripts/mine_data.py' first.")
#         return

#     data = []
#     for change in dataset:
#         data.append({'snippet': change['buggy_change'], 'label': 1})
#         data.append({'snippet': change['fixed_change'], 'label': 0})
    
#     df = pd.DataFrame(data)

#     print("Extracting text-based features from code snippets...")
#     features = df['snippet'].apply(extract_features_from_snippet)
    
#     feature_columns = ['num_lines', 'num_chars', 'keyword_count']
#     df_features = pd.DataFrame(features.tolist(), columns=feature_columns)

#     X = df_features
#     y = df['label']

#     print(f"Processed {len(y)} snippets with {len(feature_columns)} features each.")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42, stratify=y
#     )

#     # --- Hyperparameter Tuning with LightGBM ---
#     print("\n--- Searching for Best Parameters with LightGBM ---")
    
#     param_dist = {
#         'n_estimators': [100, 200, 300, 500],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'num_leaves': [31, 50, 70],
#         'max_depth': [-1, 10, 20],
#     }

#     # Use RandomizedSearchCV for a faster search
#     random_search = RandomizedSearchCV(
#         estimator=lgb.LGBMClassifier(random_state=42, class_weight='balanced'),
#         param_distributions=param_dist,
#         n_iter=20,  # Number of parameter settings that are sampled
#         cv=3,       # 3-fold cross-validation
#         verbose=1,
#         random_state=42,
#         n_jobs=-1,  # Use all available cores
#         scoring='accuracy'
#     )

#     random_search.fit(X_train, y_train)
    
#     print(f"\nBest parameters found: {random_search.best_params_}")
    
#     best_model = random_search.best_estimator_

#     # --- Final Evaluation on the Test Set ---
#     y_pred = best_model.predict(X_test)
    
#     print("\n--- FINAL MODEL EVALUATION REPORT ---")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
#     print(f"Final Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")

#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(best_model, f)
        
#     print(f"\nNew, optimized model saved to {OUTPUT_MODEL_FILE}")


# if __name__ == '__main__':
#     train_risk_model()



# import json
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# import lightgbm as lgb
# from sklearn.metrics import classification_report, accuracy_score
# # Import the new scaler
# from sklearn.preprocessing import StandardScaler
# import pickle
# import sys

# # Add the parent directory to the path so we can import the feature_extractor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features_from_snippet

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_changes.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

# def train_risk_model():
#     """
#     Trains a high-accuracy model using change-level data, text-based features,
#     feature scaling, and a comprehensively tuned LightGBM classifier.
#     """
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
        
#     print("--- Loading Change-Level Dataset ---")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{INPUT_FILE}' not found.")
#         print("Please run the new 'scripts/mine_data.py' first.")
#         return

#     data = []
#     for change in dataset:
#         data.append({'snippet': change['buggy_change'], 'label': 1})
#         data.append({'snippet': change['fixed_change'], 'label': 0})
    
#     df = pd.DataFrame(data)

#     print("Extracting text-based features from code snippets...")
#     features = df['snippet'].apply(extract_features_from_snippet)
    
#     feature_columns = ['num_lines', 'num_chars', 'keyword_count']
#     df_features = pd.DataFrame(features.tolist(), columns=feature_columns)

#     X = df_features
#     y = df['label']

#     print(f"Processed {len(y)} snippets with {len(feature_columns)} features each.")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42, stratify=y
#     )

#     # --- NEW: Feature Scaling Step ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     print("Applied StandardScaler to normalize feature data.")

#     # --- Hyperparameter Tuning with LightGBM ---
#     print("\n--- Searching for Best Parameters with LightGBM (More Iterations) ---")
    
#     # Expanded parameter grid for a more thorough search
#     param_dist = {
#         'n_estimators': [100, 200, 300, 500, 700],
#         'learning_rate': [0.01, 0.05, 0.1, 0.2],
#         'num_leaves': [31, 50, 70, 90, 120],
#         'max_depth': [-1, 10, 20, 30],
#         'reg_alpha': [0, 0.1, 0.5],
#         'reg_lambda': [0, 0.1, 0.5]
#     }

#     # Increased n_iter for a more exhaustive search
#     random_search = RandomizedSearchCV(
#         estimator=lgb.LGBMClassifier(random_state=42, class_weight='balanced'),
#         param_distributions=param_dist,
#         n_iter=50,  # Increased iterations for a better search
#         cv=3,       
#         verbose=1,
#         random_state=42,
#         n_jobs=-1,  
#         scoring='accuracy'
#     )

#     # Use the scaled data for training
#     random_search.fit(X_train_scaled, y_train)
    
#     print(f"\nBest parameters found: {random_search.best_params_}")
    
#     best_model = random_search.best_estimator_

#     # --- Final Evaluation on the Test Set ---
#     # Use the scaled data for prediction
#     y_pred = best_model.predict(X_test_scaled)
    
#     print("\n--- FINAL MODEL EVALUATION REPORT ---")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
#     print(f"Final Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")

#     # --- NEW: Save both the model and the scaler ---
#     # The scaler must be saved to process live data in the same way
#     model_payload = {
#         'model': best_model,
#         'scaler': scaler
#     }
#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(model_payload, f)
        
#     print(f"\nNew, optimized model and scaler saved to {OUTPUT_MODEL_FILE}")


# if __name__ == '__main__':
#     train_risk_model()





#0.63


# import json
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# import lightgbm as lgb
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import StandardScaler
# import pickle
# import sys

# # Add the parent directory to the path so we can import the feature_extractor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features_from_snippet

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_changes.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

# def train_risk_model():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     print("--- Loading Change-Level Dataset ---")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{INPUT_FILE}' not found.")
#         return

#     data = []
#     for change in dataset:
#         # Filter out trivial changes
#         if len(change['buggy_change'].splitlines()) < 3 or len(change['fixed_change'].splitlines()) < 3:
#             continue
#         data.append({'snippet': change['buggy_change'], 'label': 1})
#         data.append({'snippet': change['fixed_change'], 'label': 0})

#     df = pd.DataFrame(data)

#     print("Extracting enhanced features from code snippets...")
#     features = df['snippet'].apply(extract_features_from_snippet)

#     feature_columns = [
#         'num_lines', 'num_chars', 'keyword_count', 'control_count',
#         'exception_count', 'method_calls', 'comment_ratio',
#         'suspicious_count', 'line_stddev'
#     ]
#     df_features = pd.DataFrame(features.tolist(), columns=feature_columns)

#     X = df_features
#     y = df['label']

#     print(f"Processed {len(y)} snippets with {len(feature_columns)} features each.")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42, stratify=y
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     print("Applied StandardScaler to normalize feature data.")

#     print("\n--- Searching for Best Parameters with LightGBM ---")
#     param_dist = {
#         'n_estimators': [100, 200, 300, 500],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'num_leaves': [31, 50, 70],
#         'max_depth': [-1, 10, 20],
#         'reg_alpha': [0, 0.1],
#         'reg_lambda': [0, 0.1]
#     }

#     random_search = RandomizedSearchCV(
#         estimator=lgb.LGBMClassifier(random_state=42, class_weight='balanced'),
#         param_distributions=param_dist,
#         n_iter=30,
#         cv=3,
#         verbose=1,
#         random_state=42,
#         n_jobs=-1,
#         scoring='accuracy'
#     )

#     random_search.fit(X_train_scaled, y_train)

#     print(f"\nBest parameters found: {random_search.best_params_}")
#     best_model = random_search.best_estimator_

#     y_pred = best_model.predict(X_test_scaled)

#     print("\n--- FINAL MODEL EVALUATION REPORT ---")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
#     print(f"Final Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")

#     model_payload = {
#         'model': best_model,
#         'scaler': scaler
#     }
#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(model_payload, f)

#     print(f"\nOptimized model and scaler saved to {OUTPUT_MODEL_FILE}")

# if __name__ == '__main__':
#     train_risk_model()


#late debug high acc


# import json
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# import pickle
# import sys
# import torch
# from transformers import AutoTokenizer, AutoModel

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features_from_snippet

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_changes.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

# # Load CodeBERT
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("microsoft/codebert-base")

# def get_codebert_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# def train_risk_model():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     print("--- Loading Change-Level Dataset ---")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{INPUT_FILE}' not found.")
#         return

#     data = []
#     for change in dataset:
#         data.append({'snippet': change['buggy_change'], 'label': change['buggy_label']})
#         data.append({'snippet': change['fixed_change'], 'label': change['fixed_label']})

#     df = pd.DataFrame(data)

#     print("Extracting structured features...")
#     structured = df['snippet'].apply(extract_features_from_snippet)
#     feature_columns = [
#         'num_lines', 'num_chars', 'keyword_count', 'control_count',
#         'exception_count', 'method_calls', 'comment_ratio',
#         'suspicious_count', 'line_stddev', 'buggy_api_count', 'entropy'
#     ]
#     df_structured = pd.DataFrame(structured.tolist(), columns=feature_columns)

#     print("Generating CodeBERT embeddings...")
#     embeddings = np.array([get_codebert_embedding(code) for code in df['snippet']])

#     # Combine features
#     scaler = StandardScaler()
#     X_structured_scaled = scaler.fit_transform(df_structured)
#     X_combined = np.hstack([embeddings, X_structured_scaled])
#     y = df['label']

#     print(f"Final feature matrix shape: {X_combined.shape}")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_combined, y, test_size=0.25, random_state=42, stratify=(y > 0.5)
#     )

#     print("\n--- Training Ensemble Models ---")
#     cat_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_seed=42)
#     lgb_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
#     lr_model = LogisticRegression(max_iter=500)

#     cat_model.fit(X_train, y_train)
#     lgb_model.fit(X_train, y_train)
#     lr_model.fit(X_train, y_train)

#     ensemble = VotingClassifier(estimators=[
#         ('cat', cat_model),
#         ('lgb', lgb_model),
#         ('lr', lr_model)
#     ], voting='soft')

#     ensemble.fit(X_train, y_train)
#     y_pred = ensemble.predict(X_test)

#     print("\n--- FINAL MODEL EVALUATION REPORT ---")
#     print(classification_report((y_test > 0.5).astype(int), (y_pred > 0.5).astype(int),
#                                 target_names=['Not Buggy (0)', 'Buggy (1)']))
#     print(f"Final Accuracy on Test Set: {accuracy_score((y_test > 0.5).astype(int), (y_pred > 0.5).astype(int)):.4f}")

#     # Save everything
#     model_payload = {
#         'model': ensemble,
#         'scaler': scaler,
#         'codebert': model,
#         'tokenizer': tokenizer
#     }
#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(model_payload, f)

#     print(f"\nEnsemble model with CodeBERT and structured features saved to {OUTPUT_MODEL_FILE}")

# if __name__ == '__main__':
#     train_risk_model()




#0.69

# import json
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import StandardScaler, normalize
# from sklearn.ensemble import HistGradientBoostingClassifier
# import pickle
# import sys
# import torch
# from transformers import AutoTokenizer, AutoModel

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features_from_snippet

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_changes.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')
# EMBED_CACHE = 'data/distilbert_embeddings.npy'

# # Load DistilBERT
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("distilbert-base-uncased")

# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# def extract_metadata_features(snippet, file_path="", commit_msg=""):
#     file_path = file_path or ""
#     commit_msg = commit_msg or ""

#     return {
#         "msg_length": len(commit_msg),
#         "has_bug_keyword": int(any(kw in commit_msg.lower() for kw in ['fix', 'bug', 'patch'])),
#         "path_depth": file_path.count('/') + file_path.count('\\'),
#         "snippet_length": len(snippet)
#     }

# def train_risk_model():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     print("--- Loading Change-Level Dataset ---")
#     try:
#         with open(INPUT_FILE, 'r') as f:
#             dataset = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Dataset file '{INPUT_FILE}' not found.")
#         return

#     data = []
#     for change in dataset:
#         data.append({
#             'snippet': change['buggy_change'],
#             'label': change['buggy_label'],
#             'file_path': change.get('file_path', ''),
#             'commit_msg': change.get('commit_msg', '')
#         })
#         data.append({
#             'snippet': change['fixed_change'],
#             'label': change['fixed_label'],
#             'file_path': change.get('file_path', ''),
#             'commit_msg': change.get('commit_msg', '')
#         })

#     df = pd.DataFrame(data)

#     # Reduce sample size for fast debug
#     df = df.sample(n=300, random_state=42)

#     print("Extracting structured features...")
#     structured = df['snippet'].apply(extract_features_from_snippet)
#     feature_columns = [
#         'num_lines', 'num_chars', 'keyword_count', 'control_count',
#         'exception_count', 'method_calls', 'comment_ratio',
#         'suspicious_count', 'line_stddev', 'buggy_api_count', 'entropy'
#     ]
#     df_structured = pd.DataFrame(structured.tolist(), columns=feature_columns)

#     print("Extracting metadata features...")
#     metadata = df.apply(lambda row: extract_metadata_features(row['snippet'], row['file_path'], row['commit_msg']), axis=1)
#     df_metadata = pd.DataFrame(metadata.tolist())

#     print("Generating DistilBERT embeddings...")
#     if os.path.exists(EMBED_CACHE):
#         embeddings = np.load(EMBED_CACHE)
#     else:
#         embeddings = np.array([get_embedding(code) for code in df['snippet']])
#         np.save(EMBED_CACHE, embeddings)

#     # Normalize embeddings
#     embeddings = normalize(embeddings)

#     # Combine all features
#     scaler = StandardScaler()
#     X_structured_scaled = scaler.fit_transform(df_structured)
#     X_metadata_scaled = scaler.fit_transform(df_metadata)
#     X_combined = np.hstack([embeddings, X_structured_scaled, X_metadata_scaled])

#     # Convert labels to binary
#     y = df['label']
#     y_binary = (y > 0.5).astype(int)

#     # Drop NaNs
#     valid_mask = ~np.isnan(X_combined).any(axis=1)
#     X_combined = X_combined[valid_mask]
#     y_binary = y_binary[valid_mask]

#     print(f"Final feature matrix shape: {X_combined.shape}")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_combined, y_binary, test_size=0.25, random_state=42, stratify=y_binary
#     )

#     print("\n--- Training Gradient Boosting Model ---")
#     model_gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
#     model_gb.fit(X_train, y_train)
#     y_pred = model_gb.predict(X_test)

#     print("\n--- FINAL MODEL EVALUATION REPORT ---")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
#     print(f"Final Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")

#     model_payload = {
#         'model': model_gb,
#         'scaler': scaler,
#         'tokenizer': tokenizer,
#         'embedding_model': model
#     }
#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump(model_payload, f)

#     print(f"\nGradient boosting model saved to {OUTPUT_MODEL_FILE}")

# if __name__ == '__main__':
#     train_risk_model()

# import json
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import StandardScaler, normalize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from xgboost import XGBClassifier
# import pickle
# import sys
# import torch
# from transformers import AutoTokenizer, AutoModel

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from code_reviewer.feature_extractor import extract_features_from_snippet

# # --- Configuration ---
# INPUT_FILE = 'data/bugfix_changes.json'
# MODEL_DIR = 'models'
# OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

# # Load models
# distil_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# distil_model = AutoModel.from_pretrained("distilbert-base-uncased")
# codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")

# def get_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# def extract_metadata_features(snippet, file_path="", commit_msg=""):
#     file_path = file_path or ""
#     commit_msg = commit_msg or ""
#     return {
#         "msg_length": len(commit_msg),
#         "has_bug_keyword": int(any(kw in commit_msg.lower() for kw in ['fix', 'bug', 'patch'])),
#         "path_depth": file_path.count('/') + file_path.count('\\'),
#         "snippet_length": len(snippet)
#     }

# def tfidf_risk_score(snippets):
#     vectorizer = TfidfVectorizer(max_features=1000)
#     tfidf_matrix = vectorizer.fit_transform(snippets)
#     risky_tokens = ['null', 'catch', 'synchronized', 'Thread', 'System.exit', 'Exception']
#     token_index = [vectorizer.vocabulary_.get(tok.lower()) for tok in risky_tokens if vectorizer.vocabulary_.get(tok.lower()) is not None]
#     scores = tfidf_matrix[:, token_index].sum(axis=1).A1
#     return np.clip(scores / scores.max(), 0, 1)

# def train_risk_model():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     print("--- Loading Dataset ---")
#     with open(INPUT_FILE, 'r') as f:
#         dataset = json.load(f)

#     data = []
#     for change in dataset:
#         data.append({
#             'snippet': change['buggy_change'],
#             'file_path': change.get('file_path', ''),
#             'commit_msg': change.get('commit_msg', ''),
#             'label': None
#         })
#         data.append({
#             'snippet': change['fixed_change'],
#             'file_path': change.get('file_path', ''),
#             'commit_msg': change.get('commit_msg', ''),
#             'label': None
#         })

#     df = pd.DataFrame(data)
#     df = df.sample(n=400, random_state=42)

#     print("Scoring risk with TF-IDF...")
#     tfidf_scores = tfidf_risk_score(df['snippet'])
#     df['label'] = (tfidf_scores > 0.5).astype(int)

#     print("Extracting structured features...")
#     structured = df['snippet'].apply(extract_features_from_snippet)
#     df_structured = pd.DataFrame(structured.tolist())

#     print("Extracting metadata features...")
#     metadata = df.apply(lambda row: extract_metadata_features(row['snippet'], row['file_path'], row['commit_msg']), axis=1)
#     df_metadata = pd.DataFrame(metadata.tolist())

#     print("Generating fresh embeddings...")
#     distil_embed = np.array([get_embedding(code, distil_tokenizer, distil_model) for code in df['snippet']])
#     codebert_embed = np.array([get_embedding(code, codebert_tokenizer, codebert_model) for code in df['snippet']])
#     embeddings = normalize(np.hstack([distil_embed, codebert_embed]))

#     scaler = StandardScaler()
#     X_structured = scaler.fit_transform(df_structured)
#     X_metadata = scaler.fit_transform(df_metadata)
#     X = np.hstack([embeddings, X_structured, X_metadata])
#     y = df['label']

#     # Drop NaNs
#     valid_mask = ~np.isnan(X).any(axis=1)
#     X = X[valid_mask]
#     y = y[valid_mask]

#     print(f"Final shape: {X.shape}")

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

#     print("\n--- Training XGBoost ---")
#     pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
#     model = XGBClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=pos_weight, use_label_encoder=False, eval_metric='logloss')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     print("\n--- FINAL REPORT ---")
#     print(classification_report(y_test, y_pred, target_names=['Not Buggy (0)', 'Buggy (1)']))
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#     with open(OUTPUT_MODEL_FILE, 'wb') as f:
#         pickle.dump({
#             'model': model,
#             'scaler': scaler,
#             'tokenizers': {
#                 'distil': distil_tokenizer,
#                 'codebert': codebert_tokenizer
#             },
#             'embedding_models': {
#                 'distil': distil_model,
#                 'codebert': codebert_model
#             }
#         }, f)

#     print(f"\nModel saved to {OUTPUT_MODEL_FILE}")

# if __name__ == '__main__':
#     train_risk_model()


import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import sys

# Add path to import our specific feature extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code_reviewer.feature_extractor import extract_features_from_snippet

INPUT_FILE = 'data/bugfix_changes.json'
MODEL_DIR = 'models'
OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'risk_assessment_model.pkl')

def train_risk_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("--- Loading Dataset ---")
    try:
        with open(INPUT_FILE, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Please run scripts/mine_data.py first.")
        return

    data = []
    for change in dataset:
        data.append({'snippet': change['buggy_change'], 'label': 1})
        data.append({'snippet': change['fixed_change'], 'label': 0})
    
    df = pd.DataFrame(data)

    print("Extracting features (Expect 3 features)...")
    # This calls your updated feature extractor
    features = df['snippet'].apply(extract_features_from_snippet)
    
    # We verify that we are putting them into exactly 3 columns
    # These match the return values from feature_extractor.py: [num_lines, num_chars, keyword_count]
    df_features = pd.DataFrame(features.tolist(), columns=['lines', 'chars', 'keywords'])
    
    X = df_features
    y = df['label']

    print(f"Features extracted. Shape: {X.shape}") # Should print (Rows, 3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training LightGBM Model...")
    model = lgb.LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save model AND scaler so app.py can reuse them
    payload = {'model': model, 'scaler': scaler}
    with open(OUTPUT_MODEL_FILE, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Model saved to {OUTPUT_MODEL_FILE}")

if __name__ == '__main__':
    train_risk_model()
