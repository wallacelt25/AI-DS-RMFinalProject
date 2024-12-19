import optuna
import pandas as pd
import numpy as np
import joblib  # For saving models

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

# Loading and Preparing Dataset for Model
def load_and_prepare_data(file_path):
    """
    Load the dataset, encode target labels, split into training and testing sets,
    and scale the features.

    Parameters:
    - file_path (str): Path to the CSV dataset.

    Returns:
    - X_train_scaled (np.ndarray): Scaled training features.
    - X_test_scaled (np.ndarray): Scaled testing features.
    - y_train (np.ndarray): Training labels.
    - y_test (np.ndarray): Testing labels.
    - encoder (LabelEncoder): Fitted label encoder.
    """
    data = pd.read_csv(file_path)
    print("Dataset shape:", data.shape)
    print("Columns:", data.columns)

    # Features and target
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Encode target labels to 0-based integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder

# SMOTE to balance imbalanced classes
def balance_dataset(X_train, y_train):
    """
    Balance the training dataset using SMOTE to handle class imbalance.

    Parameters:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training labels.

    Returns:
    - X_train_balanced (np.ndarray): Balanced training features.
    - y_train_balanced (np.ndarray): Balanced training labels.
    """
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("Balanced Dataset Shape:", X_train_balanced.shape)
    return X_train_balanced, y_train_balanced

# Setting Optuna 
def objective_xgb(trial, X, y):
    """Objective function for XGBoost"""
    param = {
        'eta': trial.suggest_float("xgb_eta", 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int("xgb_max_depth", 2, 32, log=True),
        'n_estimators': trial.suggest_int("xgb_n_estimators", 50, 300, step=50),
        'subsample': trial.suggest_float("xgb_subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        'use_label_encoder': False,
        'eval_metric': "logloss",
        'random_state': 42,
        'n_jobs': -1
    }

    model = XGBClassifier(**param)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, val_idx in kf.split(X):
        X_train_kf, X_val_kf = X[train_idx], X[val_idx]
        y_train_kf, y_val_kf = y[train_idx], y[val_idx]

        model.fit(X_train_kf, y_train_kf)
        y_pred = model.predict(X_val_kf)
        accuracies.append(accuracy_score(y_val_kf, y_pred))

    return np.mean(accuracies)

def objective_lgb(trial, X, y):
    """Objective function for LightGBM"""
    param = {
        'num_leaves': trial.suggest_int("lgb_num_leaves", 4, 128, log=True),
        'learning_rate': trial.suggest_float("lgb_learning_rate", 1e-3, 1e-1, log=True),
        'n_estimators': trial.suggest_int("lgb_n_estimators", 50, 300, step=50),
        'min_child_samples': trial.suggest_int("lgb_min_child_samples", 5, 50),
        'subsample': trial.suggest_float("lgb_subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
        'reg_alpha': trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }

    model = lgb.LGBMClassifier(**param)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, val_idx in kf.split(X):
        X_train_kf, X_val_kf = X[train_idx], X[val_idx]
        y_train_kf, y_val_kf = y[train_idx], y[val_idx]

        model.fit(X_train_kf, y_train_kf)
        y_pred = model.predict(X_val_kf)
        accuracies.append(accuracy_score(y_val_kf, y_pred))

    return np.mean(accuracies)

def objective_cat(trial, X, y):
    """Optimized Objective function for CatBoost"""
    param = {
        'iterations': trial.suggest_int("cat_iterations", 100, 300, step=100),  # Reduced max iterations
        'depth': trial.suggest_int("cat_depth", 4, 8),                        # Limited depth
        'learning_rate': trial.suggest_float("cat_learning_rate", 1e-3, 1e-1, log=True),
        'l2_leaf_reg': trial.suggest_float("cat_l2_leaf_reg", 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int("cat_border_count", 32, 100),       # Reduced border_count
        'thread_count': 4,                                                  # Limited threads
        'random_seed': 42,
        'verbose': False
    }

    model = CatBoostClassifier(**param)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for faster execution
    accuracies = []

    for train_idx, val_idx in kf.split(X):
        X_train_kf, X_val_kf = X[train_idx], X[val_idx]
        y_train_kf, y_val_kf = y[train_idx], y[val_idx]

        model.fit(
            X_train_kf, y_train_kf,
            eval_set=(X_val_kf, y_val_kf),
            early_stopping_rounds=30,  # Reduced early stopping
            verbose=False
        )
        y_pred = model.predict(X_val_kf)
        accuracies.append(accuracy_score(y_val_kf, y_pred))

    return np.mean(accuracies)

# Set to start optuna to study parameters
def run_optuna_studies(X, y):
    """
    Run Optuna studies for XGBoost, LightGBM, and CatBoost to find the best hyperparameters.

    Parameters:
    - X (np.ndarray): Training features.
    - y (np.ndarray): Training labels.

    Returns:
    - best_xgb (optuna.trial.Trial): Best trial for XGBoost.
    - best_lgb (optuna.trial.Trial): Best trial for LightGBM.
    - best_cat (optuna.trial.Trial): Best trial for CatBoost.
    """
    print("\n==== Starting Optuna Study for XGBoost ====")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=30)  # Reduced trials
    best_xgb = study_xgb.best_trial

    print("\n==== Starting Optuna Study for LightGBM ====")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X, y), n_trials=30)  # Reduced trials
    best_lgb = study_lgb.best_trial

    print("\n==== Starting Optuna Study for CatBoost ====")
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(lambda trial: objective_cat(trial, X, y), n_trials=30)  # Reduced trials
    best_cat = study_cat.best_trial

    return best_xgb, best_lgb, best_cat

#Define to print optuna results: accuracy and parameters
def print_best_trials(best_xgb, best_lgb, best_cat):
    """
    Print the best hyperparameters and corresponding accuracy for each model.

    Parameters:
    - best_xgb (optuna.trial.Trial): Best trial for XGBoost.
    - best_lgb (optuna.trial.Trial): Best trial for LightGBM.
    - best_cat (optuna.trial.Trial): Best trial for CatBoost.
    """
    print("\n==== Best XGBoost Trial ====")
    print("  Accuracy:", best_xgb.value)
    print("  Params:", best_xgb.params)

    print("\n==== Best LightGBM Trial ====")
    print("  Accuracy:", best_lgb.value)
    print("  Params:", best_lgb.params)

    print("\n==== Best CatBoost Trial ====")
    print("  Accuracy:", best_cat.value)
    print("  Params:", best_cat.params)

#Set output for evaluation metrics
def evaluate_model(model, model_name, X_test, y_test, encoder):
    """
    Evaluate the trained model on the test set and print performance metrics.

    Parameters:
    - model (classifier): Trained classifier.
    - model_name (str): Name of the model (for display purposes).
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing labels.
    - encoder (LabelEncoder): Fitted label encoder.

    Returns:
    - y_pred (np.ndarray): Predicted labels.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    print(f"\n== Final {model_name} Results ==")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"Precision (macro): {precision:.4f}")

    y_pred_decoded = encoder.inverse_transform(y_pred)
    y_test_decoded = encoder.inverse_transform(y_test)

    print(f"\nConfusion Matrix ({model_name}):")
    print(confusion_matrix(y_test_decoded, y_pred_decoded))

    print(f"\nClassification Report ({model_name}):")
    print(classification_report(y_test_decoded, y_pred_decoded))

    return y_pred

# Start training, evaluation metrics output and then save model for ensemble
def train_evaluate_save_models(best_xgb, best_lgb, best_cat, X_train, X_test, y_train, y_test, encoder):
    """
    Train, evaluate, and save Random Forest, Gradient Boosting, LightGBM, XGBoost, and CatBoost models.

    Parameters:
    - best_xgb (optuna.trial.Trial): Best trial for XGBoost.
    - best_lgb (optuna.trial.Trial): Best trial for LightGBM.
    - best_cat (optuna.trial.Trial): Best trial for CatBoost.
    - X_train (np.ndarray): Balanced training features.
    - X_test (np.ndarray): Testing features.
    - y_train (np.ndarray): Balanced training labels.
    - y_test (np.ndarray): Testing labels.
    - encoder (LabelEncoder): Fitted label encoder.
    """
    #Random Forest
    print("\n== Training Random Forest ==")
    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    evaluate_model(rf_model, "Random Forest", X_test, y_test, encoder)
    joblib.dump(rf_model, "random_forest_model.pkl")
    print("Random Forest model saved to 'random_forest_model.pkl'.")
    
    #Gradient Boosting
    print("\n== Training Gradient Boosting ==")
    gb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 1.0,
        'random_state': 42
    }
    gb_model = GradientBoostingClassifier(**gb_params)
    gb_model.fit(X_train, y_train)
    print("Gradient Boosting training completed.")
    evaluate_model(gb_model, "Gradient Boosting", X_test, y_test, encoder)
    joblib.dump(gb_model, "gradient_boosting_model.pkl")
    print("Gradient Boosting model saved to 'gradient_boosting_model.pkl'.")
    
    #LightGBM with Optuna Set Parameters
    print("\n== Training LightGBM with Best Parameters ==")
    lgb_params = best_lgb.params
    lgb_model = lgb.LGBMClassifier(
        num_leaves=lgb_params.get("lgb_num_leaves", 31),
        learning_rate=lgb_params.get("lgb_learning_rate", 0.1),
        n_estimators=lgb_params.get("lgb_n_estimators", 100),
        min_child_samples=lgb_params.get("lgb_min_child_samples", 20),
        subsample=lgb_params.get("lgb_subsample", 1.0),
        colsample_bytree=lgb_params.get("lgb_colsample_bytree", 1.0),
        reg_alpha=lgb_params.get("lgb_reg_alpha", 0.0),
        reg_lambda=lgb_params.get("lgb_reg_lambda", 0.0),
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    print("LightGBM training completed.")
    evaluate_model(lgb_model, "LightGBM", X_test, y_test, encoder)
    joblib.dump(lgb_model, "lightgbm_model.pkl")
    print("LightGBM model saved to 'lightgbm_model.pkl'.")


    #XGBoost with Optuna Set Parameters

    print("\n== Training XGBoost with Best Parameters ==")
    xgb_params = best_xgb.params
    xgb_model = XGBClassifier(
        eta=xgb_params.get("xgb_eta", 0.3),
        max_depth=xgb_params.get("xgb_max_depth", 6),
        n_estimators=xgb_params.get("xgb_n_estimators", 100),
        subsample=xgb_params.get("xgb_subsample", 1.0),
        colsample_bytree=xgb_params.get("xgb_colsample_bytree", 1.0),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training completed.")
    evaluate_model(xgb_model, "XGBoost", X_test, y_test, encoder)
    joblib.dump(xgb_model, "xgboost_model.pkl")
    print("XGBoost model saved to 'xgboost_model.pkl'.")


    #CatBoost with Optuna SetParameters

    print("\n== Training CatBoost with Best Parameters ==")
    cat_params = best_cat.params
    cat_model = CatBoostClassifier(
        iterations=cat_params.get("cat_iterations", 300),
        depth=cat_params.get("cat_depth", 6),
        learning_rate=cat_params.get("cat_learning_rate", 0.05),
        l2_leaf_reg=cat_params.get("cat_l2_leaf_reg", 3.0),
        border_count=cat_params.get("cat_border_count", 100),
        thread_count=cat_params.get("thread_count", 4),
        random_seed=cat_params.get("random_seed", 42),
        verbose=cat_params.get("verbose", False)
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=30,
        verbose=False
    )
    print("CatBoost training completed.")
    evaluate_model(cat_model, "CatBoost", X_test, y_test, encoder)
    joblib.dump(cat_model, "catboost_model.pkl")
    print("CatBoost model saved to 'catboost_model.pkl'.")

# Execute main points
if __name__ == "__main__":
    # Specify the path to the filtered dataset
    file_path = r"C:\Users\walla\Documents\Frontend\catboostt\feature_vectors_syscallsbinders_frequency_5_Cat.csv"
    # Feature Selection Process:
    # - Employed Random Forest Classification to determine feature importances.
    # - Trained a Random Forest model on the entire dataset to compute feature importance scores.
    # - Selected the top N features based on their importance scores to create a reduced feature set.
    # - Saved the filtered dataset with only the selected features to 'updated_filtered_dataset.csv'.

    # Load and prepare data
    X_train, X_test, y_train, y_test, encoder = load_and_prepare_data(file_path)

    # Balance the dataset with SMOTE
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Run Optuna studies for LightGBM, XGBoost, and CatBoost
    best_xgb, best_lgb, best_cat = run_optuna_studies(X_train_balanced, y_train_balanced)

    # Print best trials from Optuna
    print_best_trials(best_xgb, best_lgb, best_cat)

    # Train, evaluate, and save models with Optuna-optimized hyperparameters
    train_evaluate_save_models(best_xgb, best_lgb, best_cat, X_train_balanced, X_test, y_train_balanced, y_test, encoder)
