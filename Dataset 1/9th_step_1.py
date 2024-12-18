import pandas as pd
import os
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from xgboost.callback import EarlyStopping
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
import joblib
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the directory structure exists
output_dir = "E7"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
dataset_path = 'Android_Malware.csv'
data = pd.read_csv(dataset_path, low_memory=False)

# Standardize column names
data.columns = data.columns.str.strip()

# Define selected features
selected_features = [
    'Init_Win_bytes_forward',       # This feature had the highest average importance across all models, with a value of 62.09.
    'Init_Win_bytes_backward',      # Ranked second, with an average importance of 55.14.
    'Flow IAT Min',                 # This feature, indicating the minimum inter-arrival time of packets, had an average importance of 37.65.
    'Flow Duration',                # The total duration of the flow ranked fourth, with an average importance of 35.19.
    'Total Length of Fwd Packets',  # The total length of forward packets scored an average importance of 34.80.
    'Packet Length Variance',       # This feature, representing the variability in packet lengths, had an average importance of 32.14.
    'Bwd Packet Length Max',        # The maximum length of backward packets scored 30.68 in average importance.
    'Avg Fwd Segment Size',         # The average forward segment size ranked eighth, with an average importance of 30.26.
    'Max Packet Length',            # This feature had an average importance of 28.63.
    'Flow IAT Max',                 # Representing the maximum inter-arrival time of packets, this feature scored 27.73 on average.
    'Fwd Packet Length Mean',       # The mean length of forward packets ranked eleventh, with an average importance of 26.77.
    'Subflow Fwd Bytes',            # The number of bytes in the forward subflow had an average importance of 26.73.
    'Bwd Packet Length Mean',       # The mean length of backward packets scored 25.32 on average.
    'Avg Bwd Segment Size',         # The average size of backward segments ranked fourteenth, with an average importance of 24.03.
    'Flow Packets/s',               # The number of packets per second in the flow scored 23.32 in average importance.
    'Fwd Packet Length Max',        # The maximum length of forward packets had an average importance of 23.00.
    'Flow Bytes/s',                 # The number of bytes per second in the flow ranked seventeenth, with an average importance of 22.45.
    'Fwd Packet Length Std',        # The standard deviation of forward packet lengths scored 21.56 in average importance.
    'Bwd Packet Length Std',        # The standard deviation of backward packet lengths had an average importance of 21.19.
    'Bwd Packet Length Min'         # The minimum length of backward packets rounded out the top 20, with an average importance of 20.87.
]


# Handle missing values and ensure all data is numeric
numeric_features = data[selected_features].select_dtypes(include=['float64', 'int64']).columns
non_numeric_features = data[selected_features].select_dtypes(exclude=['float64', 'int64']).columns

# Impute numeric features
imputer_numeric = SimpleImputer(strategy='mean')
data[numeric_features] = imputer_numeric.fit_transform(data[numeric_features])

# Impute non-numeric features (if any)
if len(non_numeric_features) > 0:
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')
    data[non_numeric_features] = imputer_non_numeric.fit_transform(data[non_numeric_features])

# Convert all remaining non-numeric columns to numeric
for col in selected_features:
    if not pd.api.types.is_numeric_dtype(data[col]):
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill any remaining NaN values
data[selected_features] = data[selected_features].fillna(0)

# Encode labels for multi-class classification
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Label'])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(data[selected_features])

# Handle class imbalance
smoteenn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smoteenn.fit_resample(X, y_encoded)

def optuna_objective_lightgbm(trial):
    param = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_balanced)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_balanced, y_balanced):
        X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
        y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=10),  # Early stopping callback
                log_evaluation(0)  # Optional: Suppress logs during evaluation
            ]
        )
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='weighted'))

    return np.mean(scores)


# Optuna objective function for XGBoost
def optuna_objective_xgboost(trial):
    param = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_balanced)),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_balanced, y_balanced):
        X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
        y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]

        model = xgb.XGBClassifier(**param, eval_metric='mlogloss')
        # early_stopping_callback = EarlyStopping(rounds=10, save_best=True)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='weighted'))

    return np.mean(scores)

# Optuna objective function for CatBoost
def optuna_objective_catboost(trial):
    param = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 15),
        'iterations': trial.suggest_int('iterations', 50, 200),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_balanced, y_balanced):
        X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
        y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]

        model = cb.CatBoostClassifier(**param)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='weighted'))

    return np.mean(scores)

# Run Optuna for each model
logging.info("Running Optuna for LightGBM hyperparameter tuning...")
study_lightgbm = optuna.create_study(direction='maximize')
study_lightgbm.optimize(optuna_objective_lightgbm, n_trials=30)

logging.info("Running Optuna for XGBoost hyperparameter tuning...")
study_xgboost = optuna.create_study(direction='maximize')
study_xgboost.optimize(optuna_objective_xgboost, n_trials=30)

logging.info("Running Optuna for CatBoost hyperparameter tuning...")
study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(optuna_objective_catboost, n_trials=30)

# Save the best hyperparameters for each model
best_params_path = os.path.join(output_dir, "best_params.txt")
with open(best_params_path, "w") as f:
    f.write("Best LightGBM Parameters:\n")
    for key, value in study_lightgbm.best_params.items():
        f.write(f"{key}: {value}\n")
    f.write("\nBest XGBoost Parameters:\n")
    for key, value in study_xgboost.best_params.items():
        f.write(f"{key}: {value}\n")
    f.write("\nBest CatBoost Parameters:\n")
    for key, value in study_catboost.best_params.items():
        f.write(f"{key}: {value}\n")
logging.info(f"Best parameters saved to {best_params_path}.")

# Use the best models
best_lightgbm = lgb.LGBMClassifier(**study_lightgbm.best_params)
best_xgboost = xgb.XGBClassifier(**study_xgboost.best_params, use_label_encoder=False, eval_metric='mlogloss')
best_catboost = cb.CatBoostClassifier(**study_catboost.best_params, verbose=0)

# Define base models with tuned parameters
def create_models():
    return {
        'Tuned_LightGBM': best_lightgbm,
        'Tuned_XGBoost': best_xgboost,
        'Tuned_CatBoost': best_catboost,
        'RandomForest': RandomForestClassifier(n_estimators=60, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=39, random_state=42)
    }


# Train and evaluate models and create meta-features
meta_features = []
meta_labels = []
best_model_name = None
best_model = None
best_score = -np.inf  # Initialize to negative infinity
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

step_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

models = create_models()
for train_index, val_index in skf.split(X_balanced, y_balanced):
    X_train, X_val = X_balanced[train_index], X_balanced[val_index]
    y_train, y_val = y_balanced[train_index], y_balanced[val_index]

    fold_meta_features = []
    for model_name, model in models.items():
        logging.info(f"Training {model_name} for stacking...")
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)
        fold_meta_features.append(y_pred)

        # Track performance metrics per model
        y_pred_class = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred_class)
        prec = precision_score(y_val, y_pred_class, average='weighted')
        rec = recall_score(y_val, y_pred_class, average='weighted')
        f1 = f1_score(y_val, y_pred_class, average='weighted')

        step_metrics['accuracy'].append(acc)
        step_metrics['precision'].append(prec)
        step_metrics['recall'].append(rec)
        step_metrics['f1'].append(f1)

        # Track the best base model based on F1 score
        if f1 > best_score:
            best_score = f1
            best_model_name = model_name
            best_model = model

    meta_features.append(np.hstack(fold_meta_features))
    meta_labels.append(y_val)

meta_features = np.vstack(meta_features)
meta_labels = np.hstack(meta_labels)

# Train the meta-learner
meta_learner = LogisticRegression()
meta_learner.fit(meta_features, meta_labels)

# Save the meta-learner
meta_model_path = os.path.join(output_dir, "stacked_meta_model.pkl")
joblib.dump(meta_learner, meta_model_path)
logging.info(f"Meta-learner saved to {meta_model_path}.")

# Save the best base model
if best_model:
    best_model_path = os.path.join(output_dir, f"best_base_model_{best_model_name}.pkl")
    joblib.dump(best_model, best_model_path)
    logging.info(f"Best base model ({best_model_name}) saved to {best_model_path}.")

# Plot metrics for each step
logging.info("Saving step-wise metric graphs...")
for metric_name, values in step_metrics.items():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(values) + 1), values, label=metric_name, marker='o')
    plt.title(f"{metric_name.capitalize()} Over Training Steps")
    plt.xlabel("Step")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{metric_name}_steps.png"))
    plt.close()

# Combined graph
plt.figure(figsize=(10, 6))
for metric_name, values in step_metrics.items():
    plt.plot(range(1, len(values) + 1), values, label=metric_name, marker='o')
plt.title("Performance Metrics Over Steps")
plt.xlabel("Step")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "combined_metrics_steps.png"))
plt.close()

# Evaluate and save final model metrics
logging.info("Evaluating the saved meta-learner...")
y_meta_pred = meta_learner.predict(meta_features)
final_acc = accuracy_score(meta_labels, y_meta_pred)
final_prec = precision_score(meta_labels, y_meta_pred, average='weighted')
final_rec = recall_score(meta_labels, y_meta_pred, average='weighted')
final_f1 = f1_score(meta_labels, y_meta_pred, average='weighted')

# Log final model metrics
final_metrics_path = os.path.join(output_dir, "final_model_metrics.txt")
with open(final_metrics_path, "w") as f:
    f.write("Final Model Metrics (Meta-Learner):\n")
    f.write(f"Accuracy: {final_acc}\n")
    f.write(f"Precision: {final_prec}\n")
    f.write(f"Recall: {final_rec}\n")
    f.write(f"F1 Score: {final_f1}\n")
logging.info(f"Final model metrics saved to {final_metrics_path}.")

# Plot final model metrics
plt.figure(figsize=(10, 6))
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [final_acc, final_prec, final_rec, final_f1]
plt.bar(metric_names, metric_values, color='skyblue')
plt.title("Final Model Metrics (Meta-Learner)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig(os.path.join(output_dir, "final_model_metrics.png"))
plt.close()
logging.info("Final model metrics graph saved.")