import joblib
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
# joblib to save models
try:
    rf_model = joblib.load('random_forest_model.pkl')
    gb_model = joblib.load('gradient_boosting_model.pkl')
    lgbm_model = joblib.load('lightgbm_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
    cat_model = joblib.load('catboost_model.pkl')
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Preparation of data
file_path = r"C:\Users\walla\Documents\Frontend\catboostt\feature_vectors_syscallsbinders_frequency_5_Cat.csv"

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Separate features and target
X = data.drop(columns=["Class"])
y = data["Class"]

# Encode target labels to 0-based integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Creates the  predictions (Catboost different settings because of error)
predictions = []
model_names = ['Random Forest', 'Gradient Boosting', 'LightGBM', 'XGBoost', 'CatBoost']

for model, name in zip(
    [rf_model, gb_model, lgbm_model, xgb_model, cat_model],
    model_names
):
    try:
        pred = model.predict(X_test_scaled)
        # Flatten predictions if they have an extra dimension (e.g., CatBoost)
        if len(pred.shape) > 1 and pred.shape[1] == 1:
            pred = pred.flatten()
        print(f"{name} predictions shape: {pred.shape}")
        predictions.append(pred)
    except Exception as e:
        print(f"Error predicting with {name}: {e}")
        predictions.append(None)

# Validation Process
test_size = X_test_scaled.shape[0]
valid_predictions = [
    pred for pred in predictions if pred is not None and pred.shape == (test_size,)
]
invalid_models = [
    name for pred, name in zip(predictions, model_names)
    if pred is None or pred.shape != (test_size,)
]

if invalid_models:
    print(f"\nWarning: The following models had mismatched prediction sizes or errors and will be excluded: {', '.join(invalid_models)}")

# Start Ensemble Hard Voting for Hybrid Ensemble
if len(valid_predictions) > 1:
    try:
        # Stack valid predictions for Hard Voting
        predictions_stack = np.vstack(valid_predictions)
        print("Predictions stacked successfully.")

        # Perform majority voting
        ensemble_pred, _ = mode(predictions_stack, axis=0)
        ensemble_pred = ensemble_pred.flatten()

        # Evaulation of ensemble with metrics
        acc = accuracy_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred, average='macro')
        recall = recall_score(y_test, ensemble_pred, average='macro')
        precision = precision_score(y_test, ensemble_pred, average='macro')

        print("\n== Ensemble (Hard Voting) Results ==")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score (macro): {f1:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"Precision (macro): {precision:.4f}")

        # Confusion Matrix and Classification Report
        y_pred_decoded = encoder.inverse_transform(ensemble_pred)
        y_test_decoded = encoder.inverse_transform(y_test)

        print("\nConfusion Matrix (Ensemble - Hard Voting):")
        print(confusion_matrix(y_test_decoded, y_pred_decoded))

        print("\nClassification Report (Ensemble - Hard Voting):")
        print(classification_report(y_test_decoded, y_pred_decoded))
    except Exception as e:
        print(f"Error during ensemble voting: {e}")
else:
    print("\nEnsemble cannot be performed. Not enough valid predictions.")
