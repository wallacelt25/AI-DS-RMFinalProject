import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report
)
from catboost import CatBoostClassifier
import joblib
import optuna
from imblearn.over_sampling import SMOTE
import warnings
import sklearn
import catboost

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load and Inspect the Dataset for training
# Path to your dataset
file_path = r"C:\Users\walla\Documents\Frontend\catboostt\feature_vectors_syscallsbinders_frequency_5_Cat.csv"

# Load the dataset
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Display the first few rows and the shape of the dataset
print("\nDataset shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\nFirst 5 rows of the dataset:")
print(data.head())

#Set target labels
# Separate features and target
X = data.drop(columns=["Class"])
y = data["Class"]

# Initialize LabelEncoder
encoder = LabelEncoder()

# Fit and transform the target labels
y_encoded = encoder.fit_transform(y)

# Display the classes and their corresponding encoded labels
print("\nClasses:", encoder.classes_)
print("Encoded labels:", np.unique(y_encoded))

# Dataset splitting for training
# Define the test size and random state
test_size = 0.2  # 20% for testing
random_state = 42  # Ensure reproducibility
stratify = y_encoded  # Maintain class distribution

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=test_size,
    random_state=random_state,
    stratify=stratify
)

# Display the shapes of the splits
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Scales features for catboost
# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Display the first few rows of the scaled training data
print("\nFirst 5 rows of scaled training data:")
print(X_train_scaled[:5])

# SMOTE for imbalanced classes
# Initialize SMOTE
smote = SMOTE(random_state=random_state)

# Apply SMOTE to the training data
try:
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("\nSMOTE applied successfully.")
    print("Resampled training set shape:", X_train_resampled.shape)
    print("Resampled class distribution:", np.bincount(y_train_resampled))
except Exception as e:
    print(f"Error applying SMOTE: {e}")
    exit(1)

#Set Optuna to find best parameters
def objective(trial):
    # Define the hyperparameter search space
    param = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'thread_count': 4,
        'random_seed': random_state,
        'verbose': False
    }
    
    # Initialize the CatBoostClassifier with the current hyperparameters
    model = CatBoostClassifier(**param)
    
    # Fit the model on the resampled training data
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=(X_test_scaled, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Make predictions on the validation set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate the F1-score (macro)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return f1  # Optuna will try to maximize this metric

#Running of Optuna
print("\nStarting Optuna hyperparameter optimization...")

# Create a study object and specify the direction to maximize the F1-score
study = optuna.create_study(direction='maximize', study_name='CatBoost_Optuna_Study')

# Optimize the objective function
study.optimize(objective, n_trials=50, timeout=600)  # Adjust n_trials and timeout as needed

# Display the best hyperparameters
print("\nNumber of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value (F1-score):", trial.value)
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Catboost training with Optuna
# Extract the best hyperparameters
best_params = study.best_trial.params

# Update the parameters with fixed values
best_params.update({
    'thread_count': 4,
    'random_seed': random_state,
    'verbose': False
})

print("\nTraining CatBoost with the best hyperparameters...")

# Initialize the CatBoostClassifier with the best hyperparameters
final_cat_model = CatBoostClassifier(**best_params)

# Train the model on the resampled training data
try:
    final_cat_model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=(X_test_scaled, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    print("CatBoost model trained successfully with optimal hyperparameters.")
except Exception as e:
    print(f"Error training CatBoost model with optimal hyperparameters: {e}")
    exit(1)

# Evaluation of Model with Metrics
# Make predictions on the test set
try:
    y_pred_final = final_cat_model.predict(X_test_scaled)
    print("\nCatBoost predictions made successfully.")
except Exception as e:
    print(f"Error making predictions with the final CatBoost model: {e}")
    exit(1)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final, average='macro')
recall = recall_score(y_test, y_pred_final, average='macro')
precision = precision_score(y_test, y_pred_final, average='macro')

print("\n== Final CatBoost Model Evaluation ==")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score (macro): {f1:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"Precision (macro): {precision:.4f}")

# Confusion Matrix and Classification Report
try:
    y_pred_final_decoded = encoder.inverse_transform(y_pred_final)
    y_test_decoded = encoder.inverse_transform(y_test)

    print("\nConfusion Matrix (CatBoost):")
    print(confusion_matrix(y_test_decoded, y_pred_final_decoded))

    print("\nClassification Report (CatBoost):")
    print(classification_report(y_test_decoded, y_pred_final_decoded))
except Exception as e:
    print(f"Error decoding labels: {e}")

# Save model of Catboost
# Save the trained CatBoost model
try:
    joblib.dump(final_cat_model, 'catboost_model.pkl')
    print("\nCatBoost model saved to 'catboost_model.pkl'.")
except Exception as e:
    print(f"Error saving CatBoost model: {e}")
    exit(1)
