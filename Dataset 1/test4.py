import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
model_path = "C:/codes5/aifin/E6/comb/best_base_model_RandomForest.pkl"
logging.info(f"Loading trained model from {model_path}...")
model = joblib.load(model_path)

# Define the features used in training
selected_features = [
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Flow IAT Min',
    'Flow Duration',
    'Total Length of Fwd Packets',
    'Packet Length Variance',
    'Bwd Packet Length Max',
    'Avg Fwd Segment Size',
    'Max Packet Length',
    'Flow IAT Max',
    'Fwd Packet Length Mean',
    'Subflow Fwd Bytes',
    'Bwd Packet Length Mean',
    'Avg Bwd Segment Size',
    'Flow Packets/s',
    'Fwd Packet Length Max',
    'Flow Bytes/s',
    'Fwd Packet Length Std',
    'Bwd Packet Length Std',
    'Bwd Packet Length Min'
]

# Load input data from the CSV file
csv_file_path = "flow_features.csv"  # Path to the generated CSV file
logging.info(f"Loading input data from {csv_file_path}...")
data = pd.read_csv(csv_file_path)

# Ensure the CSV contains all required features
missing_features = set(selected_features) - set(data.columns)
if missing_features:
    logging.error(f"Missing features in CSV: {missing_features}")
    raise ValueError(f"The following features are missing in the input CSV: {missing_features}")

# Extract the required features
input_data = data[selected_features]

# Normalize features
logging.info("Normalizing features...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(input_data)

# Apply PCA
n_components = 20  # Explicitly match the number of features used during training
logging.info(f"Applying PCA with n_components={n_components}...")
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_normalized)

# Verify feature compatibility with the model
if X_pca.shape[1] != model.n_features_in_:
    logging.error(f"Feature mismatch: Model expects {model.n_features_in_} features, but input has {X_pca.shape[1]}")
    raise ValueError("Input features do not match the model's training configuration.")

# Predict using the trained model
logging.info("Making predictions for input data...")
outputs = model.predict(X_pca)

# Define output labels for the prediction
output_labels = ['Android_Adware', 'Android_Scareware', 'Android_SMS_Malware', 'Benign']
predicted_labels = [output_labels[output] for output in outputs]

# Display results
logging.info("Displaying predictions...")
for i, (index, row) in enumerate(data.iterrows()):
    print(f"Input Example {i+1}: {row[selected_features].to_dict()}")
    print(f"Predicted Label = {predicted_labels[i]}\n")

# Check for any missing labels
missing_labels = set(output_labels) - set(predicted_labels)
if missing_labels:
    for label in missing_labels:
        logging.warning(f"Warning: No example predicted for label {label}")
