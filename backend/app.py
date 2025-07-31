import pandas as pd
from flask import Flask, request, jsonify
import joblib
import os
from sklearn.preprocessing import LabelEncoder # Ensure LabelEncoder is imported as it's used in label_encoders.pkl
from flask_cors import CORS # Import CORS for cross-origin requests

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for all origins, allowing your frontend (served from file:// or a different domain)
# to make requests to this backend. In a production environment, you should restrict this
# to specific origins for security.
CORS(app)

# Define the directory where models and encoders are saved
# This assumes 'fitverse_models' is a subdirectory within the 'backend' folder
# where app.py resides.
MODELS_DIR = 'fitverse_models'

# Global variables to store loaded models, encoders, and feature columns
# These will be loaded once when the application starts.
loaded_models = {}
loaded_label_encoders = {}
loaded_feature_columns = []
# List of target variables, must match the keys used when saving models
target_variables = ['Exercises', 'Equipment', 'Diet', 'Recommendation']

def load_assets():
    """
    Loads all trained machine learning models, label encoders, and feature column names
    from the specified MODELS_DIR. This function is called once when the Flask app starts
    to avoid reloading assets on every request.
    """
    global loaded_models, loaded_label_encoders, loaded_feature_columns

    print("Loading assets for the FitVerse backend...")

    # Load each trained Random Forest model
    for target_col in target_variables:
        # Construct the full path to the model file
        model_filename = os.path.join(MODELS_DIR, f'random_forest_model_{target_col.lower().replace(" ", "_")}.pkl')
        if os.path.exists(model_filename):
            loaded_models[target_col] = joblib.load(model_filename)
            print(f"Loaded model for {target_col}")
        else:
            # If a model file is missing, raise an error to prevent the app from starting
            print(f"Error: Model file '{model_filename}' not found. Ensure it's in the '{MODELS_DIR}' directory.")
            raise FileNotFoundError(f"Model file not found: {model_filename}")

    # Load the dictionary of LabelEncoder objects
    label_encoders_filename = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    if os.path.exists(label_encoders_filename):
        loaded_label_encoders = joblib.load(label_encoders_filename)
        print("Loaded label encoders.")
    else:
        print(f"Error: Label encoders file '{label_encoders_filename}' not found.")
        raise FileNotFoundError(f"Label encoders file not found: {label_encoders_filename}")

    # Load the list of expected feature column names (for input data alignment)
    feature_columns_filename = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    if os.path.exists(feature_columns_filename):
        loaded_feature_columns = joblib.load(feature_columns_filename)
        print("Loaded feature columns.")
    else:
        print(f"Error: Feature columns file '{feature_columns_filename}' not found.")
        raise FileNotFoundError(f"Feature columns file not found: {feature_columns_filename}")

    print("All backend assets loaded successfully!")

# Load assets once when the Flask app starts.
# This block attempts to load the assets. If any are missing, it prints an error
# and exits the application, preventing it from running in an incomplete state.
try:
    load_assets()
except FileNotFoundError as e:
    print(f"Failed to load assets: {e}. Please ensure all .pkl files are in the '{MODELS_DIR}' directory.")
    # In a production environment, you might log this error and exit more gracefully.
    exit(1) # Exit with a non-zero status code to indicate an error

# Define a simple home route for testing if the server is running
@app.route('/')
def home():
    """
    A simple home route to confirm the Flask backend is accessible.
    """
    return "FitVerse Backend is running! Send POST requests to /predict for recommendations."

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to make fitness plan predictions.
    It expects a JSON payload with user input features.
    """
    try:
        # Get JSON data from the request body
        user_data = request.get_json()
        if not user_data:
            return jsonify({"error": "Invalid JSON data. Please send a JSON object in the request body."}), 400

        print(f"Received data for prediction: {user_data}")

        # --- Data Preprocessing (CRITICAL: MUST match your training preprocessing) ---
        # Convert received data into a pandas DataFrame.
        # Ensure numerical values are cast to the correct types (int/float).
        # Categorical values are left as strings for one-hot encoding.
        processed_data = {
            'Age': int(user_data.get('age')),
            'Height': float(user_data.get('height')),
            'Weight': float(user_data.get('weight')),
            'BMI': float(user_data.get('bmi')),
            'Sex': user_data.get('sex'),
            'Hypertension': user_data.get('hypertension'),
            'Diabetes': user_data.get('diabetes'),
            'Level': user_data.get('level'),
            'Fitness Goal': user_data.get('fitness_goal'),
            'Fitness Type': user_data.get('fitness_type')
        }

        # Create a DataFrame from the single user input
        new_df = pd.DataFrame([processed_data])

        # Define categorical features as they were in your training script.
        # This list MUST be consistent with what you used in pd.get_dummies during model training.
        categorical_input_features = ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']

        # Apply one-hot encoding to the new input data.
        # `drop_first=True` is used to match the training data's encoding.
        new_df_encoded = pd.get_dummies(new_df, columns=categorical_input_features, drop_first=True)

        # Align columns with the training data's feature set.
        # This is CRUCIAL to ensure the input DataFrame for prediction has the exact
        # same columns in the exact same order as the data the model was trained on.
        # It adds any missing one-hot encoded columns (e.g., if 'Sex_Female' was not in input but in training)
        # and sets their value to 0. It also reorders the columns.
        final_input_df = pd.DataFrame(columns=loaded_feature_columns)
        for col in loaded_feature_columns:
            if col in new_df_encoded.columns:
                final_input_df[col] = new_df_encoded[col]
            else:
                final_input_df[col] = 0 # Fill missing one-hot columns with 0

        # Ensure correct data types for boolean columns (from get_dummies).
        # `pd.get_dummies` might create boolean columns, but scikit-learn models
        # often expect numerical inputs (0s and 1s).
        for col in final_input_df.columns:
            if final_input_df[col].dtype == 'bool':
                final_input_df[col] = final_input_df[col].astype(int)

        print("Processed input for prediction (first 5 columns for brevity):")
        print(final_input_df.head()) # Print head for inspection

        # --- Make Predictions using the loaded models ---
        predicted_output = {}
        for target_col in target_variables:
            model = loaded_models[target_col]
            encoder = loaded_label_encoders[target_col]

            # Make prediction for the current target variable
            pred_encoded = model.predict(final_input_df)[0]
            # Inverse transform the numerical prediction back to original text label
            pred_original = encoder.inverse_transform([pred_encoded])[0]
            predicted_output[target_col] = pred_original

        print(f"Prediction made: {predicted_output}")
        # Return the predictions as a JSON response
        return jsonify(predicted_output), 200

    except KeyError as e:
        # Handle cases where expected form fields are missing from the request
        return jsonify({"error": f"Missing data in request: '{e}'. Please ensure all required form fields are sent."}), 400
    except ValueError as e:
        # Handle cases where numerical inputs cannot be converted (e.g., non-numeric string for age)
        return jsonify({"error": f"Invalid data type or value: {e}. Please check numerical inputs (Age, Height, Weight, BMI)."}), 400
    except Exception as e:
        # Catch any other unexpected errors during the prediction process
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({"error": "An internal server error occurred during prediction. Please try again later."}), 500

# Entry point for running the Flask application
if __name__ == '__main__':
    # When running locally, use debug=True for automatic reloading and debugging.
    # For production deployment, set debug=False and use a production-ready WSGI server
    # like Gunicorn or uWSGI (e.g., `gunicorn -w 4 app:app`).
    # host='0.0.0.0' makes the server accessible from other devices on the network,
    # not just localhost.
    app.run(debug=True, host='0.0.0.0', port=5000)