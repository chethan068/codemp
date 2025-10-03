import pickle
import os
from .feature_extractor import extract_features
import numpy as np

class RiskModel:
    """
    A wrapper class to load the trained scikit-learn model and make predictions.
    """
    def __init__(self, model_path='models/risk_assessment_model.pkl'):
        self.model = None
        # Adjust the path to be relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        full_model_path = os.path.join(project_root, model_path)
        
        try:
            with open(full_model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Risk assessment model loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Model file not found at '{full_model_path}'.")
            print("Please run 'scripts/train_model.py' to create the model.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    def predict(self, code_string):
        """
        Takes a raw code string, extracts features, and returns a risk prediction.
        Returns 1 for high-risk, 0 for low-risk.
        """
        if not self.model:
            print("Error: Model is not loaded, cannot make a prediction.")
            return 0 # Default to low-risk if model isn't available

        # 1. Extract features from the code
        features = extract_features(code_string)
        
        # 2. Reshape features for a single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # 3. Make a prediction
        prediction = self.model.predict(features_array)
        
        return prediction[0]

# Create a single, reusable instance of the model for the app to import
risk_model = RiskModel()

