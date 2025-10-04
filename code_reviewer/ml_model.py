import pickle
import os
import sys

# Add the parent directory to the path so we can import the feature_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# CORRECTED IMPORT: Use the correct function name from our latest feature extractor
from .feature_extractor import extract_features_from_snippet

class RiskModel:
    """
    A wrapper class to load the trained model and its associated scaler,
    and provide a simple prediction method.
    """
    def __init__(self):
        model_path = os.path.join('models', 'risk_assessment_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
        
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
            self.model = payload['model']
            self.scaler = payload['scaler']
        
    def predict(self, code_snippet):
        """
        Takes a raw code snippet, extracts features, scales them,
        and returns a prediction from the trained model.
        """
        # 1. Extract features using the correct function
        features = extract_features_from_snippet(code_snippet)
        
        # 2. Scale the features using the loaded scaler
        # The model expects a 2D array, so we reshape
        scaled_features = self.scaler.transform([features])
        
        # 3. Make a prediction
        return self.model.predict(scaled_features)


