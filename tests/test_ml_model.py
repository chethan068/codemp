import sys
import os
import pytest
import pickle
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch

# Add the parent directory to the path so we can import the ml_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code_reviewer.ml_model import RiskModel

@pytest.fixture
def mock_model_file(tmp_path):
    """
    A pytest fixture to create a temporary, fake trained model file for testing.
    This avoids needing the real, large model file to run tests.
    """
    # Create a simple, dummy classifier
    dummy_model = RandomForestClassifier(n_estimators=1)
    # Fit it with some dummy data so it's a valid "trained" model
    dummy_model.fit([[0, 0, 0]], [0])
    
    # Define the path for the temporary model file
    model_path = tmp_path / "test_model.pkl"
    
    # Save the dummy model to the file
    with open(model_path, 'wb') as f:
        pickle.dump(dummy_model, f)
        
    return str(model_path)

def test_risk_model_initialization(mock_model_file):
    """
    Tests that the RiskModel class can be initialized and loads a model successfully.
    """
    # Use patch to override the default MODEL_FILE_PATH with our temporary mock model
    with patch('code_reviewer.ml_model.MODEL_FILE_PATH', mock_model_file):
        risk_model_instance = RiskModel()
        assert risk_model_instance.model is not None
        assert isinstance(risk_model_instance.model, RandomForestClassifier)

def test_risk_model_prediction(mock_model_file):
    """
    Tests that the predict method returns a valid prediction (0 or 1).
    """
    with patch('code_reviewer.ml_model.MODEL_FILE_PATH', mock_model_file):
        risk_model_instance = RiskModel()
        
        # A simple Java code snippet for prediction
        java_code = "class Test { void method() {} }"
        
        prediction = risk_model_instance.predict(java_code)
        
        # The prediction should be either 0 (low-risk) or 1 (high-risk)
        assert prediction in [0, 1]

def test_risk_model_file_not_found():
    """
    Tests that initializing RiskModel with a non-existent path gracefully sets the model to None.
    """
    # Patch the model path to point to a file that doesn't exist
    with patch('code_reviewer.ml_model.MODEL_FILE_PATH', 'non_existent_model.pkl'):
        risk_model_instance = RiskModel()
        assert risk_model_instance.model is None
