import joblib
import numpy as np
import os
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class HeartFailurePredictor:
    """Heart failure prediction service"""
    
    def __init__(self, model_path: str = "model.pkl", scaler_path: str = "scaler.pkl"):
        """
        Initialize the predictor with saved model and scaler
        
        Args:
            model_path: Path to the saved model file
            scaler_path: Path to the saved scaler file
        """
        self.model_path = self._find_file(model_path)
        self.scaler_path = self._find_file(scaler_path)
        self.model = None
        self.scaler = None
        self._load_model()
        self._load_scaler()
    
    def _find_file(self, filename: str) -> str:
        """Find file in common locations"""
        # Check current directory
        if os.path.exists(filename):
            return filename
        
        # Check parent directory (when running from app/)
        parent_path = Path(__file__).parent.parent / filename
        if parent_path.exists():
            return str(parent_path)
        
        # Check app directory
        app_path = Path(__file__).parent / filename
        if app_path.exists():
            return str(app_path)
        
        # Return original if not found (will raise error later)
        return filename
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_scaler(self):
        """Load the fitted scaler"""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded successfully from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
    
    def predict(self, features: dict) -> Tuple[int, float, str]:
        """
        Make a prediction based on input features
        
        Args:
            features: Dictionary containing all required features
            
        Returns:
            Tuple of (prediction, probability, message)
        """
        try:
            # Extract features in the correct order
            feature_order = [
                'age', 'anemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking'
            ]
            
            # Create numpy array with features in correct order
            input_array = np.array([[features[feature] for feature in feature_order]])
            
            # Scale the input
            scaled_input = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(scaled_input)[0]
            
            # Get prediction probability
            probability = self.model.predict_proba(scaled_input)[0][1]
            
            # Generate message
            if prediction == 1:
                message = f"High risk of death event detected (probability: {probability:.2%})"
            else:
                message = f"Low risk of death event (probability: {probability:.2%})"
            
            return int(prediction), float(probability), message
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")


# Initialize global predictor instance
predictor = None


def get_predictor() -> HeartFailurePredictor:
    """Get or create the predictor instance"""
    global predictor
    if predictor is None:
        predictor = HeartFailurePredictor()
    return predictor

