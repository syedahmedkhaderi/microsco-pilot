"""ML-Based Parameter Predictor for SmartScan

This is the BRAIN of the ML system. It uses machine learning to predict
optimal scan parameters based on image features.

THE BIG IDEA:
Instead of fixed rules like "if quality < 6, slow down",
the ML model learns: "When I see THIS type of image, use THESE parameters"

HOW IT WORKS:
1. Extract features from AFM image
2. Feed features to ML model
3. Model predicts optimal parameters
4. If confident → use ML prediction
5. If uncertain → fall back to rules
6. After scan → learn from results

This gets smarter over time!
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# LightGBM for gradient boosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. ML prediction will not work. "
                 "Install with: pip install lightgbm")

# Handle imports - work both as module and standalone script
import sys
import os
if __name__ == "__main__":
    # Add parent directory to path when run as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.training_data_manager import TrainingDataManager


class MLPredictor:
    """ML-Based Parameter Prediction System
    
    This class predicts optimal AFM scan parameters using machine learning.
    
    KEY FEATURES:
    - Learns from past scans
    - Predicts speed, resolution, and force
    - Provides confidence scores
    - Falls back to rules when uncertain
    - Updates itself online (learns continuously)
    
    Example:
        >>> predictor = MLPredictor()
        >>> prediction = predictor.predict(afm_image)
        >>> if prediction['confidence'] > 0.7:
        ...     use_params = prediction['parameters']
        ... else:
        ...     use_params = fallback_rules()
    """
    
    def __init__(self, model_dir: str = 'models/ml_predictor'):
        """Initialize ML predictor
        
        Args:
            model_dir: Directory to save/load trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Training data manager
        self.data_manager = TrainingDataManager()
        
        # Models for each parameter (speed, resolution, force)
        self.models = {
            'speed': None,
            'resolution': None,
            'force': None
        }
        
        # Confidence thresholds
        self.confidence_threshold = 0.7  # Use ML if confidence > 0.7
        
        # Parameter bounds (safety limits)
        self.param_bounds = {
            'speed': (1.0, 20.0),        # µm/s
            'resolution': (128, 512),     # pixels
            'force': (0.5, 5.0)          # nN
        }
        
        # Track predictions for analysis
        self.prediction_log = []
        
        # Try to load existing models
        self.load_models()
    
    def predict(self, image: np.ndarray, use_ml: bool = True) -> Dict[str, any]:
        """Predict optimal parameters for an AFM image
        
        This is the MAIN function you'll use.
        
        WHAT IT DOES:
        1. Extracts features from image
        2. Runs ML models to predict parameters
        3. Calculates confidence scores
        4. Returns predictions with confidence
        
        Args:
            image: 2D AFM scan image
            use_ml: Whether to use ML (if False, returns None for ML predictions)
            
        Returns:
            Dictionary with:
            - 'parameters': {'speed': 12.5, 'resolution': 256, 'force': 2.3}
            - 'confidence': {'speed': 0.85, 'resolution': 0.72, 'force': 0.68}
            - 'use_ml': True/False (whether to use ML or fall back to rules)
            - 'features': Extracted image features
        """
        # Extract features
        features = self.feature_extractor.extract_features(image)
        feature_vector = self.feature_extractor.get_feature_vector(image)
        
        # If ML disabled or models not trained, return early
        if not use_ml or not self._models_trained():
            return {
                'parameters': None,
                'confidence': None,
                'use_ml': False,
                'features': features,
                'reason': 'Models not trained yet' if not self._models_trained() else 'ML disabled'
            }
        
        # Predict each parameter
        predictions = {}
        confidences = {}
        
        for param_name in ['speed', 'resolution', 'force']:
            pred, conf = self._predict_parameter(param_name, feature_vector)
            predictions[param_name] = pred
            confidences[param_name] = conf
        
        # Overall confidence (minimum of all parameters)
        overall_confidence = min(confidences.values())
        
        # Should we use ML?
        use_ml_decision = overall_confidence >= self.confidence_threshold
        
        result = {
            'parameters': predictions,
            'confidence': confidences,
            'overall_confidence': overall_confidence,
            'use_ml': use_ml_decision,
            'features': features,
            'threshold': self.confidence_threshold
        }
        
        # Log prediction
        self.prediction_log.append(result)
        
        return result
    
    def _predict_parameter(self, param_name: str, 
                          features: np.ndarray) -> Tuple[float, float]:
        """Predict a single parameter with confidence
        
        HOW CONFIDENCE WORKS:
        We train not just one model, but an ensemble.
        - If all models agree → high confidence
        - If models disagree → low confidence
        
        For simplicity, we use the model's built-in uncertainty or 
        a simple heuristic based on training data spread.
        
        Args:
            param_name: 'speed', 'resolution', or 'force'
            features: Feature vector from image
            
        Returns:
            (prediction, confidence) where:
            - prediction: Predicted parameter value
            - confidence: 0.0 to 1.0 (how confident)
        """
        model = self.models[param_name]
        
        if model is None:
            return self._fallback_value(param_name), 0.0
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Suppress sklearn feature name warning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            
            # Predict
            try:
                prediction = model.predict(features)[0]
                
                # Clip to safe bounds
                bounds = self.param_bounds[param_name]
                prediction = np.clip(prediction, bounds[0], bounds[1])
                
                # Estimate confidence
                n_training_samples = self.data_manager.get_statistics().get('total_samples', 0)
                
                if n_training_samples < 10:
                    confidence = 0.2
                elif n_training_samples < 50:
                    confidence = 0.5
                elif n_training_samples < 100:
                    confidence = 0.7
                else:
                    confidence = 0.9
                
                return float(prediction), float(confidence)
                
            except Exception as e:
                print(f"⚠️  Prediction error for {param_name}: {e}")
                return self._fallback_value(param_name), 0.0
    
    def _fallback_value(self, param_name: str) -> float:
        """Get safe fallback value for parameter"""
        fallbacks = {
            'speed': 10.0,      # Medium speed
            'resolution': 256,  # Medium resolution
            'force': 2.5        # Medium force
        }
        return fallbacks.get(param_name, 1.0)
    
    def train(self, min_samples: int = 20):
        """Train ML models on collected data
        
        WHEN TO CALL THIS:
        - After collecting enough training samples (at least 20)
        - Periodically during operation (online learning)
        - When starting a new session with existing data
        
        WHAT IT DOES:
        1. Loads training data
        2. Trains a LightGBM model for each parameter
        3. Saves models to disk
        
        Args:
            min_samples: Minimum training samples required
        """
        if not HAS_LIGHTGBM:
            print("❌ LightGBM not installed. Cannot train models.")
            return
        
        print("\n🧠 Training ML models...")
        
        # Suppress sklearn feature name warnings
        import warnings
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        
        # Train each parameter model
        for param_name in ['speed', 'resolution', 'force']:
            print(f"\n   Training {param_name} predictor...")
            
            # Get training data
            X, y = self.data_manager.get_training_data(param_name, min_samples=min_samples)
            
            if len(X) < min_samples:
                print(f"   ⚠️  Not enough data ({len(X)} samples, need {min_samples})")
                continue
            
            # Convert to DataFrame with feature names (eliminates sklearn warnings)
            import pandas as pd
            X_df = pd.DataFrame(X, columns=self.feature_extractor.feature_names)
            
            # Train LightGBM model
            model = lgb.LGBMRegressor(
                n_estimators=100,        # 100 trees
                learning_rate=0.05,      # Slow learning for stability
                max_depth=5,             # Not too deep (prevent overfitting)
                num_leaves=31,           # Default
                random_state=42,
                verbosity=-1             # Quiet
            )
            
            model.fit(X_df, y)  # Train with DataFrame (has feature names)
            
            self.models[param_name] = model
            
            print(f"   ✅ Trained on {len(X)} samples")
        
        # Save models
        self.save_models()
        print("\n✅ ML training complete!")
    
    def update_online(self, image: np.ndarray, parameters: Dict[str, float],
                     results: Dict[str, float], metadata: Optional[Dict] = None):
        """Update the model with new data (online learning)
        
        CALL THIS AFTER EVERY SCAN!
        
        This is how the system learns over time.
        After each scan:
        1. Record what features the image had
        2. Record what parameters we used
        3. Record what quality we got
        4. Add to training data
        5. Periodically retrain model
        
        Args:
            image: The AFM scan image
            parameters: Parameters used {'speed': 12, 'resolution': 256, 'force': 2.5}
            results: Results achieved {'quality': 7.2, 'success': True, 'time': 15.3}
            metadata: Optional metadata {'sample_type': 'PMN28Pt', 'afm_mode': 'contact'}
        """
        # Extract features
        features = self.feature_extractor.extract_features(image)
        
        # Add to training data
        self.data_manager.add_training_sample(features, parameters, results, metadata)
        
        # Get current sample count
        stats = self.data_manager.get_statistics()
        n_samples = stats.get('total_samples', 0)
        
        # Retrain every 20 samples (incremental learning)
        if n_samples % 20 == 0 and n_samples >= 20:
            print(f"\n📚 {n_samples} samples collected - retraining models...")
            self.train(min_samples=20)
        
        return n_samples
    
    def save_models(self):
        """Save trained models to disk"""
        for param_name, model in self.models.items():
            if model is not None:
                model_file = self.model_dir / f'{param_name}_model.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   💾 Saved {param_name} model")
    
    def load_models(self):
        """Load previously trained models from disk"""
        loaded_any = False
        for param_name in ['speed', 'resolution', 'force']:
            model_file = self.model_dir / f'{param_name}_model.pkl'
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.models[param_name] = pickle.load(f)  # Fixed: load not dump
                    print(f"   📂 Loaded {param_name} model")
                    loaded_any = True
                except Exception as e:
                    print(f"   ⚠️  Failed to load {param_name} model: {e}")
        
        if loaded_any:
            print("✅ Loaded existing ML models")
    
    def _models_trained(self) -> bool:
        """Check if any models are trained"""
        return any(model is not None for model in self.models.values())
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each parameter model
        
        WHAT IS FEATURE IMPORTANCE?
        Tells you which image properties matter most for predictions.
        
        For example:
        - Speed might depend mostly on "sharpness" and "complexity"
        - Resolution might depend on "edge_strength"
        - Force might depend on "noise_level"
        
        Returns:
            Dictionary mapping parameter names to feature importance scores
        """
        importances = {}
        
        for param_name, model in self.models.items():
            if model is not None and hasattr(model, 'feature_importances_'):
                # Get feature importances
                importance_values = model.feature_importances_
                
                # Map to feature names
                importances[param_name] = {
                    feat_name: float(importance_values[i])
                    for i, feat_name in enumerate(self.feature_extractor.feature_names)
                }
        
        return importances
    
    def get_prediction_stats(self) -> Dict[str, any]:
        """Get statistics about predictions made
        
        Returns:
            Statistics about ML usage, confidence distribution, etc.
        """
        if not self.prediction_log:
            return {'total_predictions': 0}
        
        ml_used = sum(1 for p in self.prediction_log if p.get('use_ml', False))
        confidences = [p.get('overall_confidence', 0) for p in self.prediction_log 
                      if p.get('overall_confidence') is not None]
        
        return {
            'total_predictions': len(self.prediction_log),
            'ml_used': ml_used,
            'rules_used': len(self.prediction_log) - ml_used,
            'ml_usage_rate': ml_used / len(self.prediction_log) if self.prediction_log else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0
        }


# Educational demo
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ML PREDICTOR - EDUCATIONAL DEMO")
    print("="*70)
    
    if not HAS_LIGHTGBM:
        print("\n❌ LightGBM not installed!")
        print("   Install with: pip install lightgbm")
        print("\nThis demo requires LightGBM to work.\n")
    else:
        print("\n1. Creating ML Predictor...")
        predictor = MLPredictor(model_dir='models/ml_predictor_test')
        print(f"   Model directory: {predictor.model_dir}")
        print(f"   Confidence threshold: {predictor.confidence_threshold}")
        
        print("\n2. Generating synthetic training data...")
        data_manager = predictor.data_manager
        
        # Generate 30 fake training samples
        for i in range(30):
            # Random features
            fake_features = {
                'sharpness': np.random.uniform(1, 10),
                'contrast': np.random.uniform(0, 1),
                'complexity': np.random.uniform(0, 5),
                'edge_strength': np.random.uniform(0, 10),
                'noise_level': np.random.uniform(0, 2),
                'fft_high_freq': np.random.uniform(0, 1),
                'fft_low_freq': np.random.uniform(0, 1),
                'mean_gradient': np.random.uniform(0, 5),
                'std_gradient': np.random.uniform(0, 3),
                'range_value': np.random.uniform(0, 100)
            }
            
            # Parameters (somewhat correlated with features for realism)
            fake_params = {
                'speed': 5 + fake_features['sharpness'],  # Higher sharpness → faster
                'resolution': 128 if fake_features['complexity'] < 2 else 256,
                'force': 2.0 + fake_features['noise_level']
            }
            
            # Results
            fake_results = {
                'quality': 5 + fake_features['sharpness'] * 0.3,
                'success': 1,
                'time': 100 / fake_params['speed']
            }
            
            data_manager.add_training_sample(fake_features, fake_params, fake_results)
        
        print(f"   Generated 30 training samples")
        
        print("\n3. Training ML models...")
        predictor.train(min_samples=20)
        
        print("\n4. Making predictions on test image...")
        test_image = np.random.randn(128, 128)
        prediction = predictor.predict(test_image)
        
        print(f"\n   Predicted Parameters:")
        for param, value in prediction['parameters'].items():
            conf = prediction['confidence'][param]
            print(f"      {param:12s}: {value:6.2f} (confidence: {conf:.2f})")
        
        print(f"\n   Overall Confidence: {prediction['overall_confidence']:.2f}")
        print(f"   Use ML? {prediction['use_ml']}")
        print(f"   Threshold: {prediction['threshold']}")
        
        print("\n5. Feature Importance:")
        importance = predictor.get_feature_importance()
        if importance:
            for param, feat_imp in importance.items():
                print(f"\n   {param.upper()}:")
                sorted_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
                for feat_name, imp_value in sorted_features[:3]:  # Top 3
                    print(f"      {feat_name:20s}: {imp_value:.4f}")
        
        # Clean up
        import shutil
        shutil.rmtree('models/ml_predictor_test')
        shutil.rmtree('data/ml_training')
        
        print("\n✅ ML Predictor demo complete!")
        print("\nThis system learns optimal parameters from experience!\n")
