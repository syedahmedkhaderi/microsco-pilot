"""
Test Physics-Based ML Models
=============================

This script tests the trained ML models to verify they're making
reasonable predictions based on the physics-validated training data.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.h5_data_loader import DatasetManager
from src.ml_predictor import MLPredictor
from src.feature_extractor import FeatureExtractor

def main():
    print("\n" + "="*70)
    print("🧪 TESTING PHYSICS-BASED ML MODELS")
    print("="*70)
    
    # Load predictor
    predictor = MLPredictor()
    
    if not predictor._models_trained():
        print("❌ No trained models found. Run train_ml_models.py first.")
        return
    
    print("✅ Loaded trained models")
    
    # Test on multiple regions
    test_file = 'data/AFM/PMN28Pt0000.h5'
    
    print(f"\n📄 Testing on: {os.path.basename(test_file)}")
    
    with DatasetManager(test_file) as dm:
        regions = dm.get_scan_regions(num_regions=3)
        
        print(f"   Extracted {len(regions)} test regions\n")
        
        for idx, region in enumerate(regions):
            print(f"Region {idx + 1}:")
            
            # Get prediction
            prediction = predictor.predict(region)
            
            print(f"   Speed:      {prediction['parameters']['speed']:.2f} µm/s")
            print(f"   Resolution: {prediction['parameters']['resolution']} px")
            print(f"   Force:      {prediction['parameters']['force']:.2f} nN")
            print(f"   Confidence: {prediction['overall_confidence']:.2f}")
            print(f"   Use ML:     {prediction['use_ml']}")
            
            # Show feature values
            features = prediction['features']
            print(f"   Features:")
            print(f"      Sharpness:   {features.get('sharpness', 0):.4f}")
            print(f"      Contrast:    {features.get('contrast', 0):.4f}")
            print(f"      Complexity:  {features.get('complexity', 0):.4f}")
            print()
    
    # Get feature importance
    print("\n📊 Feature Importance:")
    importance = predictor.get_feature_importance()
    
    for param_name, importances in importance.items():
        print(f"\n   {param_name.upper()}:")
        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for feat_name, imp_value in sorted_features[:5]:  # Top 5
            print(f"      {feat_name:20s}: {imp_value:.4f}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
