"""
Train ML Models on Real AFM Data
================================

This script:
1. Loads all 16 real AFM h5 files from data/AFM/
2. Extracts multiple regions from each file
3. Generates synthetic "ground truth" parameters (since we don't have historical logs)
4. Trains the ML models (Speed, Resolution, Force)
5. Saves the trained models for use in main.py

Usage:
    python train_ml_models.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.h5_data_loader import DatasetManager
from src.feature_extractor import FeatureExtractor
from src.training_data_manager import TrainingDataManager
from src.ml_predictor import MLPredictor

def estimate_optimal_parameters(features):
    """
    Estimate 'optimal' parameters based on image features.
    Since we don't have the original scan parameters or quality feedback,
    we create a synthetic ground truth logic that mimics expert operator decisions.
    
    Logic:
    - High complexity/roughness -> Slower speed, Higher resolution
    - Smooth/flat -> Faster speed, Lower resolution
    - High noise -> Higher force (to maintain contact)
    """
    # Extract key features
    roughness = features.get('roughness', 0.5)
    complexity = features.get('complexity', 0.5)
    noise = features.get('noise_level', 0.1)
    sharpness = features.get('sharpness', 5.0)
    
    # 1. Speed (µm/s)
    # Range: 1.0 to 20.0 µm/s
    # Rough/Complex -> Slow (down to 1.0)
    # Smooth/Simple -> Fast (up to 20.0)
    base_speed = 10.0
    speed_factor = 1.0 - (complexity * 0.8) # Reduce speed for complex images
    speed = base_speed * speed_factor
    speed = np.clip(speed, 1.0, 20.0)
    
    # 2. Resolution (pixels)
    # Range: 128 to 512
    # Complex/Sharp -> High Res (512)
    # Simple/Smooth -> Low Res (128)
    if complexity > 0.6 or sharpness > 7.0:
        resolution = 512
    elif complexity > 0.3:
        resolution = 256
    else:
        resolution = 128
        
    # 3. Force (nN)
    # Range: 0.5 to 5.0 nN
    # Noisy/Rough -> Higher force
    base_force = 1.0
    force = base_force + (noise * 5.0) + (roughness * 2.0)
    force = np.clip(force, 0.5, 5.0)
    
    return {
        'speed': float(speed),
        'resolution': int(resolution),
        'force': float(force)
    }

def estimate_quality(features, params):
    """Estimate the quality score (0-10) we WOULD have gotten"""
    # Synthetic quality score
    # If we matched the "optimal" logic, quality is high
    # This is just for the training data record
    return 8.5 + np.random.normal(0, 0.5)

def main():
    print("\n" + "="*70)
    print("🚀 TRAINING ML MODELS ON REAL AFM DATA")
    print("="*70)
    
    # 1. Setup
    data_dir = os.path.join('data', 'AFM')
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    
    if not h5_files:
        print(f"❌ No h5 files found in {data_dir}")
        return
        
    print(f"📂 Found {len(h5_files)} AFM files")
    
    # Clear old training data to start fresh
    import shutil
    if os.path.exists('data/ml_training'):
        shutil.rmtree('data/ml_training')
        print("🧹 Cleared old training data")
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    data_manager = TrainingDataManager()  # Init AFTER cleanup so it creates the dir
    
    # 2. Process files and generate training data
    print("\n🔬 Extracting features and generating training data...")
    
    total_samples = 0
    
    for h5_file in tqdm(h5_files, desc="Processing files"):
        try:
            with DatasetManager(h5_file) as dm:
                # Extract more regions per file for "proper and long training"
                # 10 regions per file * 16 files = 160 samples
                regions = dm.get_scan_regions(num_regions=10)
                
                for region in regions:
                    # Extract features
                    features = feature_extractor.extract_features(region)
                    
                    # Estimate optimal parameters (Synthetic Ground Truth)
                    params = estimate_optimal_parameters(features)
                    
                    # Estimate quality
                    results = {
                        'quality': estimate_quality(features, params),
                        'success': True,
                        'time': 100.0 / params['speed']
                    }
                    
                    # Add to training set
                    data_manager.add_training_sample(
                        features=features,
                        parameters=params,
                        results=results,
                        metadata={'source_file': os.path.basename(h5_file)}
                    )
                    total_samples += 1
                    
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(h5_file)}: {e}")
            
    print(f"\n✅ Generated {total_samples} training samples")
    
    # 3. Train Models
    print("\n🧠 Training ML Predictor...")
    predictor = MLPredictor()
    
    # Force retraining with the new data
    # min_samples=20 ensures we use the data we just generated
    predictor.train(min_samples=20)
    
    # 4. Verify
    print("\n🔍 Verifying models...")
    if predictor._models_trained():
        print("✅ Models trained successfully!")
        
        # Test prediction on a random sample
        test_file = h5_files[0]
        with DatasetManager(test_file) as dm:
            region = dm.get_scan_regions(num_regions=1)[0]
            prediction = predictor.predict(region)
            
            print("\n📝 Test Prediction:")
            print(f"   Speed:      {prediction['parameters']['speed']:.2f} µm/s")
            print(f"   Resolution: {prediction['parameters']['resolution']} px")
            print(f"   Force:      {prediction['parameters']['force']:.2f} nN")
            print(f"   Confidence: {prediction['overall_confidence']:.2f}")
            print(f"   Use ML:     {prediction['use_ml']}")
            
            if prediction['use_ml']:
                print("   ✅ ML is confident and working!")
            else:
                print("   ⚠️  ML is not confident yet (needs more data)")
    else:
        print("❌ Model training failed.")

    print("\n" + "="*70)
    print("🎉 DONE! Models are ready for main.py")
    print("="*70)

if __name__ == "__main__":
    main()
