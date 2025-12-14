"""Train ML Models on Real AFM Data

This script trains the ML parameter prediction models using your real AFM h5 files.

WHAT IT DOES:
1. Loads all AFM h5 files from data/AFM/
2. Extracts features from each scan
3. Creates training data with reasonable parameter assumptions
4. Trains ML models
5. Saves models for future use

NOTE: Since we don't have the actual parameters that were used for your scans,
we'll use reasonable defaults based on typical AFM settings. As you use SmartScan
and it learns from actual scan results, the models will improve.
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.h5_data_loader import DatasetManager
from src.feature_extractor import FeatureExtractor
from src.training_data_manager import TrainingDataManager
from src.ml_predictor import MLPredictor


def generate_training_data_from_afm_files(afm_dir='data/AFM'):
    """Generate training data from real AFM h5 files
    
    Args:
        afm_dir: Directory containing AFM h5 files
    """
    print("\n" + "="*70)
    print("TRAINING ML MODELS ON REAL AFM DATA")
    print("="*70)
    
    # Find all h5 files
    h5_files = sorted(glob.glob(os.path.join(afm_dir, '*.h5')))
    
    if not h5_files:
        print(f"\n❌ No h5 files found in {afm_dir}")
        return
    
    print(f"\n📂 Found {len(h5_files)} AFM files")
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    data_manager = TrainingDataManager()
    
    # Clear existing data (fresh start)
    print("\n🗑️  Clearing old training data...")
    data_manager.clear_data()
    
    # Process each AFM file
    print(f"\n🔬 Extracting features from AFM files...")
    
    samples_created = 0
    
    for i, h5_file in enumerate(h5_files):
        filename = os.path.basename(h5_file)
        print(f"\n   [{i+1}/{len(h5_files)}] {filename}")
        
        try:
            # Load AFM file
            with DatasetManager(h5_file) as dm:
                # Get metadata
                metadata = dm.get_metadata()
                
                # Extract regions (we'll create training samples from each)
                regions = dm.get_scan_regions(num_regions=5)  # 5 regions per file
                
                print(f"      Extracted {len(regions)} regions")
                
                for j, region in enumerate(regions):
                    # Extract features
                    features = feature_extractor.extract_features(region)
                    
                    # Estimate reasonable parameters based on features
                    # (In future, these will come from actual scan parameters)
                    parameters = estimate_parameters_from_features(features)
                    
                    # Estimate quality based on features
                    # (In future, this will be actual measured quality)
                    quality = estimate_quality_from_features(features)
                    
                    # Create results
                    results = {
                        'quality': quality,
                        'success': 1 if quality > 5 else 0,
                        'time': 100 / parameters['speed']  # Faster speed = less time
                    }
                    
                    # Get sample type from filename
                    sample_type = 'unknown'
                    if 'PMN' in filename:
                        sample_type = 'PMN28Pt'
                    elif 'PTO' in filename:
                        sample_type = 'PTO_Funakubo'
                    elif 'PZT' in filename:
                        sample_type = 'PZT'
                    
                    # Add to training data
                    data_manager.add_training_sample(
                        features=features,
                        parameters=parameters,
                        results=results,
                        metadata={
                            'sample_type': sample_type,
                            'afm_mode': metadata.get('afm_mode', 'contact')
                        }
                    )
                    
                    samples_created += 1
                
                print(f"      ✅ Created {len(regions)} training samples")
        
        except Exception as e:
            print(f"      ⚠️  Error processing {filename}: {e}")
            continue
    
    print(f"\n✅ Created {samples_created} total training samples")
    
    # Show statistics
    stats = data_manager.get_statistics()
    print(f"\n📊 Training Data Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Average quality: {stats['quality_stats'].get('mean', 0):.2f}")
    
    print(f"\n💾 Training data saved to: {data_manager.data_file}")
    
    return samples_created


def estimate_parameters_from_features(features):
    """Estimate reasonable scan parameters based on image features
    
    This creates plausible parameter estimates until we have real data.
    The ML model will learn the actual optimal mappings over time.
    
    Args:
        features: Feature dictionary
        
    Returns:
        Parameter dictionary with speed, resolution, force
    """
    # Speed: Based on sharpness and complexity
    # High sharpness → can go faster
    # High complexity → need to go slower
    base_speed = 10.0
    sharpness_factor = features['sharpness'] / 5.0  # Normalize
    complexity_penalty = features['complexity'] / 2.0
    
    speed = base_speed + sharpness_factor - complexity_penalty
    speed = np.clip(speed, 5.0, 15.0)  # Keep in reasonable range
    
    # Resolution: Based on complexity and edge strength
    # High complexity → need higher resolution
    if features['complexity'] > 2.0 or features['edge_strength'] > 5.0:
        resolution = 512
    elif features['complexity'] > 1.0:
        resolution = 256
    else:
        resolution = 128
    
    # Force: Based on noise level
    # Higher noise might need more force for stability
    base_force = 2.5
    noise_adjustment = features['noise_level'] * 0.5
    force = base_force + noise_adjustment
    force = np.clip(force, 1.5, 3.5)
    
    return {
        'speed': float(speed),
        'resolution': float(resolution),
        'force': float(force)
    }


def estimate_quality_from_features(features):
    """Estimate scan quality from features
    
    Better features (high sharpness, good contrast, low noise) → higher quality
    
    Args:
        features: Feature dictionary
        
    Returns:
        Quality score (0-10)
    """
    # Weighted combination of features
    quality = (
        features['sharpness'] * 0.3 +
        features['contrast'] * 10.0 +  # Contrast is 0-1, scale up
        (10 - features['noise_level']) * 0.2 +  # Lower noise = better
        features['edge_strength'] * 0.2
    )
    
    # Normalize to 0-10 range
    quality = np.clip(quality, 0, 10)
    
    return float(quality)


def train_models_on_data():
    """Train ML models on the collected data"""
    print("\n" + "="*70)
    print("TRAINING ML MODELS")
    print("="*70)
    
    # Create predictor (this loads the data manager)
    predictor = MLPredictor()
    
    # Train models
    print("\n🧠 Training models...")
    predictor.train(min_samples=10)
    
    print("\n✅ Models trained and saved!")
    print(f"   Model directory: {predictor.model_dir}")
    print(f"   - speed_model.pkl")
    print(f"   - resolution_model.pkl")
    print(f"   - force_model.pkl")
    
    # Show feature importance
    print("\n📊 Feature Importance (what matters most):")
    importance = predictor.get_feature_importance()
    
    for param_name, feat_importance in importance.items():
        print(f"\n   {param_name.upper()}:")
        sorted_features = sorted(feat_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        
        for feat_name, imp_value in sorted_features[:5]:  # Top 5
            if imp_value > 0:
                bar = "█" * int(imp_value * 50)  # Visual bar
                print(f"      {feat_name:20s}: {bar} {imp_value:.4f}")
    
    return predictor


def test_predictions(predictor):
    """Test the trained models with a sample AFM file"""
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    # Load a test file
    test_file = 'data/AFM/PMN28Pt0000.h5'
    
    if not os.path.exists(test_file):
        print(f"\n⚠️  Test file not found: {test_file}")
        return
    
    print(f"\n🧪 Testing with: {os.path.basename(test_file)}")
    
    with DatasetManager(test_file) as dm:
        # Get a region
        regions = dm.get_scan_regions(num_regions=1)
        test_region = regions[0]
        
        # Make prediction
        prediction = predictor.predict(test_region)
        
        print(f"\n🤖 ML Prediction:")
        print(f"   Speed:      {prediction['parameters']['speed']:.1f} µm/s")
        print(f"   Resolution: {prediction['parameters']['resolution']:.0f} px")
        print(f"   Force:      {prediction['parameters']['force']:.1f} nN")
        
        print(f"\n   Confidence:")
        for param, conf in prediction['confidence'].items():
            print(f"      {param:12s}: {conf:.2f}")
        
        print(f"\n   Overall Confidence: {prediction['overall_confidence']:.2f}")
        print(f"   Use ML? {prediction['use_ml']}")
        
        if prediction['use_ml']:
            print("\n   ✅ ML model is confident - would use these parameters!")
        else:
            print(f"\n   ⚠️  ML not confident yet (< {prediction['threshold']}) - would use rules")


def main():
    """Main training workflow"""
    print("\n" + "="*70)
    print("🧠 ML MODEL TRAINING SCRIPT")
    print("="*70)
    
    print("\nThis script will:")
    print("  1. Load all your AFM h5 files from data/AFM/")
    print("  2. Extract features from each scan")
    print("  3. Generate training data")
    print("  4. Train ML models")
    print("  5. Save models for SmartScan to use")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Generate training data
    n_samples = generate_training_data_from_afm_files('data/AFM')
    
    if n_samples == 0:
        print("\n❌ No training samples created. Exiting.")
        return
    
    # Step 2: Train models
    predictor = train_models_on_data()
    
    # Step 3: Test predictions
    test_predictions(predictor)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print("\nWhat was created:")
    print(f"  📊 Training data: data/ml_training/training_data.csv")
    print(f"  🧠 ML models: models/ml_predictor/")
    print(f"     - speed_model.pkl")
    print(f"     - resolution_model.pkl")
    print(f"     - force_model.pkl")
    
    print("\nNext steps:")
    print("  1. The models are now ready to use in SmartScan")
    print("  2. As you scan more samples, SmartScan will learn and improve")
    print("  3. Models automatically retrain every 20 scans (online learning)")
    
    print("\nThe ML system will get smarter with every scan! 🚀\n")


if __name__ == "__main__":
    main()
