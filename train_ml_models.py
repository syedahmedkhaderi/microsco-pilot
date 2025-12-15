"""
Train ML Models on Real AFM Data Using Physics Simulation
==========================================================

This script:
1. Loads all 16 real AFM h5 files from data/AFM/
2. Extracts multiple regions from each file as ground truth topography
3. Uses DTMicroscope physics simulator to scan each region with different parameter combinations
4. Calculates TRUE quality from tracking error: quality = 10 * exp(-error)
5. Trains ML models on physics-validated data
6. Saves trained models for use in main.py

Parameter Grid (Optimized):
- Speed: [2, 10, 15] µm/s
- Resolution: [128, 512] pixels  
- Force: [1.0, 3.0] nN
- Total: 3 × 2 × 2 = 12 combinations per region

Expected: 16 files × 5 regions × 12 params = 960 training samples

Usage:
    python train_ml_models.py
"""

import os
import sys
import glob
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.h5_data_loader import DatasetManager
from src.feature_extractor import FeatureExtractor
from src.training_data_manager import TrainingDataManager
from src.ml_predictor import MLPredictor
from src.dtm_controller import AdaptiveAFMController


# Parameter grid (optimized for speed)
PARAM_GRID = {
    'speed': [2, 10, 15],      # µm/s
    'resolution': [128, 512],   # pixels
    'force': [1.0, 3.0]        # nN
}


def generate_param_combinations():
    """Generate all parameter combinations from grid"""
    combinations = []
    for speed in PARAM_GRID['speed']:
        for resolution in PARAM_GRID['resolution']:
            for force in PARAM_GRID['force']:
                combinations.append({
                    'speed': speed,
                    'resolution': resolution,
                    'force': force
                })
    return combinations


def scan_with_params(ground_truth, params, controller):
    """
    Scan ground truth with specific parameters using physics simulation.
    
    Args:
        ground_truth: numpy array of true topography
        params: dict with speed, resolution, force
        controller: AdaptiveAFMController instance
        
    Returns:
        dict with quality_metrics or None if failed
    """
    try:
        result = controller.scan_ground_truth(ground_truth, params)
        return result['quality_metrics']
    except Exception as e:
        print(f"⚠️  Scan failed for params {params}: {e}")
        return None


def process_region(region, region_idx, file_name, param_combinations):
    """
    Process a single region with all parameter combinations.
    This function is designed to be called in parallel.
    
    Args:
        region: numpy array of ground truth topography
        region_idx: region index
        file_name: source file name
        param_combinations: list of parameter dicts
        
    Returns:
        list of training samples
    """
    samples = []
    
    # Create controller for this process
    controller = AdaptiveAFMController()
    feature_extractor = FeatureExtractor()
    
    # Extract features from ground truth (not scanned image)
    features = feature_extractor.extract_features(region)
    
    # Scan with each parameter combination
    for param_idx, params in enumerate(param_combinations):
        quality_metrics = scan_with_params(region, params, controller)
        
        if quality_metrics is not None:
            # Create training sample
            sample = {
                'features': features,
                'parameters': params,
                'results': {
                    'quality': quality_metrics['quality'],
                    'tracking_error': quality_metrics.get('tracking_error', 0.5),
                    'snr': quality_metrics.get('snr', 10.0),
                    'success': True
                },
                'metadata': {
                    'source_file': file_name,
                    'region_idx': region_idx,
                    'param_idx': param_idx,
                    'simulation_mode': quality_metrics.get('source', 'unknown')
                }
            }
            samples.append(sample)
    
    return samples


def main():
    print("\n" + "="*70)
    print("🚀 TRAINING ML MODELS WITH PHYSICS-BASED SIMULATION")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Setup
    data_dir = os.path.join('data', 'AFM')
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    
    if not h5_files:
        print(f"❌ No h5 files found in {data_dir}")
        return
        
    print(f"📂 Found {len(h5_files)} AFM files")
    
    # Generate parameter combinations
    param_combinations = generate_param_combinations()
    print(f"🔧 Parameter grid: {len(param_combinations)} combinations")
    print(f"   Speed: {PARAM_GRID['speed']} µm/s")
    print(f"   Resolution: {PARAM_GRID['resolution']} px")
    print(f"   Force: {PARAM_GRID['force']} nN")
    
    # Clear old training data to start fresh
    import shutil
    if os.path.exists('data/ml_training'):
        shutil.rmtree('data/ml_training')
        print("🧹 Cleared old training data")
    
    # Initialize components
    data_manager = TrainingDataManager()
    
    # 2. Process files and generate training data
    print("\n🔬 Generating physics-based training data...")
    print(f"   Expected samples: {len(h5_files)} files × 5 regions × {len(param_combinations)} params = {len(h5_files) * 5 * len(param_combinations)}")
    
    total_samples = 0
    all_samples = []
    
    # Process each file
    for file_idx, h5_file in enumerate(h5_files):
        file_name = os.path.basename(h5_file)
        print(f"\n📄 Processing {file_name} ({file_idx + 1}/{len(h5_files)})...")
        
        try:
            with DatasetManager(h5_file) as dm:
                # Extract 5 regions per file
                regions = dm.get_scan_regions(num_regions=5)
                
                print(f"   Extracted {len(regions)} regions")
                
                # Process regions sequentially (parallelization happens within each region)
                # This avoids memory issues from loading too many controllers
                for region_idx, region in enumerate(tqdm(regions, desc=f"   Regions", leave=False)):
                    # Process this region with all parameter combinations
                    samples = process_region(region, region_idx, file_name, param_combinations)
                    
                    # Add samples to training set
                    for sample in samples:
                        data_manager.add_training_sample(
                            features=sample['features'],
                            parameters=sample['parameters'],
                            results=sample['results'],
                            metadata=sample['metadata']
                        )
                        total_samples += 1
                    
                    all_samples.extend(samples)
                
        except Exception as e:
            print(f"⚠️  Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Generated {total_samples} training samples in {elapsed:.1f}s")
    
    # 3. Analyze quality distribution
    if all_samples:
        qualities = [s['results']['quality'] for s in all_samples]
        print(f"\n📊 Quality Distribution:")
        print(f"   Mean: {np.mean(qualities):.2f}")
        print(f"   Std:  {np.std(qualities):.2f}")
        print(f"   Min:  {np.min(qualities):.2f}")
        print(f"   Max:  {np.max(qualities):.2f}")
        
        # Check if we have physics data
        physics_samples = [s for s in all_samples if s['metadata'].get('simulation_mode') == 'DTMicroscope']
        if physics_samples:
            print(f"   ✅ Physics-based samples: {len(physics_samples)}/{len(all_samples)} ({100*len(physics_samples)/len(all_samples):.1f}%)")
        else:
            print(f"   ⚠️  No DTMicroscope samples - using fallback mode")
    
    # 4. Train Models
    print("\n🧠 Training ML Predictor...")
    predictor = MLPredictor()
    
    # Train with minimum 20 samples
    predictor.train(min_samples=20)
    
    # 5. Verify
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
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"🎉 DONE! Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("="*70)


if __name__ == "__main__":
    main()
