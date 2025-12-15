
"""
Remedial Training using REAL AFM Data
-------------------------------------
This script retrains the SmartScan ML model using ONLY real AFM data 
from the data/AFM/ directory.

It performs "simulated scans" on the real data using the improved physics engine
to teach the ML model the relationship between:
Image Features (Real) -> Scan Parameters -> Scan Quality
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.h5_data_loader import DatasetManager
from src.dtm_controller import AdaptiveAFMController
from src.ml_predictor import MLPredictor

def load_real_data_regions():
    """Load all regions from all h5 files in data/AFM"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'AFM')
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    
    if not h5_files:
        print(f"❌ No h5 files found in {data_dir}")
        return []
    
    all_regions = []
    print(f"📂 Found {len(h5_files)} real AFM files. Loading regions...")
    
    for h5_file in h5_files:
        try:
            with DatasetManager(h5_file) as dm:
                # Get more regions per file to have enough training data
                regions = dm.get_scan_regions(num_regions=10)
                for region in regions:
                    all_regions.append(region)
        except Exception as e:
            print(f"   ⚠️  Skipping {os.path.basename(h5_file)}: {e}")
            
    print(f"✅ Loaded {len(all_regions)} real texturized regions for training.")
    return all_regions

def train_on_real_data():
    print("🚀 Starting SmartScan Training on REAL DATA...")
    
    # 1. Load Real Data
    regions = load_real_data_regions()
    if not regions:
        return
    
    # 2. Initialize Components
    controller = AdaptiveAFMController()
    controller.simulation_mode = False # Use patched DTMicroscope with Drift Physics
    predictor = MLPredictor()
    
    print("\n🔬 Simulating Scans on Real Data to Generate Training Examples...")
    
    # 3. Generate Training Data
    # For each real region, try multiple parameter sets to find what works best
    # This acts as our "Ground Truth" generation
    
    samples_generated = 0
    target_samples = 80 # Reduced for speed, DTMicroscope is slower than fallback
    
    # We loop through regions and apply random parameters to build a dataset
    # We ensure we cover high speeds to teach the model they are safe now
    
    while samples_generated < target_samples:
        for region in regions:
            if samples_generated >= target_samples:
                break
                
            # Random exploration of parameters on this REAL image
            # We want to explore the full range to learn the "curve" -> Low speed (drift) vs High speed (error)
            speed = np.random.uniform(2.0, 18.0) 
            resolution = np.random.choice([128, 256, 512])
            force = np.random.uniform(0.5, 4.0)
            
            params = {
                'speed': speed,
                'resolution': resolution,
                'force': force
            }
            
            # Update controller
            controller.scan_params.update(params)
            
            # Physics Scan Simulation on REAL Data
            # We pass the real region as 'external_image'
            scanned_image_pil, time_taken = controller.scan_region(
                0, 0, 1000, external_image=region
            )
            
            # Get Quality from the Physics Engine
            quality_metrics = controller.get_quality_from_scan(controller.last_scan_result)
            quality = quality_metrics['quality']
            
            # Teach the ML Model
            # Input: The original REAL image
            # Action: Parameters used
            # Reward: Quality achieved
            predictor.update_online(
                image=region, 
                parameters=params,
                results={'quality': quality}
            )
            
            samples_generated += 1
            if samples_generated % 20 == 0:
                print(f"   Generated {samples_generated}/{target_samples} samples. Latest Quality: {quality:.1f} (Speed: {speed:.1f})")

    print("\n✅ Training dataset complete.")
    
    # Force minimal training right now
    predictor.train(min_samples=50)
    predictor.save_models()
    
    print("\n💾 ML Models trained on REAL DATA and saved.")
    
    # Test Verification
    print("\n🧪 Verification Prediction on a Real Region:")
    test_region = regions[0]
    pred = predictor.predict(test_region)
    print(f"   For Real Region 0:")
    print(f"   Predicted Speed: {pred['parameters']['speed']:.1f} um/s")
    print(f"   Predicted Res:   {pred['parameters']['resolution']}")
    print(f"   Confidence:      {pred['overall_confidence']:.2f}")
    
    if pred['parameters']['speed'] > 8.0:
        print("   ✅ Success: Model learned to use higher speeds!")
    else:
        print("   ⚠️  Warning: Model still conservative. Might need more high-speed positive samples.")

if __name__ == "__main__":
    train_on_real_data()
