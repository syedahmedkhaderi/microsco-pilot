"""Training Data Manager for ML System

This module manages the collection, storage, and retrieval of training data
for the machine learning models.

WHAT IS TRAINING DATA?
Training data is a collection of examples that teach the ML model.
Each example shows:
- What the image looked like (features)
- What parameters we used (speed, resolution, force)
- What quality we got (good or bad result)

The model learns: "When I see THIS kind of image, use THESE parameters"
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TrainingDataManager:
    """Manage training data for ML parameter prediction
    
    This class handles:
    1. Storing training examples (features + parameters + results)
    2. Loading training data for model training
    3. Managing a sliding window (keep recent, discard old)
    4. Preventing data quality issues
    
    Think of this as a database of past scans that the ML learns from.
    
    Example:
        >>> manager = TrainingDataManager()
        >>> manager.add_training_sample(features, parameters, results)
        >>> X, y = manager.get_training_data('speed')
        >>> # Now train model on X (features) and y (speeds)
    """
    
    def __init__(self, data_dir: str = 'data/ml_training', max_samples: int = 1000):
        """Initialize training data manager
        
        Args:
            data_dir: Directory to store training data
            max_samples: Maximum number of samples to keep (sliding window)
                        Older samples are deleted when we exceed this
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_samples = max_samples
        self.data_file = self.data_dir / 'training_data.csv'
        self.metadata_file = self.data_dir / 'metadata.json'
        
        # Feature names (must match FeatureExtractor) - DEFINE BEFORE _initialize_storage
        self.feature_names = [
            'sharpness', 'contrast', 'complexity', 'edge_strength',
            'noise_level', 'fft_high_freq', 'fft_low_freq',
            'mean_gradient', 'std_gradient', 'range_value'
        ]
        
        # Parameter names we're predicting
        self.parameter_names = ['speed', 'resolution', 'force']
        
        # Result metrics we're tracking
        self.result_names = ['quality', 'success', 'time']
        
        # Initialize files if they don't exist
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Create storage files if they don't exist"""
        if not self.data_file.exists():
            # Create CSV with header
            header = self._get_csv_header()
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        
        if not self.metadata_file.exists():
            # Create metadata
            metadata = {
                'created': datetime.now().isoformat(),
                'total_samples': 0,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _get_csv_header(self) -> List[str]:
        """Generate CSV header row
        
        Returns a list like:
        ['timestamp', 'sharpness', 'contrast', ..., 'speed', 'resolution', ...]
        """
        header = ['timestamp']
        
        # Feature columns
        for feat in self.feature_names:
            header.append(f'feat_{feat}')
        
        # Parameter columns
        for param in self.parameter_names:
            header.append(f'param_{param}')
        
        # Result columns
        for result in self.result_names:
            header.append(f'result_{result}')
        
        # Metadata columns
        header.extend(['sample_type', 'afm_mode'])
        
        return header
    
    def add_training_sample(self, 
                          features: Dict[str, float],
                          parameters: Dict[str, float],
                          results: Dict[str, float],
                          metadata: Optional[Dict[str, str]] = None):
        """Add a new training sample
        
        This is called after each scan completes.
        We record what we tried and what happened.
        
        WHAT GOES IN:
        - features: Image characteristics (from FeatureExtractor)
        - parameters: What settings we used
        - results: What quality we got
        - metadata: Extra info (sample type, etc.)
        
        THE ML MODEL LEARNS:
        Given these features → use these parameters → to get this quality
        
        Args:
            features: Feature dictionary (sharpness, contrast, etc.)
            parameters: Parameter dictionary (speed, resolution, force)
            results: Result dictionary (quality, success, time)
            metadata: Optional metadata (sample_type, afm_mode)
        """
        if metadata is None:
            metadata = {}
        
        # Create row for CSV
        row = [datetime.now().isoformat()]
        
        # Add features
        for feat in self.feature_names:
            row.append(features.get(feat, 0.0))
        
        # Add parameters
        for param in self.parameter_names:
            row.append(parameters.get(param, 0.0))
        
        # Add results
        for result in self.result_names:
            row.append(results.get(result, 0.0))
        
        # Add metadata
        row.append(metadata.get('sample_type', 'unknown'))
        row.append(metadata.get('afm_mode', 'unknown'))
        
        # Append to CSV
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Update metadata
        self._update_metadata()
        
        # Manage sliding window
        self._enforce_max_samples()
    
    def get_training_data(self, 
                         target: str = 'speed',
                         min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data for a specific prediction target
        
        WHAT THIS DOES:
        Loads all training samples and returns:
        - X: Feature matrix (each row = one scan's features)
        - y: Target values (what we're trying to predict)
        
        Example for speed prediction:
        X = [[sharpness, contrast, ...],  ← Scan 1
             [sharpness, contrast, ...],  ← Scan 2
             ...]
        y = [10.5,  ← Speed used in scan 1
             15.2,  ← Speed used in scan 2
             ...]
        
        Args:
            target: What to predict ('speed', 'resolution', or 'force')
            min_samples: Minimum samples required (return empty if less)
            
        Returns:
            (X, y) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Target values (n_samples,)
        """
        # Load data
        try:
            df = pd.read_csv(self.data_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return np.array([]), np.array([])
        
        # Check if we have enough samples
        if len(df) < min_samples:
            return np.array([]), np.array([])
        
        # Extract features (X)
        feature_columns = [f'feat_{name}' for name in self.feature_names]
        X = df[feature_columns].values
        
        # Extract target (y)
        target_column = f'param_{target}'
        if target_column not in df.columns:
            raise ValueError(f"Unknown target: {target}")
        
        y = df[target_column].values
        
        # Remove any rows with NaN or Inf
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def get_recent_samples(self, n: int = 100) -> pd.DataFrame:
        """Get the most recent n training samples
        
        Useful for:
        - Checking what the model is learning from
        - Debugging
        - Visualizing recent performance
        
        Args:
            n: Number of recent samples to return
            
        Returns:
            DataFrame with recent samples
        """
        try:
            df = pd.read_csv(self.data_file)
            return df.tail(n)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the training data
        
        Returns:
            Dictionary with:
            - total_samples: How many training examples we have
            - feature_means: Average value of each feature
            - parameter_ranges: Min/max of each parameter
            - quality_distribution: How good our scans have been
        """
        try:
            df = pd.read_csv(self.data_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return {'total_samples': 0}
        
        stats = {
            'total_samples': len(df),
            'feature_means': {},
            'parameter_ranges': {},
            'quality_stats': {}
        }
        
        # Feature means
        for feat in self.feature_names:
            col = f'feat_{feat}'
            if col in df.columns:
                stats['feature_means'][feat] = float(df[col].mean())
        
        # Parameter ranges
        for param in self.parameter_names:
            col = f'param_{param}'
            if col in df.columns:
                stats['parameter_ranges'][param] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean())
                }
        
        # Quality stats
        if 'result_quality' in df.columns:
            stats['quality_stats'] = {
                'mean': float(df['result_quality'].mean()),
                'std': float(df['result_quality'].std()),
                'min': float(df['result_quality'].min()),
                'max': float(df['result_quality'].max())
            }
        
        return stats
    
    def _update_metadata(self):
        """Update metadata file with current stats"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {'created': datetime.now().isoformat()}
        
        # Update counts
        try:
            df = pd.read_csv(self.data_file)
            metadata['total_samples'] = len(df)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            metadata['total_samples'] = 0
        
        metadata['last_updated'] = datetime.now().isoformat()
        
        # Save
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _enforce_max_samples(self):
        """Keep only the most recent max_samples
        
        This implements the SLIDING WINDOW:
        - Keep the most recent 1000 samples (or whatever max is set to)
        - Delete older samples
        
        WHY?
        - Prevents file from growing infinitely
        - Focuses learning on recent, relevant data
        - Old samples might be from different conditions
        """
        try:
            df = pd.read_csv(self.data_file)
            
            if len(df) > self.max_samples:
                # Keep only most recent
                df = df.tail(self.max_samples)
                
                # Rewrite file
                df.to_csv(self.data_file, index=False)
                
                print(f"⚙️  Sliding window: Kept {self.max_samples} most recent samples")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
    
    def clear_data(self):
        """Clear all training data (use with caution!)"""
        if self.data_file.exists():
            self.data_file.unlink()
        self._initialize_storage()
        print("🗑️  Cleared all training data")


# Educational example
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING DATA MANAGER - EDUCATIONAL DEMO")
    print("="*70)
    
    # Create manager
    print("\n1. Creating TrainingDataManager...")
    manager = TrainingDataManager(data_dir='data/ml_training_test')
    print(f"   Data directory: {manager.data_dir}")
    print(f"   Max samples: {manager.max_samples}")
    
    # Add some example samples
    print("\n2. Adding training samples...")
    for i in range(5):
        # Example features
        features = {
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
        
        # Example parameters
        parameters = {
            'speed': np.random.uniform(5, 15),
            'resolution': np.random.choice([128, 256, 512]),
            'force': np.random.uniform(1, 4)
        }
        
        # Example results
        results = {
            'quality': np.random.uniform(3, 8),
            'success': 1,
            'time': np.random.uniform(10, 30)
        }
        
        manager.add_training_sample(features, parameters, results)
        print(f"   Added sample {i+1}/5")
    
    # Get training data
    print("\n3. Loading training data for speed prediction...")
    X, y = manager.get_training_data('speed', min_samples=3)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target values shape: {y.shape}")
    print(f"   Speed range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Get statistics
    print("\n4. Training data statistics:")
    stats = manager.get_statistics()
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Average sharpness: {stats['feature_means'].get('sharpness', 0):.2f}")
    print(f"   Speed range: {stats['parameter_ranges']['speed']}")
    
    # Clean up test data
    import shutil
    shutil.rmtree('data/ml_training_test')
    
    print("\n✅ Training data management complete!")
    print("\nThis system stores and manages all the examples the ML learns from.\n")
