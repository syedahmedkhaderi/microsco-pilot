"""Feature Extractor for AFM Images

This module extracts numerical features from AFM scan images that can be used
by machine learning models to predict optimal scan parameters.

WHAT ARE FEATURES?
Features are measurable properties of an image that describe its characteristics.
Think of them like describing a person: height, weight, hair color, etc.
For images, we measure things like sharpness, contrast, complexity, etc.

WHY DO WE NEED FEATURES?
Machine learning models can't look at pictures directly. They need numbers.
Features convert images → numbers that the ML model can understand.
"""

import numpy as np
from scipy import ndimage, fft
from typing import Dict, Tuple
import warnings


class FeatureExtractor:
    """Extract numerical features from AFM images for ML prediction
    
    This class converts AFM scan images into feature vectors that describe:
    - How sharp the image is
    - How much contrast it has
    - How complex the structures are
    - How noisy the data is
    - What frequency components are present
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> image = np.random.randn(256, 256)
        >>> features = extractor.extract_features(image)
        >>> print(features['sharpness'])
        4.52
    """
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.feature_names = [
            'sharpness',
            'contrast', 
            'complexity',
            'edge_strength',
            'noise_level',
            'fft_high_freq',
            'fft_low_freq',
            'mean_gradient',
            'std_gradient',
            'range_value'
        ]
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract all features from an AFM image
        
        This is the main function you'll use. Give it an image, get back features.
        
        Args:
            image: 2D numpy array or PIL Image of AFM scan data
            
        Returns:
            Dictionary with feature names and values
            
        Example:
            >>> features = extractor.extract_features(afm_image)
            >>> # Now you have all features as numbers
            >>> print(f"Sharpness: {features['sharpness']:.2f}")
        """
        # Handle PIL Image (convert to numpy array)
        if not isinstance(image, np.ndarray):
            try:
                from PIL import Image
                if isinstance(image, Image.Image):
                    image = np.array(image, dtype=np.float32)
                    # If RGB, convert to grayscale
                    if len(image.shape) == 3:
                        image = np.mean(image, axis=2)
            except:
                pass
        
        # Make sure image is 2D
        if len(image.shape) != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")
        
        # Handle edge cases (empty or all zeros)
        if image.size == 0 or np.all(image == 0):
            return self._zero_features()
        
        # Remove NaN and Inf
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract each type of feature
        features = {}
        
        # 1. Sharpness (Laplacian variance)
        features['sharpness'] = self._compute_sharpness(image)
        
        # 2. Contrast (standard deviation)
        features['contrast'] = self._compute_contrast(image)
        
        # 3. Complexity (gradient statistics)
        features['complexity'] = self._compute_complexity(image)
        
        # 4. Edge strength (Sobel)
        features['edge_strength'] = self._compute_edge_strength(image)
        
        # 5. Noise level
        features['noise_level'] = self._compute_noise(image)
        
        # 6. Frequency features (FFT)
        fft_features = self._compute_frequency_features(image)
        features.update(fft_features)
        
        # 7. Gradient statistics
        grad_features = self._compute_gradient_stats(image)
        features.update(grad_features)
        
        # 8. Range
        features['range_value'] = self._compute_range(image)
        
        return features
    
    def _compute_sharpness(self, image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance
        
        WHAT IS SHARPNESS?
        Sharpness measures how clear and well-defined features are in the image.
        A sharp image has crisp edges. A blurry image has soft edges.
        
        HOW WE MEASURE IT:
        We use the Laplacian operator, which responds strongly to edges.
        The variance (spread) of the Laplacian tells us how sharp the image is.
        
        High variance = Sharp image = Good scan parameters
        Low variance = Blurry image = Bad scan parameters
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            # Laplacian filter detects edges
            laplacian = ndimage.laplace(image)
            
            # Variance measures how much the edges vary
            # (more variation = sharper edges)
            variance = np.var(laplacian)
            
            return float(variance)
        except Exception:
            return 0.0
    
    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute image contrast
        
        WHAT IS CONTRAST?
        Contrast is the difference between bright and dark areas.
        High contrast = Easy to see features
        Low contrast = Everything looks similar
        
        HOW WE MEASURE IT:
        Standard deviation of pixel values.
        If all pixels are similar → low std → low contrast
        If pixels vary a lot → high std → high contrast
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Contrast score (higher = more contrast)
        """
        try:
            # Normalize to 0-1 range
            if np.max(image) != np.min(image):
                normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
            else:
                normalized = image
            
            # Standard deviation measures spread
            contrast = np.std(normalized)
            
            return float(contrast)
        except Exception:
            return 0.0
    
    def _compute_complexity(self, image: np.ndarray) -> float:
        """Compute structural complexity
        
        WHAT IS COMPLEXITY?
        Complexity measures how "busy" or detailed the image is.
        Simple = flat surface with few features
        Complex = lots of structures, edges, patterns
        
        HOW WE MEASURE IT:
        We look at the gradients (rate of change).
        Flat areas have low gradients.
        Busy areas have high, varying gradients.
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Complexity score (higher = more complex)
        """
        try:
            # Sobel filters compute gradients in x and y
            grad_x = ndimage.sobel(image, axis=0)
            grad_y = ndimage.sobel(image, axis=1)
            
            # Magnitude of gradient
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Complexity = how much gradients vary
            complexity = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-10)
            
            return float(complexity)
        except Exception:
            return 0.0
    
    def _compute_edge_strength(self, image: np.ndarray) -> float:
        """Compute average edge strength
        
        WHAT ARE EDGES?
        Edges are boundaries between different regions (e.g., particle edge).
        Strong edges = well-defined structures
        Weak edges = unclear boundaries
        
        HOW WE MEASURE IT:
        Sobel operator detects edges.
        We average the edge magnitudes across the image.
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Edge strength score (higher = stronger edges)
        """
        try:
            grad_x = ndimage.sobel(image, axis=0)
            grad_y = ndimage.sobel(image, axis=1)
            
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Average edge strength
            edge_strength = np.mean(edge_magnitude)
            
            return float(edge_strength)
        except Exception:
            return 0.0
    
    def _compute_noise(self, image: np.ndarray) -> float:
        """Estimate noise level
        
        WHAT IS NOISE?
        Noise is random variations that don't represent real features.
        Like TV static - it obscures the real image.
        
        HOW WE MEASURE IT:
        1. Smooth the image with Gaussian filter (removes noise)
        2. Subtract smoothed from original
        3. What's left is mostly noise
        4. Measure the variance of this residual
        
        High noise = Poor scan parameters
        Low noise = Good scan parameters
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Noise level (higher = noisier)
        """
        try:
            # Smooth image with Gaussian filter
            smoothed = ndimage.gaussian_filter(image, sigma=2.0)
            
            # Residual = original - smoothed (mostly noise)
            residual = image - smoothed
            
            # Variance of residual estimates noise
            noise = np.std(residual)
            
            return float(noise)
        except Exception:
            return 0.0
    
    def _compute_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute frequency domain features using FFT
        
        WHAT IS FFT (Fast Fourier Transform)?
        FFT converts image from spatial domain (pixels) to frequency domain.
        
        SPATIAL DOMAIN: "This pixel is bright, that pixel is dark"
        FREQUENCY DOMAIN: "This image has mostly low frequencies (smooth)
                           or high frequencies (detailed)"
        
        WHY IS THIS USEFUL?
        - Low frequencies = smooth, large-scale features
        - High frequencies = fine details, edges, noise
        
        If we see mostly high frequencies:
        - Either: lots of fine detail (good!)
        - Or: lots of noise (bad!)
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Dictionary with 'fft_high_freq' and 'fft_low_freq'
        """
        try:
            # Compute 2D FFT
            fft_image = fft.fft2(image)
            fft_shifted = fft.fftshift(fft_image)  # Move DC to center
            
            # Power spectrum (magnitude squared)
            power = np.abs(fft_shifted) ** 2
            
            # Get center (low frequencies) and edges (high frequencies)
            h, w = power.shape
            center_h, center_w = h // 2, w // 2
            
            # Low frequency region (center 25%)
            low_freq_region = power[
                center_h - h//8:center_h + h//8,
                center_w - w//8:center_w + w//8
            ]
            
            # High frequency region (outer 25%)
            high_freq_mask = np.ones_like(power, dtype=bool)
            high_freq_mask[
                center_h - h//4:center_h + h//4,
                center_w - w//4:center_w + w//4
            ] = False
            high_freq_region = power[high_freq_mask]
            
            # Average power in each region
            low_freq = np.mean(low_freq_region) if low_freq_region.size > 0 else 0
            high_freq = np.mean(high_freq_region) if high_freq_region.size > 0 else 0
            
            # Normalize
            total = low_freq + high_freq + 1e-10
            
            return {
                'fft_low_freq': float(low_freq / total),
                'fft_high_freq': float(high_freq / total)
            }
        except Exception:
            return {'fft_low_freq': 0.0, 'fft_high_freq': 0.0}
    
    def _compute_gradient_stats(self, image: np.ndarray) -> Dict[str, float]:
        """Compute gradient statistics
        
        WHAT ARE GRADIENTS?
        Gradients measure how quickly the image changes.
        Flat area = zero gradient
        Edge/slope = high gradient
        
        WHY MEAN AND STD?
        - Mean gradient: Overall "steepness" of features
        - Std gradient: How much steepness varies
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Dictionary with 'mean_gradient' and 'std_gradient'
        """
        try:
            grad_x = ndimage.sobel(image, axis=0)
            grad_y = ndimage.sobel(image, axis=1)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return {
                'mean_gradient': float(np.mean(gradient_magnitude)),
                'std_gradient': float(np.std(gradient_magnitude))
            }
        except Exception:
            return {'mean_gradient': 0.0, 'std_gradient': 0.0}
    
    def _compute_range(self, image: np.ndarray) -> float:
        """Compute value range (max - min)
        
        WHAT IS RANGE?
        Range is the difference between highest and lowest pixel values.
        
        In AFM: Range tells you the height variation
        - Small range = flat sample
        - Large range = tall features
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            Range value
        """
        try:
            return float(np.max(image) - np.min(image))
        except Exception:
            return 0.0
    
    def _zero_features(self) -> Dict[str, float]:
        """Return zero features (fallback for empty images)"""
        return {name: 0.0 for name in self.feature_names}
    
    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """Extract features as a numpy array (for ML models)
        
        This is what we'll feed to the ML model.
        ML models want arrays, not dictionaries.
        
        Args:
            image: 2D AFM scan image
            
        Returns:
            1D numpy array of feature values in consistent order
        """
        features_dict = self.extract_features(image)
        return np.array([features_dict[name] for name in self.feature_names])


# Educational example at bottom of file
if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE EXTRACTOR - EDUCATIONAL DEMO")
    print("="*70)
    
    # Create example image
    print("\n1. Creating a test AFM image...")
    test_image = np.random.randn(128, 128) + 5 * np.sin(
        np.linspace(0, 10, 128)[:, None]
    )
    print(f"   Image shape: {test_image.shape}")
    print(f"   Value range: [{test_image.min():.2f}, {test_image.max():.2f}]")
    
    # Extract features
    print("\n2. Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(test_image)
    
    print("\n3. Feature Results:")
    print("   " + "-"*60)
    for name, value in features.items():
        print(f"   {name:20s}: {value:10.4f}")
    print("   " + "-"*60)
    
    print("\n4. As ML-ready array:")
    feature_vector = extractor.get_feature_vector(test_image)
    print(f"   {feature_vector}")
    
    print("\n✅ Feature extraction complete!")
    print("\nThese numbers describe the image properties that ML will use")
    print("to predict optimal scan parameters.\n")
