'''AFM H5 Data Loader Module

This module provides comprehensive functionality for downloading, loading, and processing
AFM (Atomic Force Microscopy) h5 files from DTMicroscope and other repositories.

Features:
- Download sample AFM datasets from public repositories
- Read and validate h5 file structures
- Extract raw topography data, metadata, and spectroscopic information
- Support for multiple AFM modes (contact, tapping, force spectroscopy)
- Robust error handling for corrupted/incomplete data
- Data normalization for SmartScan integration
'''

import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py is required for this module. Install it with: pip install h5py>=3.8"
    )

try:
    import requests
except ImportError:
    requests = None
    warnings.warn("requests library not found. Download functionality will be limited.")


# ============================================================================
# Custom Exceptions
# ============================================================================

class H5DataLoaderError(Exception):
    """Base exception for h5_data_loader module"""
    pass


class CorruptedDataError(H5DataLoaderError):
    """Raised when h5 file data is corrupted or invalid"""
    pass


class UnsupportedStructureError(H5DataLoaderError):
    """Raised when h5 file structure is not recognized"""
    pass


class MissingChannelError(H5DataLoaderError):
    """Raised when expected channel is not found"""
    pass


# ============================================================================
# Sample Dataset Registry
# ============================================================================

def get_sample_datasets() -> List[Dict[str, str]]:
    """Get list of publicly available AFM sample datasets
    
    Returns:
        List of dataset dictionaries with 'name', 'url', 'description', and 'mode' keys
    """
    # DTMicroscope GitHub repository base URL
    base_url = "https://raw.githubusercontent.com/pycroscopy/DTMicroscope/main/data/AFM/"
    
    datasets = [
        {
            'name': 'graphene_afm',
            'url': f'{base_url}AFM_Graphene.h5',
            'description': 'AFM topography of graphene sample from DTMicroscope',
            'mode': 'contact',
            'source': 'pycroscopy/DTMicroscope'
        },
        {
            'name': 'afm_sample_1',
            'url': f'{base_url}sample1.h5',
            'description': 'AFM sample data from DTMicroscope repository',
            'mode': 'tapping',
            'source': 'pycroscopy/DTMicroscope',
            'note': 'Check https://github.com/pycroscopy/DTMicroscope/tree/main/data/AFM for available files'
        },
        {
            'name': 'afm_sample_2',
            'url': f'{base_url}sample2.h5',
            'description': 'AFM sample data from DTMicroscope repository',
            'mode': 'contact',
            'source': 'pycroscopy/DTMicroscope',
            'note': 'Check https://github.com/pycroscopy/DTMicroscope/tree/main/data/AFM for available files'
        }
    ]
    return datasets


# ============================================================================
# Download Functions
# ============================================================================

def download_sample_data(url: str, save_path: str, chunk_size: int = 8192) -> str:
    """Download AFM h5 file from repository with progress tracking
    
    Args:
        url: URL to download from
        save_path: Local path to save file
        chunk_size: Download chunk size in bytes
        
    Returns:
        Path to downloaded file
        
    Raises:
        ImportError: If requests library is not available
        H5DataLoaderError: If download fails
    """
    if requests is None:
        raise ImportError(
            "requests library is required for downloads. "
            "Install it with: pip install requests"
        )
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        print(f"📥 Downloading AFM data from {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"   Progress: {progress:.1f}%", end='\r')
        
        print(f"\n✅ Downloaded to: {save_path}")
        
        # Validate downloaded file
        if not validate_h5_file(save_path):
            os.remove(save_path)
            raise CorruptedDataError(f"Downloaded file is corrupted: {save_path}")
        
        return save_path
        
    except requests.exceptions.RequestException as e:
        raise H5DataLoaderError(f"Download failed: {e}")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise H5DataLoaderError(f"Error during download: {e}")


# ============================================================================
# H5 File Inspection & Validation
# ============================================================================

def validate_h5_file(file_path: str) -> bool:
    """Check if file is a valid h5 file
    
    Args:
        file_path: Path to h5 file
        
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Basic validation - file can be opened and has some structure
            return len(f.keys()) > 0
    except (OSError, IOError, Exception):
        return False


def inspect_h5_structure(file_path: str, max_depth: int = 5) -> None:
    """Explore and print h5 file hierarchy
    
    Args:
        file_path: Path to h5 file
        max_depth: Maximum depth to explore
        
    Raises:
        FileNotFoundError: If file doesn't exist
        H5DataLoaderError: If file cannot be opened
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n🔍 H5 File Structure: {file_path}")
            print("=" * 60)
            _print_h5_tree(f, "", max_depth)
            print("=" * 60)
    except Exception as e:
        raise H5DataLoaderError(f"Failed to open h5 file: {e}")


def _print_h5_tree(group: h5py.Group, prefix: str = "", max_depth: int = 5, 
                   current_depth: int = 0) -> None:
    """Recursively print h5 group/dataset tree
    
    Args:
        group: h5py Group to print
        prefix: String prefix for indentation
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """
    if current_depth >= max_depth:
        return
    
    items = list(group.items())
    for i, (name, item) in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        if isinstance(item, h5py.Group):
            print(f"{prefix}{connector}📁 {name}/")
            extension = "    " if is_last else "│   "
            _print_h5_tree(item, prefix + extension, max_depth, current_depth + 1)
        elif isinstance(item, h5py.Dataset):
            shape_str = f"shape={item.shape}" if hasattr(item, 'shape') else ""
            dtype_str = f"dtype={item.dtype}" if hasattr(item, 'dtype') else ""
            print(f"{prefix}{connector}📄 {name} ({shape_str}, {dtype_str})")
            
            # Print some attributes
            if len(item.attrs) > 0:
                extension = "    " if is_last else "│   "
                for attr_name in list(item.attrs.keys())[:3]:  # Show first 3 attributes
                    print(f"{prefix}{extension}  ⚙️  {attr_name}: {item.attrs[attr_name]}")


# ============================================================================
# Data Extraction Functions
# ============================================================================

def extract_raw_data(h5_group: Union[h5py.File, h5py.Group], 
                     channel_path: Optional[str] = None) -> np.ndarray:
    """Extract raw topography or channel data from h5 file
    
    Args:
        h5_group: h5py File or Group object
        channel_path: Path to channel data (e.g., '/Measurement_000/Channel_000/Raw_Data')
                     If None, will search common locations
        
    Returns:
        Raw data as numpy array
        
    Raises:
        MissingChannelError: If channel data cannot be found
    """
    # Common channel paths to try
    common_paths = [
        '/Measurement_000/Channel_000/Raw_Data',
        '/Measurement/Channel_000/Raw_Data',
        '/Raw_Data',
        '/Channel_000/Raw_Data',
        '/data',
        '/topography',
        '/AFM_Image',
    ]
    
    if channel_path:
        common_paths.insert(0, channel_path)
    
    # Try to find data in common locations
    for path in common_paths:
        try:
            if path in h5_group:
                data = h5_group[path][()]
                return np.array(data)
        except Exception:
            continue
    
    # If not found in common locations, search for datasets
    datasets = []
    def find_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)
    
    h5_group.visititems(find_datasets)
    
    # Look for likely candidates (large numerical datasets)
    for dataset_path in datasets:
        try:
            dataset = h5_group[dataset_path]
            # Check if it's a reasonable AFM data array
            if (isinstance(dataset, h5py.Dataset) and 
                len(dataset.shape) >= 1 and
                np.issubdtype(dataset.dtype, np.number)):
                return np.array(dataset[()])
        except Exception:
            continue
    
    raise MissingChannelError(
        f"Could not find raw data. Searched paths: {common_paths}"
    )


def extract_position_data(h5_group: Union[h5py.File, h5py.Group]) -> Dict[str, np.ndarray]:
    """Extract position indices and physical values
    
    Args:
        h5_group: h5py File or Group object
        
    Returns:
        Dictionary with 'indices' and 'values' keys containing position data
    """
    position_data = {
        'indices': None,
        'values': None,
        'units': 'nm'  # Default unit
    }
    
    # Common position data paths
    position_paths = [
        '/Position_Indices',
        '/Position_Values',
        '/X_Position',
        '/Y_Position',
        '/position',
    ]
    
    for path in position_paths:
        try:
            if path in h5_group:
                data = np.array(h5_group[path][()])
                if 'Indices' in path:
                    position_data['indices'] = data
                else:
                    position_data['values'] = data
                    
                # Try to get units from attributes
                if 'units' in h5_group[path].attrs:
                    position_data['units'] = h5_group[path].attrs['units']
        except Exception:
            continue
    
    return position_data


def extract_spectroscopic_data(h5_group: Union[h5py.File, h5py.Group]) -> Dict[str, np.ndarray]:
    """Extract spectroscopic indices and values
    
    Args:
        h5_group: h5py File or Group object
        
    Returns:
        Dictionary with spectroscopic data
    """
    spectroscopy_data = {
        'indices': None,
        'values': None,
        'units': 'Hz'  # Default unit
    }
    
    # Common spectroscopy data paths
    spectroscopy_paths = [
        '/Spectroscopic_Indices',
        '/Spectroscopic_Values',
        '/frequency',
        '/force',
    ]
    
    for path in spectroscopy_paths:
        try:
            if path in h5_group:
                data = np.array(h5_group[path][()])
                if 'Indices' in path:
                    spectroscopy_data['indices'] = data
                else:
                    spectroscopy_data['values'] = data
                    
                # Try to get units from attributes
                if 'units' in h5_group[path].attrs:
                    spectroscopy_data['units'] = h5_group[path].attrs['units']
        except Exception:
            continue
    
    return spectroscopy_data


def extract_metadata(h5_group: Union[h5py.File, h5py.Group]) -> Dict[str, Any]:
    """Extract scan parameters and metadata from h5 file
    
    Args:
        h5_group: h5py File or Group object
        
    Returns:
        Dictionary with metadata
    """
    metadata = {
        'scan_size': None,
        'resolution': None,
        'scan_rate': None,
        'scan_speed': None,
        'force': None,
        'afm_mode': 'unknown',
        'date': None,
        'software': None,
        'units': {},
        'other_attributes': {}
    }
    
    # Extract root-level attributes
    for attr_name in h5_group.attrs.keys():
        value = h5_group.attrs[attr_name]
        attr_lower = attr_name.lower()
        
        if 'scan_size' in attr_lower or 'scansize' in attr_lower:
            metadata['scan_size'] = value
        elif 'resolution' in attr_lower or 'pixels' in attr_lower:
            metadata['resolution'] = value
        elif 'scan_rate' in attr_lower or 'scanrate' in attr_lower:
            metadata['scan_rate'] = value
        elif 'speed' in attr_lower:
            metadata['scan_speed'] = value
        elif 'force' in attr_lower:
            metadata['force'] = value
        elif 'mode' in attr_lower:
            metadata['afm_mode'] = str(value).lower()
        elif 'date' in attr_lower or 'time' in attr_lower:
            metadata['date'] = value
        elif 'software' in attr_lower or 'version' in attr_lower:
            metadata['software'] = value
        else:
            metadata['other_attributes'][attr_name] = value
    
    # Try to detect AFM mode from attributes
    if metadata['afm_mode'] == 'unknown':
        metadata['afm_mode'] = _detect_afm_mode(h5_group, metadata)
    
    return metadata


def _detect_afm_mode(h5_group: Union[h5py.File, h5py.Group], 
                     metadata: Dict[str, Any]) -> str:
    """Detect AFM mode from file structure and metadata
    
    Args:
        h5_group: h5py File or Group object
        metadata: Extracted metadata dictionary
        
    Returns:
        Detected AFM mode as string
    """
    # Check for mode indicators in group names
    mode_indicators = {
        'contact': ['contact', 'contact_mode', 'dc'],
        'tapping': ['tapping', 'ac', 'amplitude', 'phase'],
        'force_spectroscopy': ['force', 'spectroscopy', 'force_curve', 'fdc']
    }
    
    # Search all paths in file
    all_paths = []
    def collect_paths(name, obj):
        all_paths.append(name.lower())
    
    h5_group.visititems(collect_paths)
    
    # Check for mode indicators
    for mode, indicators in mode_indicators.items():
        for indicator in indicators:
            for path in all_paths:
                if indicator in path:
                    return mode
    
    # Default to contact mode if unable to detect
    return 'contact'


# ============================================================================
# Data Normalization
# ============================================================================

def normalize_data(data: Union[np.ndarray, List[np.ndarray]], 
                   target_shape: Optional[Tuple[int, int]] = None,
                   preserve_aspect_ratio: bool = True) -> np.ndarray:
    """Normalize AFM data to match SmartScan's expected format
    
    This function handles:
    - 2D position x spectroscopic format conversion
    - Reshaping to target dimensions
    - Physical units and scale preservation
    - Multiple data format conversions
    
    Args:
        data: Input data (single array or list of arrays)
        target_shape: Target shape (height, width), defaults to (256, 256)
        preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
        
    Returns:
        Normalized data as numpy array
    """
    if target_shape is None:
        target_shape = (256, 256)
    
    # Convert list to array if needed
    if isinstance(data, list):
        data = np.array(data)
    
    # Handle different data formats
    if len(data.shape) == 1:
        # 1D data - try to reshape to square
        size = int(np.sqrt(len(data)))
        if size * size == len(data):
            data = data.reshape(size, size)
        else:
            # Pad to nearest square
            size = int(np.ceil(np.sqrt(len(data))))
            padded = np.zeros(size * size)
            padded[:len(data)] = data
            data = padded.reshape(size, size)
    
    elif len(data.shape) == 3:
        # 3D data - take first channel or average
        if data.shape[0] < data.shape[2]:
            # Shape is (channels, height, width)
            data = data[0]  # Take first channel
        else:
            # Shape is (height, width, channels)
            data = np.mean(data, axis=2)
    
    elif len(data.shape) > 3:
        # Higher dimensional - flatten extra dimensions
        target_dims = data.shape[-2:]
        data = data.reshape(-1, *target_dims)
        data = data[0]  # Take first slice
    
    # Ensure 2D
    if len(data.shape) != 2:
        raise ValueError(f"Cannot normalize data with shape {data.shape} to 2D")
    
    # Resize to target shape if needed
    if data.shape != target_shape:
        data = _resize_array(data, target_shape, preserve_aspect_ratio)
    
    # Normalize values to reasonable range (0-1 or preserve physical units)
    data = data.astype(np.float64)
    
    # Remove NaN and Inf values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data


def _resize_array(data: np.ndarray, target_shape: Tuple[int, int], 
                  preserve_aspect_ratio: bool = True) -> np.ndarray:
    """Resize 2D array using interpolation
    
    Args:
        data: Input 2D array
        target_shape: Target (height, width)
        preserve_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Resized array
    """
    from scipy.ndimage import zoom
    
    if preserve_aspect_ratio:
        # Calculate zoom factors
        h_ratio = target_shape[0] / data.shape[0]
        w_ratio = target_shape[1] / data.shape[1]
        zoom_factor = min(h_ratio, w_ratio)
        
        # Zoom
        resized = zoom(data, zoom_factor, order=1)
        
        # Pad if needed
        if resized.shape != target_shape:
            padded = np.zeros(target_shape)
            h_offset = (target_shape[0] - resized.shape[0]) // 2
            w_offset = (target_shape[1] - resized.shape[1]) // 2
            padded[h_offset:h_offset+resized.shape[0], 
                   w_offset:w_offset+resized.shape[1]] = resized
            resized = padded
    else:
        # Direct zoom to target shape
        zoom_factors = (target_shape[0] / data.shape[0], 
                       target_shape[1] / data.shape[1])
        resized = zoom(data, zoom_factors, order=1)
    
    return resized


# ============================================================================
# DatasetManager Class
# ============================================================================

class DatasetManager:
    """Manager class for AFM h5 datasets
    
    This class provides a high-level interface for:
    - Loading and parsing h5 files
    - Extracting scan regions
    - Accessing metadata
    - Visualizing raw data
    - Converting to SmartScan-compatible formats
    
    Example:
        >>> with DatasetManager('data/sample.h5') as dm:
        ...     metadata = dm.get_metadata()
        ...     regions = dm.get_scan_regions(num_regions=10)
        ...     dm.visualize_raw_data(save_path='results/raw.png')
    """
    
    def __init__(self, file_path: Optional[str] = None):
        """Initialize DatasetManager
        
        Args:
            file_path: Optional path to h5 file to load immediately
        """
        self.file_path = None
        self.h5_file = None
        self.metadata = {}
        self.available_channels = []
        self.afm_mode = 'unknown'
        self._raw_data = None
        
        if file_path:
            self.load_dataset(file_path)
    
    def load_dataset(self, file_path: str) -> None:
        """Load and parse h5 file structure
        
        Args:
            file_path: Path to h5 file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            CorruptedDataError: If file is corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not validate_h5_file(file_path):
            raise CorruptedDataError(f"File is corrupted or not a valid h5 file: {file_path}")
        
        # Close existing file if open
        self.close()
        
        try:
            self.h5_file = h5py.File(file_path, 'r')
            self.file_path = file_path
            
            # Extract metadata
            self.metadata = extract_metadata(self.h5_file)
            self.afm_mode = self.metadata.get('afm_mode', 'unknown')
            
            # Find available channels
            self.available_channels = self._find_channels()
            
            # Load raw data
            self._raw_data = extract_raw_data(self.h5_file)
            
            # Infer resolution if not in metadata
            if self.metadata.get('resolution') is None and self._raw_data is not None:
                if len(self._raw_data.shape) >= 2:
                    self.metadata['resolution'] = max(self._raw_data.shape[-2:])
            
            print(f"✅ Loaded dataset: {os.path.basename(file_path)}")
            print(f"   AFM Mode: {self.afm_mode}")
            print(f"   Resolution: {self.metadata.get('resolution', 'unknown')}")
            print(f"   Data shape: {self._raw_data.shape if self._raw_data is not None else 'N/A'}")
            print(f"   Channels: {len(self.available_channels)}")
            
        except Exception as e:
            self.close()
            raise H5DataLoaderError(f"Failed to load dataset: {e}")
    
    def _find_channels(self) -> List[str]:
        """Find all available data channels in the h5 file
        
        Returns:
            List of channel names/paths
        """
        channels = []
        
        def find_channel_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Look for datasets that look like channel data
                if any(keyword in name.lower() for keyword in 
                       ['channel', 'raw_data', 'topography', 'phase', 'amplitude']):
                    channels.append(name)
        
        if self.h5_file:
            self.h5_file.visititems(find_channel_datasets)
        
        return channels
    
    def get_scan_regions(self, num_regions: int = 10) -> List[np.ndarray]:
        """Extract spatial regions from scan data
        
        This divides the scan data into a grid of regions suitable for
        SmartScan adaptive scanning.
        
        Args:
            num_regions: Number of regions to extract
            
        Returns:
            List of numpy arrays, each representing a scan region
            
        Raises:
            H5DataLoaderError: If no data is loaded
        """
        if self._raw_data is None:
            raise H5DataLoaderError("No data loaded. Call load_dataset() first.")
        
        # Normalize data to 2D
        data_2d = normalize_data(self._raw_data)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_regions)))
        
        # Calculate region size
        height, width = data_2d.shape
        region_height = height // grid_size
        region_width = width // grid_size
        
        regions = []
        for i in range(num_regions):
            row = i // grid_size
            col = i % grid_size
            
            # Extract region
            y_start = row * region_height
            y_end = min((row + 1) * region_height, height)
            x_start = col * region_width
            x_end = min((col + 1) * region_width, width)
            
            if y_start < height and x_start < width:
                region = data_2d[y_start:y_end, x_start:x_end]
                
                # Normalize region size
                if region.shape[0] > 0 and region.shape[1] > 0:
                    region = normalize_data(region, target_shape=(region_height, region_width))
                    regions.append(region)
        
        return regions
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return comprehensive metadata dictionary
        
        Returns:
            Metadata dictionary with scan parameters
        """
        return self.metadata.copy()
    
    def get_channel_data(self, channel_name: str) -> np.ndarray:
        """Extract specific channel data
        
        Args:
            channel_name: Name or path of channel
            
        Returns:
            Channel data as numpy array
            
        Raises:
            MissingChannelError: If channel not found
        """
        if self.h5_file is None:
            raise H5DataLoaderError("No dataset loaded")
        
        # Try exact match first
        if channel_name in self.h5_file:
            return np.array(self.h5_file[channel_name][()])
        
        # Try partial match in available channels
        for channel in self.available_channels:
            if channel_name.lower() in channel.lower():
                return np.array(self.h5_file[channel][()])
        
        raise MissingChannelError(f"Channel not found: {channel_name}")
    
    def visualize_raw_data(self, channel: int = 0, save_path: Optional[str] = None,
                          cmap: str = 'viridis') -> None:
        """Plot raw topography/channel data
        
        Args:
            channel: Channel index (0 for primary/topography)
            save_path: Optional path to save figure
            cmap: Matplotlib colormap name
        """
        if self._raw_data is None:
            raise H5DataLoaderError("No data loaded")
        
        # Normalize to 2D for visualization
        data_2d = normalize_data(self._raw_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data_2d, cmap=cmap, aspect='equal')
        ax.set_title(f'AFM Raw Data - {os.path.basename(self.file_path or "Unknown")}\n'
                    f'Mode: {self.afm_mode}, Shape: {data_2d.shape}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Y Position (pixels)', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Height (a.u.)', fontsize=10)
        
        # Add grid
        ax.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def close(self) -> None:
        """Properly close h5 file handle"""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file is closed"""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - ensure file is closed"""
        self.close()


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Demonstrate h5_data_loader module usage"""
    print("\n" + "="*70)
    print("AFM H5 Data Loader - Example Usage")
    print("="*70)
    
    # Note: This example uses placeholder data since we don't have actual h5 files
    # Users should replace with their own h5 file paths
    
    print("\n📋 Available sample datasets:")
    datasets = get_sample_datasets()
    for i, ds in enumerate(datasets):
        print(f"   {i+1}. {ds['name']}")
        print(f"      Description: {ds['description']}")
        print(f"      Mode: {ds['mode']}")
        print(f"      Note: {ds.get('note', '')}\n")
    
    # Example: Create synthetic h5 file for demonstration
    print("📝 For actual usage, load your h5 file like this:")
    print("""
    # Option 1: Download sample data
    url = get_sample_datasets()[0]['url']
    local_path = download_sample_data(url, 'data/sample_afm.h5')
    
    # Option 2: Use existing file
    local_path = 'path/to/your/afm_data.h5'
    
    # Load and analyze
    with DatasetManager(local_path) as dm:
        # Inspect structure
        inspect_h5_structure(local_path)
        
        # Get metadata
        metadata = dm.get_metadata()
        print(f"Scan size: {metadata['scan_size']}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"AFM Mode: {metadata['afm_mode']}")
        
        # Visualize
        dm.visualize_raw_data(save_path='results/raw_topography.png')
        
        # Extract regions for SmartScan
        regions = dm.get_scan_regions(num_regions=10)
        
        # Integrate with SmartScan
        from src.vision_evaluator import VisionEvaluator
        evaluator = VisionEvaluator()
        
        for i, region in enumerate(regions):
            analysis = evaluator.analyze_region(region)
            print(f"Region {i}: Quality={analysis['quality']:.2f}")
    """)
    
    print("\n✅ Module loaded successfully!")
    print("   All functions and classes are ready to use.")
    print("   Install dependencies with: pip install h5py requests scipy")


if __name__ == "__main__":
    example_usage()
