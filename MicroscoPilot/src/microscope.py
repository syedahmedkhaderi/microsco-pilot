"""
Microscope wrapper module

This module wraps the DTMicroscope AFM digital twin to provide
a simple interface for the agent to interact with the microscope.
"""

import logging
import numpy as np
from pathlib import Path

# Import DTMicroscope classes
try:
    from DTMicroscope.base.afm import AFM_Microscope
except ImportError:
    raise ImportError(
        "DTMicroscope not found! Please install it first.\n"
        "You can install it with: pip install DTMicroscope"
    )

# Set up logging for this module
logger = logging.getLogger(__name__)


class MicroscopeWrapper:
    """
    A simple wrapper around DTMicroscope's AFM_Microscope class.
    
    This makes it easier for beginners to use the microscope without
    needing to understand all the details of the DTMicroscope API.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the microscope wrapper.
        
        Args:
            data_path (str, optional): Path to the dataset file (.h5 format).
                                     If None, will try to use default data.
        """
        logger.info("Initializing microscope...")
        
        try:
            # Create the AFM microscope instance
            self.microscope = AFM_Microscope(data_path=data_path)
            logger.info("Microscope created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize microscope: {e}")
            raise
        
        # Store current position
        self.current_x = None
        self.current_y = None
        
        # Store available channels
        self.available_channels = []
        
    def setup(self, data_source='Compound_Dataset_1', dset_subset=None):
        """
        Set up the microscope with a dataset.
        
        Args:
            data_source (str): Name of the dataset to use
            dset_subset (str, optional): Subset of the dataset if it's compound
        """
        logger.info(f"Setting up microscope with data source: {data_source}")
        
        try:
            # Set up the microscope
            self.microscope.setup_microscope(
                data_source=data_source,
                dset_subset=dset_subset
            )
            
            # Get information about available channels
            info = self.microscope.get_dataset_info()
            
            # Extract channel names from the info
            # info is a list of tuples like [('channels', [...]), ...]
            for item in info:
                if item[0] == 'channels':
                    self.available_channels = item[1]
                    break
            
            # Get current position (microscope starts at center)
            self.current_x = self.microscope.x
            self.current_y = self.microscope.y
            
            logger.info(f"Microscope setup complete. Available channels: {self.available_channels}")
            logger.info(f"Current position: ({self.current_x:.4f}, {self.current_y:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to setup microscope: {e}")
            raise
    
    def get_position(self):
        """
        Get the current position of the probe.
        
        Returns:
            tuple: (x, y) coordinates
        """
        self.current_x = self.microscope.x
        self.current_y = self.microscope.y
        return (self.current_x, self.current_y)
    
    def move_to(self, x, y):
        """
        Move the probe to a new position.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            
        Returns:
            tuple: The actual position after moving (may be clamped to valid range)
        """
        logger.info(f"Moving probe to ({x:.4f}, {y:.4f})")
        
        try:
            # Move the probe
            self.microscope.go_to(x, y)
            
            # Get the actual position (might be clamped)
            actual_x = self.microscope.x
            actual_y = self.microscope.y
            
            self.current_x = actual_x
            self.current_y = actual_y
            
            if actual_x != x or actual_y != y:
                logger.warning(
                    f"Position clamped: requested ({x:.4f}, {y:.4f}), "
                    f"actual ({actual_x:.4f}, {actual_y:.4f})"
                )
            
            return (actual_x, actual_y)
            
        except Exception as e:
            logger.error(f"Failed to move probe: {e}")
            raise
    
    def capture_image(self, channels=None, scan_rate=0.5):
        """
        Capture an image at the current position.
        
        Args:
            channels (list, optional): List of channel names to capture.
                                     If None, captures all available channels.
            scan_rate (float): Scanning rate (Hz). Lower is slower but more accurate.
        
        Returns:
            numpy.ndarray: Image data as a numpy array
        """
        logger.info(f"Capturing image at ({self.current_x:.4f}, {self.current_y:.4f})")
        
        try:
            # If no channels specified, use all available image channels
            if channels is None:
                channels = self.available_channels
            
            # Get the scan data
            image_data = self.microscope.get_scan(
                channels=channels,
                scan_rate=scan_rate
            )
            
            logger.info(f"Image captured: shape {image_data.shape}")
            return image_data
            
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            raise
    
    def get_bounds(self):
        """
        Get the valid scanning bounds.
        
        Returns:
            dict: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
        """
        return {
            'x_min': float(self.microscope.x_coords.min()),
            'x_max': float(self.microscope.x_coords.max()),
            'y_min': float(self.microscope.y_coords.min()),
            'y_max': float(self.microscope.y_coords.max())
        }
    
    def get_info(self):
        """
        Get information about the microscope and dataset.
        
        Returns:
            list: List of tuples with dataset information
        """
        return self.microscope.get_dataset_info()

