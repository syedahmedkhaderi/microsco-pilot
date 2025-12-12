"""
Memory module for tracking discoveries

This module stores information about what the agent has seen,
where it has been, and what features it has discovered.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class DiscoveryMemory:
    """
    Stores and manages the agent's memory of its exploration.
    
    This keeps track of:
    - Positions visited
    - Images captured
    - Features discovered
    - Analysis results from Claude
    """
    
    def __init__(self, save_dir="outputs"):
        """
        Initialize the memory system.
        
        Args:
            save_dir (str): Directory to save memory data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # List to store all discoveries
        self.discoveries: List[Dict] = []
        
        # List of positions visited (x, y)
        self.positions_visited: List[Tuple[float, float]] = []
        
        # Counter for unique discovery IDs
        self.discovery_counter = 0
        
        logger.info(f"Memory initialized. Save directory: {self.save_dir}")
    
    def add_discovery(self, position, image_data, analysis, feature_description=None):
        """
        Add a new discovery to memory.
        
        Args:
            position (tuple): (x, y) position where image was taken
            image_data (numpy.ndarray): Image data captured
            analysis (str): Analysis from Claude Vision API
            feature_description (str, optional): Description of features found
        """
        discovery = {
            'id': self.discovery_counter,
            'timestamp': datetime.now().isoformat(),
            'position': {
                'x': float(position[0]),
                'y': float(position[1])
            },
            'analysis': analysis,
            'feature_description': feature_description,
            'image_shape': list(image_data.shape),
            'image_mean': float(np.mean(image_data)),
            'image_std': float(np.std(image_data)),
            'image_min': float(np.min(image_data)),
            'image_max': float(np.max(image_data))
        }
        
        self.discoveries.append(discovery)
        self.positions_visited.append(position)
        self.discovery_counter += 1
        
        logger.info(f"Added discovery #{self.discovery_counter} at position {position}")
    
    def has_visited(self, position, tolerance=1e-6):
        """
        Check if we've already visited a position (within tolerance).
        
        Args:
            position (tuple): (x, y) position to check
            tolerance (float): Distance tolerance for considering positions the same
        
        Returns:
            bool: True if position has been visited
        """
        x, y = position
        for visited_x, visited_y in self.positions_visited:
            distance = np.sqrt((x - visited_x)**2 + (y - visited_y)**2)
            if distance < tolerance:
                return True
        return False
    
    def get_summary(self):
        """
        Get a summary of all discoveries.
        
        Returns:
            dict: Summary statistics
        """
        if not self.discoveries:
            return {
                'total_discoveries': 0,
                'positions_visited': 0,
                'message': 'No discoveries yet'
            }
        
        return {
            'total_discoveries': len(self.discoveries),
            'positions_visited': len(self.positions_visited),
            'unique_positions': len(set(self.positions_visited)),
            'first_discovery': self.discoveries[0]['timestamp'],
            'last_discovery': self.discoveries[-1]['timestamp']
        }
    
    def save(self, filename="memory.json"):
        """
        Save memory to a JSON file.
        
        Args:
            filename (str): Name of the file to save
        """
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'discoveries': self.discoveries,
                    'summary': self.get_summary()
                }, f, indent=2)
            
            logger.info(f"Memory saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise
    
    def load(self, filename="memory.json"):
        """
        Load memory from a JSON file.
        
        Args:
            filename (str): Name of the file to load
        """
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Memory file {filepath} not found. Starting fresh.")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.discoveries = data.get('discoveries', [])
            
            # Reconstruct positions_visited
            self.positions_visited = [
                (d['position']['x'], d['position']['y'])
                for d in self.discoveries
            ]
            
            # Update counter
            if self.discoveries:
                self.discovery_counter = max(d['id'] for d in self.discoveries) + 1
            
            logger.info(f"Memory loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            raise
    
    def get_recent_discoveries(self, n=5):
        """
        Get the most recent discoveries.
        
        Args:
            n (int): Number of recent discoveries to return
        
        Returns:
            list: List of recent discovery dictionaries
        """
        return self.discoveries[-n:] if len(self.discoveries) >= n else self.discoveries

