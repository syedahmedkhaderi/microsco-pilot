"""
Agent module - Main autonomous exploration logic

This module contains the MicroscoPilot agent that makes decisions
about where to explore and what to investigate.
"""

import logging
import random
import re
from typing import Tuple, Optional

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class MicroscoPilotAgent:
    """
    The main autonomous agent for microscopy exploration.
    
    This agent:
    1. Decides where to move next
    2. Captures images
    3. Analyzes images with Claude Vision API
    4. Tracks discoveries
    5. Makes exploration decisions
    """
    
    def __init__(self, microscope, vision_analyzer, memory, visualizer):
        """
        Initialize the agent.
        
        Args:
            microscope: MicroscopeWrapper instance
            vision_analyzer: VisionAnalyzer instance
            memory: DiscoveryMemory instance
            visualizer: Visualizer instance
        """
        self.microscope = microscope
        self.vision = vision_analyzer
        self.memory = memory
        self.visualizer = visualizer
        
        # Exploration parameters
        self.step_size = 0.1  # How far to move in each step (as fraction of range)
        self.max_steps = 20   # Maximum number of exploration steps
        
        logger.info("MicroscoPilot agent initialized")
    
    def extract_coordinates_from_text(self, text):
        """
        Try to extract x, y coordinates from Claude's text response.
        
        This is a simple parser - Claude might say things like:
        "Move to x=0.5, y=0.3" or "coordinates (0.5, 0.3)"
        
        Args:
            text (str): Text response from Claude
        
        Returns:
            tuple or None: (x, y) coordinates if found, None otherwise
        """
        # Try to find patterns like "x=0.5, y=0.3" or "(0.5, 0.3)"
        patterns = [
            r'x\s*[=:]\s*([0-9.-]+).*?y\s*[=:]\s*([0-9.-]+)',
            r'\(([0-9.-]+)\s*,\s*([0-9.-]+)\)',
            r'coordinates?\s*[:\s]+([0-9.-]+)\s*,\s*([0-9.-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    return (x, y)
                except ValueError:
                    continue
        
        return None
    
    def choose_next_position(self, current_position, bounds, claude_suggestion=None):
        """
        Choose the next position to explore.
        
        This is a simple strategy:
        1. If Claude suggests coordinates, try to use them
        2. Otherwise, pick a random unexplored position
        
        Args:
            current_position (tuple): Current (x, y) position
            bounds (dict): Scanning bounds
            claude_suggestion (str, optional): Claude's suggestion text
        
        Returns:
            tuple: (x, y) coordinates for next position
        """
        logger.info("Choosing next exploration position...")
        
        # Try to extract coordinates from Claude's suggestion
        if claude_suggestion:
            coords = self.extract_coordinates_from_text(claude_suggestion)
            if coords:
                x, y = coords
                # Clamp to bounds
                x = max(bounds['x_min'], min(bounds['x_max'], x))
                y = max(bounds['y_min'], min(bounds['y_max'], y))
                
                # Check if we've already visited (approximately)
                if not self.memory.has_visited((x, y), tolerance=0.01):
                    logger.info(f"Using Claude's suggestion: ({x:.4f}, {y:.4f})")
                    return (x, y)
                else:
                    logger.info("Claude's suggestion already visited, trying random position")
        
        # Random exploration strategy
        # Pick a random direction and distance
        x_range = bounds['x_max'] - bounds['x_min']
        y_range = bounds['y_max'] - bounds['y_min']
        
        # Try up to 10 times to find an unvisited position
        for _ in range(10):
            # Random step in a random direction
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0.1, 0.3) * min(x_range, y_range)
            
            new_x = current_position[0] + distance * np.cos(angle)
            new_y = current_position[1] + distance * np.sin(angle)
            
            # Clamp to bounds
            new_x = max(bounds['x_min'], min(bounds['x_max'], new_x))
            new_y = max(bounds['y_min'], min(bounds['y_max'], new_y))
            
            # Check if we've visited this position
            if not self.memory.has_visited((new_x, new_y), tolerance=0.01):
                logger.info(f"Chose random position: ({new_x:.4f}, {new_y:.4f})")
                return (new_x, new_y)
        
        # If all nearby positions visited, pick any random position
        new_x = random.uniform(bounds['x_min'], bounds['x_max'])
        new_y = random.uniform(bounds['y_min'], bounds['y_max'])
        logger.info(f"All nearby positions visited, picking random: ({new_x:.4f}, {new_y:.4f})")
        return (new_x, new_y)
    
    def explore_step(self):
        """
        Perform one step of exploration:
        1. Get current position
        2. Capture image
        3. Analyze with Claude
        4. Store discovery
        5. Choose next position
        
        Returns:
            bool: True if exploration should continue, False if done
        """
        logger.info("=" * 60)
        logger.info("Starting exploration step")
        
        # Get current position
        current_pos = self.microscope.get_position()
        logger.info(f"Current position: {current_pos}")
        
        # Capture image
        try:
            image_data = self.microscope.capture_image()
            logger.info("Image captured successfully")
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return False
        
        # Save visualization of the image
        self.visualizer.plot_image(image_data, current_pos)
        
        # Analyze with Claude Vision API
        try:
            # First, get general analysis
            analysis = self.vision.identify_features(image_data)
            logger.info(f"Claude analysis: {analysis[:100]}...")  # Log first 100 chars
            
            # Get exploration suggestion
            bounds = self.microscope.get_bounds()
            suggestion = self.vision.get_exploration_suggestion(
                image_data, current_pos, bounds
            )
            logger.info(f"Claude suggestion: {suggestion[:100]}...")
            
        except Exception as e:
            logger.error(f"Failed to analyze image with Claude: {e}")
            # Continue anyway with basic analysis
            analysis = "Analysis failed"
            suggestion = None
        
        # Store discovery
        self.memory.add_discovery(
            position=current_pos,
            image_data=image_data,
            analysis=analysis,
            feature_description=suggestion
        )
        
        # Check if we should continue
        if len(self.memory.discoveries) >= self.max_steps:
            logger.info(f"Reached maximum steps ({self.max_steps})")
            return False
        
        # Choose next position
        bounds = self.microscope.get_bounds()
        next_pos = self.choose_next_position(current_pos, bounds, suggestion)
        
        # Move to next position
        try:
            self.microscope.move_to(next_pos[0], next_pos[1])
            logger.info(f"Moved to new position: {next_pos}")
        except Exception as e:
            logger.error(f"Failed to move: {e}")
            return False
        
        return True
    
    def run_exploration(self):
        """
        Run the full autonomous exploration loop.
        
        This continues until max_steps is reached or an error occurs.
        """
        logger.info("=" * 60)
        logger.info("Starting autonomous exploration")
        logger.info(f"Maximum steps: {self.max_steps}")
        logger.info("=" * 60)
        
        step_count = 0
        
        while step_count < self.max_steps:
            step_count += 1
            logger.info(f"\n--- Step {step_count}/{self.max_steps} ---")
            
            # Perform exploration step
            should_continue = self.explore_step()
            
            if not should_continue:
                break
        
        # Save memory
        self.memory.save()
        
        # Create visualizations
        bounds = self.microscope.get_bounds()
        self.visualizer.plot_exploration_map(self.memory, bounds)
        self.visualizer.create_summary_plot(self.memory)
        
        # Print summary
        summary = self.memory.get_summary()
        logger.info("=" * 60)
        logger.info("Exploration complete!")
        logger.info(f"Total discoveries: {summary['total_discoveries']}")
        logger.info(f"Positions visited: {summary['positions_visited']}")
        logger.info("=" * 60)

