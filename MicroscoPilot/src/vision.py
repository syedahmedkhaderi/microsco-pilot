"""
Vision module for Claude Vision API

This module handles communication with Claude Sonnet 4 Vision API
to analyze microscope images and make decisions.
"""

import logging
import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Import Anthropic SDK for Claude API
try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError(
        "Anthropic SDK not found! Please install it:\n"
        "pip install anthropic"
    )

# Set up logging
logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """
    Uses Claude Vision API to analyze microscope images and provide insights.
    
    This class converts numpy arrays (images) to a format Claude can understand,
    sends them to the API, and returns the analysis.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the vision analyzer.
        
        Args:
            api_key (str, optional): Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if api_key is None:
            raise ValueError(
                "API key not found! Please set ANTHROPIC_API_KEY environment variable "
                "or pass it as a parameter."
            )
        
        # Initialize the Anthropic client
        try:
            self.client = Anthropic(api_key=api_key)
            logger.info("Claude Vision API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    def _array_to_base64(self, image_array):
        """
        Convert a numpy array to a base64-encoded image string.
        
        This is needed because Claude API expects images as base64 strings.
        
        Args:
            image_array (numpy.ndarray): Image data as numpy array
            
        Returns:
            str: Base64-encoded image string
        """
        try:
            # Handle different array shapes
            # If it's 3D (multiple channels), take the first channel
            if len(image_array.shape) == 3:
                # Shape might be (channels, height, width) or (height, width, channels)
                if image_array.shape[0] < image_array.shape[2]:
                    # Likely (channels, height, width)
                    image_data = image_array[0]
                else:
                    # Likely (height, width, channels)
                    image_data = image_array[:, :, 0]
            elif len(image_array.shape) == 2:
                # Already 2D
                image_data = image_array
            else:
                raise ValueError(f"Unexpected image shape: {image_array.shape}")
            
            # Normalize to 0-255 range if needed
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_data, mode='L')  # 'L' = grayscale
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to convert array to base64: {e}")
            raise
    
    def analyze_image(self, image_array, question="What do you see in this microscope image?"):
        """
        Send an image to Claude Vision API and get analysis.
        
        Args:
            image_array (numpy.ndarray): Image data from the microscope
            question (str): Question to ask Claude about the image
        
        Returns:
            str: Claude's response/analysis
        """
        logger.info("Analyzing image with Claude Vision API...")
        
        try:
            # Convert numpy array to base64
            base64_image = self._array_to_base64(image_array)
            
            # Prepare the message for Claude
            # Using Claude 3.5 Sonnet which supports vision
            # You can change this to other vision-capable models like:
            # - "claude-3-opus-20240229"
            # - "claude-3-sonnet-20240229"
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet with vision support
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ]
            )
            
            # Extract the response text
            response_text = message.content[0].text
            
            logger.info("Received analysis from Claude")
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            raise
    
    def get_exploration_suggestion(self, image_array, current_position, bounds):
        """
        Ask Claude to suggest where to explore next based on the current image.
        
        Args:
            image_array (numpy.ndarray): Current image
            current_position (tuple): Current (x, y) position
            bounds (dict): Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
        
        Returns:
            str: Claude's suggestion for next action
        """
        question = (
            f"You are analyzing a microscope image taken at position ({current_position[0]:.4f}, {current_position[1]:.4f}). "
            f"The scanning area bounds are: x=[{bounds['x_min']:.4f}, {bounds['x_max']:.4f}], "
            f"y=[{bounds['y_min']:.4f}, {bounds['y_max']:.4f}]. "
            f"Based on what you see, suggest where to move next to explore interesting features. "
            f"Provide specific x, y coordinates and explain why."
        )
        
        return self.analyze_image(image_array, question)
    
    def identify_features(self, image_array):
        """
        Ask Claude to identify interesting features in the image.
        
        Args:
            image_array (numpy.ndarray): Image to analyze
        
        Returns:
            str: Description of identified features
        """
        question = (
            "Identify any interesting features in this microscope image. "
            "Describe what you see: structures, patterns, anomalies, or anything noteworthy."
        )
        
        return self.analyze_image(image_array, question)

