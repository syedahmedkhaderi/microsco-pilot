"""
Vision module for Claude Vision API

This module handles communication with Claude Sonnet 4 Vision API
to analyze Atomic Force Microscopy (AFM) images and provide structured analysis.

TEACHING NOTES:
- Base64 encoding: Images are binary data, but APIs often need text. Base64 converts
  binary data (like images) into a text string that can be safely transmitted.
  Think of it like translating an image into a long string of letters and numbers.
  
- API calls: We send the image and a question to Claude, and it responds with analysis.
  This is like asking an expert scientist to look at your microscope image.
"""

import logging
import base64
import os
import time
import json
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import numpy as np
import random
import time

# Try to load .env so the API key is available when running this file directly
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # It's okay if python-dotenv is not installed; we will rely on environment variables.
    pass

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
    Uses Claude Vision API to analyze AFM microscope images and provide structured insights.
    
    This class:
    1. Loads images from files
    2. Converts them to base64 (required by API)
    3. Sends to Claude with specialized AFM analysis prompts
    4. Parses responses into structured dictionaries
    5. Handles errors and retries gracefully
    """
    
    def __init__(self, api_key=None, max_retries=3, retry_delay=2):
        """
        Initialize the vision analyzer.
        
        Args:
            api_key (str, optional): Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            max_retries (int): Maximum number of retry attempts for API calls (default: 3)
            retry_delay (float): Seconds to wait between retries (default: 2)
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
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            logger.info("Claude Vision API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert an image file to base64-encoded string.
        
        BASE64 ENCODING EXPLANATION:
        - Images are stored as binary data (0s and 1s)
        - APIs often need text format, not binary
        - Base64 converts binary → text using 64 characters (A-Z, a-z, 0-9, +, /)
        - Example: A small image might become "iVBORw0KGgoAAAANS..."
        - This text can be safely sent over the internet
        
        Args:
            image_path (str): Path to the image file (PNG, JPG, etc.)
        
        Returns:
            str: Base64-encoded image string
        """
        try:
            # Read the image file as binary data
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
            
            # Convert binary data to base64 string
            # b64encode() returns bytes, so we decode to get a string
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            logger.debug(f"Converted image {image_path} to base64 ({len(base64_string)} chars)")
            return base64_string
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            raise
    
    def _create_afm_analysis_prompt(self) -> str:
        """
        Create a specialized prompt for AFM image analysis.
        
        PROMPT ENGINEERING EXPLANATION:
        This is where the "intelligence" comes from! We tell Claude:
        1. What type of image it's looking at (AFM)
        2. What to look for (specific features)
        3. How to structure the response (what information we need)
        
        A good prompt is like giving clear instructions to an expert scientist.
        
        Returns:
            str: The prompt text for Claude
        """
        prompt = """You are an expert analyzing an Atomic Force Microscopy (AFM) image.

Please analyze this AFM image and provide a structured response with the following information:

1. FEATURES IDENTIFIED:
   - Grain boundaries: Are there visible grain boundaries? Describe their characteristics.
   - Defects: Any cracks, voids, or structural defects?
   - Particles: Are there particles or contaminants visible?
   - Rough regions: Areas with high surface roughness?
   - Smooth regions: Areas with low surface roughness?
   - Other notable features: Any other interesting structures?

2. IMAGE QUALITY (rate 0-10):
   - 0-3: Poor (blurry, artifacts, low contrast)
   - 4-6: Fair (some issues but usable)
   - 7-8: Good (clear, well-resolved)
   - 9-10: Excellent (very clear, high resolution)

3. INTERESTINGNESS ASSESSMENT:
   - Is there something worth investigating further? (Yes/No)
   - What makes it interesting? (Brief explanation)

4. NEXT ACTION SUGGESTION:
   - What should the operator do next?
   - Options: "zoom in", "move left", "move right", "move up", "move down", "adjust settings", "continue scanning"
   - Provide specific coordinates if suggesting movement (format: x=value, y=value)
   - Explain why this action would be helpful

5. CONFIDENCE SCORE (0-10):
   - How confident are you in your analysis?
   - 0-3: Low confidence (image unclear or ambiguous)
   - 4-6: Moderate confidence
   - 7-10: High confidence (clear features, good image quality)

6. REASONING:
   - Explain your analysis and how you reached your conclusions
   - What specific features in the image led to your assessment?

Please format your response clearly, using the numbered sections above."""
        
        return prompt
    
    def _parse_claude_response(self, response_text: str) -> Dict:
        """
        Parse Claude's text response into a structured dictionary.
        
        RESPONSE PARSING EXPLANATION:
        Claude returns text, but we want structured data (a dictionary).
        We look for patterns in the text to extract:
        - Numbers (like quality scores)
        - Keywords (like "zoom in", "move left")
        - Descriptions (feature lists)
        
        This is like reading a scientist's notes and organizing them into categories.
        
        Args:
            response_text (str): Raw text response from Claude
        
        Returns:
            dict: Structured analysis with keys:
                - features: dict of identified features
                - image_quality: int (0-10)
                - interestingness: dict with 'is_interesting' and 'reason'
                - next_action: dict with 'action' and 'explanation'
                - confidence: int (0-10)
                - reasoning: str
                - raw_response: str (original response)
        """
        # Initialize the result dictionary
        result = {
            'features': {},
            'image_quality': None,
            'interestingness': {'is_interesting': None, 'reason': None},
            'next_action': {'action': None, 'explanation': None, 'coordinates': None},
            'confidence': None,
            'reasoning': None,
            'raw_response': response_text
        }
        
        try:
            # Extract image quality score (look for "IMAGE QUALITY" or "quality" followed by a number)
            import re
            
            # Look for quality score (0-10)
            quality_match = re.search(r'(?:quality|IMAGE QUALITY).*?(\d+)(?:\s*/\s*10)?', 
                                      response_text, re.IGNORECASE)
            if quality_match:
                try:
                    result['image_quality'] = int(quality_match.group(1))
                except ValueError:
                    pass
            
            # Look for confidence score
            confidence_match = re.search(r'(?:confidence|CONFIDENCE).*?(\d+)(?:\s*/\s*10)?', 
                                         response_text, re.IGNORECASE)
            if confidence_match:
                try:
                    result['confidence'] = int(confidence_match.group(1))
                except ValueError:
                    pass
            
            # Extract interestingness (look for "Yes" or "No" after "interesting")
            interesting_match = re.search(r'(?:interesting|INTERESTINGNESS).*?(yes|no)', 
                                          response_text, re.IGNORECASE)
            if interesting_match:
                result['interestingness']['is_interesting'] = interesting_match.group(1).lower() == 'yes'
            
            # Extract next action (look for action keywords)
            action_keywords = ['zoom in', 'move left', 'move right', 'move up', 'move down', 
                             'adjust settings', 'continue scanning']
            for keyword in action_keywords:
                if keyword.lower() in response_text.lower():
                    result['next_action']['action'] = keyword
                    break
            
            # Extract coordinates if mentioned (format: x=value, y=value)
            coord_match = re.search(r'x\s*[=:]\s*([0-9.-]+).*?y\s*[=:]\s*([0-9.-]+)', 
                                   response_text, re.IGNORECASE)
            if coord_match:
                try:
                    result['next_action']['coordinates'] = (
                        float(coord_match.group(1)),
                        float(coord_match.group(2))
                    )
                except ValueError:
                    pass
            
            # Extract features (look for section headers)
            feature_section = re.search(r'FEATURES.*?(?=IMAGE QUALITY|INTERESTINGNESS|$)', 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if feature_section:
                feature_text = feature_section.group(0)
                # Look for common feature keywords
                features_found = []
                feature_types = ['grain boundaries', 'defects', 'particles', 
                               'rough regions', 'smooth regions']
                for feature_type in feature_types:
                    if feature_type.lower() in feature_text.lower():
                        features_found.append(feature_type)
                result['features'] = {'types': features_found, 'description': feature_text[:500]}
            
            # Extract reasoning section
            reasoning_match = re.search(r'(?:REASONING|reasoning).*?(?=FEATURES|IMAGE|INTERESTINGNESS|$)', 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(0)[:1000]  # Limit length
            
            logger.info("Parsed Claude response successfully")
            
        except Exception as e:
            logger.warning(f"Error parsing response, using raw response: {e}")
            # If parsing fails, we still have the raw response
        
        return result
    
    def _call_claude_api(self, base64_image: str, prompt: str) -> str:
        """
        Make a single API call to Claude Vision API.
        
        API CALL FORMAT EXPLANATION:
        We send Claude a "message" with:
        1. The image (as base64 text)
        2. Our question/prompt (as text)
        
        Claude responds with text analysis. The format is:
        {
            "content": [{"text": "Claude's response here..."}]
        }
        
        Args:
            base64_image (str): Base64-encoded image
            prompt (str): The prompt/question for Claude
        
        Returns:
            str: Claude's response text
        
        Raises:
            Exception: If API call fails
        """
        try:
            # Prepare the message for Claude
            # Using Claude 3.5 Sonnet which supports vision
            # Note: For free tier, be mindful of rate limits!
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet with vision support
                max_tokens=1500,  # Allow enough tokens for detailed response
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",  # PNG format
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Extract the response text
            # Claude returns content as a list, first item is usually the text
            response_text = message.content[0].text
            
            return response_text
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an AFM image using Claude Vision API with retry logic.
        
        This is the main function you'll use! It:
        1. Loads the image from file
        2. Converts to base64
        3. Sends to Claude with specialized AFM prompt
        4. Parses the response into a structured dictionary
        5. Retries if the API call fails
        
        RETRY LOGIC EXPLANATION:
        - Sometimes APIs fail (network issues, rate limits, etc.)
        - We try up to 3 times
        - Wait 2 seconds between tries (don't spam the API!)
        - This makes the code more robust
        
        Args:
            image_path (str): Path to the image file to analyze
        
        Returns:
            dict: Structured analysis with keys:
                - features: dict of identified features
                - image_quality: int (0-10)
                - interestingness: dict with 'is_interesting' and 'reason'
                - next_action: dict with 'action' and 'explanation'
                - confidence: int (0-10)
                - reasoning: str
                - raw_response: str (original response)
                - success: bool (whether analysis succeeded)
                - error: str (error message if failed)
        
        Example:
            >>> analyzer = VisionAnalyzer()
            >>> result = analyzer.analyze_image("outputs/image_0.5_0.5.png")
            >>> print(result['image_quality'])
            8
            >>> print(result['next_action']['action'])
            "zoom in"
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Check if file exists
        if not Path(image_path).exists():
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
        
        # Convert image to base64
        try:
            base64_image = self._image_to_base64(image_path)
        except Exception as e:
            error_msg = f"Failed to load image: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
        
        # Get the specialized AFM analysis prompt
        prompt = self._create_afm_analysis_prompt()
        
        # Try API call with retry logic
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"API call attempt {attempt}/{self.max_retries}")
                
                # Make the API call
                response_text = self._call_claude_api(base64_image, prompt)
                
                # Parse the response
                result = self._parse_claude_response(response_text)
                result['success'] = True
                result['error'] = None
                
                # Log the analysis to a file
                self._log_analysis(image_path, result)
                
                logger.info(f"Analysis successful! Quality: {result.get('image_quality', 'N/A')}, "
                          f"Confidence: {result.get('confidence', 'N/A')}")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")
                
                # Wait before retrying (except on last attempt)
                if attempt < self.max_retries:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
        
        # All retries failed
        error_msg = f"Failed after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'raw_response': None
        }
    
    def _log_analysis(self, image_path: str, result: Dict):
        """
        Log the analysis to a file for later review.
        
        Args:
            image_path (str): Path to the analyzed image
            result (dict): Analysis result dictionary
        """
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("outputs")
            log_dir.mkdir(exist_ok=True, parents=True)
            
            log_file = log_dir / "vision_analysis.log"
            
            # Append to log file
            with open(log_file, 'a') as f:
                log_entry = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'image_path': str(image_path),
                    'image_quality': result.get('image_quality'),
                    'confidence': result.get('confidence'),
                    'interestingness': result.get('interestingness', {}).get('is_interesting'),
                    'next_action': result.get('next_action', {}).get('action'),
                    'features_found': result.get('features', {}).get('types', [])
                }
                f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"Analysis logged to {log_file}")
            
        except Exception as e:
            logger.warning(f"Failed to log analysis: {e}")


def test_analyze_image():
    """
    Test function to analyze a sample image.
    
    This function demonstrates how to use the VisionAnalyzer class.
    Run this to test if everything is working correctly.
    
    Usage:
        python -c "from src.vision import test_analyze_image; test_analyze_image()"
    """
    print("=" * 60)
    print("Testing Vision Analyzer")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set!")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key'")
        return
    
    # Initialize analyzer
    try:
        analyzer = VisionAnalyzer()
        print("✓ VisionAnalyzer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Look for a test image
    # Check if there are any images in the outputs directory
    test_image_paths = [
        "outputs/image_0.5_0.5.png",
        "data/sample_image.png",
        "../DTMicroscope/assets/graphene1.png"
    ]
    
    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image = path
            break
    
    if not test_image:
        print("\nNo test image found. Please provide an image path:")
        print("Available options:")
        for path in test_image_paths:
            print(f"  - {path}")
        print("\nOr create a test image first by running the agent.")
        return
    
    print(f"\nAnalyzing test image: {test_image}")
    print("This may take a few seconds...")
    print()
    
    # Analyze the image
    try:
        result = analyzer.analyze_image(test_image)
        
        if result['success']:
            print("✓ Analysis successful!")
            print()
            print("RESULTS:")
            print("-" * 60)
            print(f"Image Quality: {result.get('image_quality', 'N/A')}/10")
            print(f"Confidence: {result.get('confidence', 'N/A')}/10")
            print(f"Interesting: {result.get('interestingness', {}).get('is_interesting', 'N/A')}")
            print(f"Next Action: {result.get('next_action', {}).get('action', 'N/A')}")
            print()
            print("Features Found:")
            features = result.get('features', {}).get('types', [])
            if features:
                for feature in features:
                    print(f"  - {feature}")
            else:
                print("  (No specific features identified)")
            print()
            print("Raw Response (first 200 chars):")
            print(result.get('raw_response', '')[:200] + "...")
            print()
            print("=" * 60)
            print("Test completed successfully!")
        else:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


class MockVisionAnalyzer:
    """Simulated vision analyzer - NO API needed!"""

    def analyze_image(self, image_path: str) -> Dict:
        """Return fake but realistic analysis in the same schema used by VisionAnalyzer."""
        # Simulate latency similar to an API call
        time.sleep(0.5)

        # Randomized feature sets and suggestions
        feature_types = random.choice([
            ['grain_boundary', 'smooth_region'],
            ['particles', 'defect'],
            ['rough_region', 'grain_boundary'],
            ['smooth_region'],
            ['particles', 'rough_region', 'defect']
        ])
        quality = random.uniform(5, 9)
        interesting_score = random.uniform(4, 9)
        suggestion = random.choice([
            'zoom_in',
            'move_left',
            'move_right',
            'move_up',
            'move_down',
            'adjust_focus'
        ])

        is_interesting = interesting_score >= 6.5
        next_action = {
            'action': suggestion,
            'explanation': 'Mock suggestion based on simulated feature detection',
        }

        return {
            'success': True,
            'features': {
                'types': feature_types
            },
            'image_quality': int(round(quality)),
            'interestingness': {
                'is_interesting': is_interesting,
                'reason': f"mock score {interesting_score:.1f}/10",
            },
            'next_action': next_action,
            'confidence': int(round(quality)),
            'reasoning': f"Detected {len(feature_types)} features. Quality: {quality:.1f}/10. Suggesting: {suggestion}",
            'raw_response': 'mock',
        }


if __name__ == "__main__":
    import argparse

    # Set up basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test the VisionAnalyzer on an image file."
    )
    parser.add_argument(
        "--test",
        dest="test_image",
        type=str,
        help="Path to an image file to analyze (e.g., data/sample_afm_image.png)",
    )
    args = parser.parse_args()

    def _generate_sample_afm_image(output_path: str, image_size: int = 256):
        """Generate a synthetic AFM-like image and save to PNG."""
        try:
            rng = np.random.default_rng(1234)
            base = rng.normal(loc=0.0, scale=0.15, size=(image_size, image_size))
            cx = rng.integers(0, image_size)
            cy = rng.integers(0, image_size)
            sigma = rng.uniform(12, 28)
            xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
            bump = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
            img_arr = base + bump * rng.uniform(0.6, 1.3)
            # Normalize 0-255
            img_arr = img_arr - img_arr.min()
            img_arr = (img_arr / (img_arr.max() + 1e-8) * 255.0).astype(np.uint8)
            Image.fromarray(img_arr).save(output_path)
            return True
        except Exception as e:
            print(f"ERROR: Failed to generate sample image: {e}")
            return False

    def _local_analyze_image(image_path: str) -> Dict:
        """Local heuristic analysis without calling external API."""
        try:
            img = Image.open(image_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            var = float(np.var(arr))
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            quality = int(np.clip(var * 10, 0, 10))
            is_interesting = quality > 7
            next_action = {
                'action': 'zoom in' if is_interesting else 'continue scanning',
                'explanation': 'Heuristic based on image variance',
            }
            return {
                'success': True,
                'features': {'heuristic': {'variance': var, 'mean': mean, 'std': std}},
                'image_quality': quality,
                'interestingness': {'is_interesting': is_interesting, 'reason': 'variance proxy'},
                'next_action': next_action,
                'confidence': quality,
                'reasoning': 'Local analysis without external API',
                'raw_response': None,
            }
        except Exception as e:
            return {'success': False, 'error': f'Local analysis failed: {e}', 'raw_response': None}

    # If a specific image is provided, ensure it exists and analyze
    if args.test_image:
        img_path = Path(args.test_image)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        if not img_path.exists():
            ok = _generate_sample_afm_image(str(img_path))
            if ok:
                print(f"Generated sample AFM image at {img_path}")
            else:
                print("ERROR: Could not create sample image.")
                raise SystemExit(1)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                analyzer = VisionAnalyzer(api_key=api_key)
                result = analyzer.analyze_image(str(img_path))
                if not result.get('success'):
                    print("API analysis failed, using local heuristic analysis.")
                    result = _local_analyze_image(str(img_path))
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"API call error: {e}. Using local heuristic analysis.")
                result = _local_analyze_image(str(img_path))
                print(json.dumps(result, indent=2))
        else:
            print("No ANTHROPIC_API_KEY found. Running local heuristic analysis.")
            result = _local_analyze_image(str(img_path))
            print(json.dumps(result, indent=2))
    else:
        # Fallback to built-in guided test flow
        test_analyze_image()
