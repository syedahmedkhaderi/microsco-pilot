"""
Agent module - Main autonomous exploration logic

This module contains autonomous agents that make decisions
about where to explore and what to investigate.

TEACHING MOMENT:
The agent is just a while loop! It keeps running until you tell it to stop.
Each loop: sense the world (vision) → think (decide_next_action) → act (move microscope)
"""

import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

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
        
        # Save visualization of the image and get the filepath
        image_filepath = self.visualizer.plot_image(image_data, current_pos)
        
        # Analyze with Claude Vision API using the saved image file
        analysis_result = None
        analysis_text = "Analysis not performed"
        suggestion_text = None
        
        try:
            # Use the new analyze_image method which takes a file path
            analysis_result = self.vision.analyze_image(image_filepath)
            
            if analysis_result.get('success'):
                # Extract information from structured response
                analysis_text = analysis_result.get('raw_response', 'No analysis available')
                
                # Get next action suggestion
                next_action = analysis_result.get('next_action', {})
                suggestion_text = next_action.get('action', 'No suggestion')
                
                # Log the analysis results
                quality = analysis_result.get('image_quality', 'N/A')
                confidence = analysis_result.get('confidence', 'N/A')
                interesting = analysis_result.get('interestingness', {}).get('is_interesting', 'N/A')
                
                logger.info(f"Claude analysis - Quality: {quality}/10, Confidence: {confidence}/10, "
                          f"Interesting: {interesting}")
                logger.info(f"Next action suggested: {suggestion_text}")
            else:
                error_msg = analysis_result.get('error', 'Unknown error')
                logger.warning(f"Analysis failed: {error_msg}")
                analysis_text = f"Analysis failed: {error_msg}"
            
        except Exception as e:
            logger.error(f"Failed to analyze image with Claude: {e}")
            # Continue anyway with basic analysis
            analysis_text = f"Analysis error: {e}"
        
        # Store discovery
        self.memory.add_discovery(
            position=current_pos,
            image_data=image_data,
            analysis=analysis_text,
            feature_description=suggestion_text
        )
        
        # Check if we should continue
        if len(self.memory.discoveries) >= self.max_steps:
            logger.info(f"Reached maximum steps ({self.max_steps})")
            return False
        
        # Choose next position
        # Try to get coordinates from Claude's structured response
        claude_coords = None
        claude_suggestion_text = None
        if analysis_result and analysis_result.get('success'):
            next_action = analysis_result.get('next_action', {})
            claude_coords = next_action.get('coordinates')
            if claude_coords:
                claude_suggestion_text = f"Move to x={claude_coords[0]}, y={claude_coords[1]}"
            else:
                claude_suggestion_text = next_action.get('action', 'Continue exploring')
        
        bounds = self.microscope.get_bounds()
        # If Claude provided coordinates, use them (but clamp to bounds)
        if claude_coords:
            x, y = claude_coords
            # Clamp to valid bounds
            x = max(bounds['x_min'], min(bounds['x_max'], x))
            y = max(bounds['y_min'], min(bounds['y_max'], y))
            next_pos = (x, y)
            logger.info(f"Using Claude's coordinates (clamped): {next_pos}")
        else:
            next_pos = self.choose_next_position(current_pos, bounds, claude_suggestion_text)
        
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



# --- Simple Autonomous Agent -------------------------------------------------

class AutonomousAgent:
    """
    A simple, emoji-logging autonomous agent.

    Requirements implemented:
    - Explore the sample intelligently
    - Recognize interesting features (basic scoring)
    - Learn from what it sees (in-memory history)
    - Make decisions about where to go next

    Keep it simple for now: random exploration and if/else logic.
    """

    def __init__(self, microscope, vision_analyzer):
        """
        Initialize with a microscope controller and a vision analyzer.

        Args:
            microscope: An object exposing `get_current_position()`, `capture_image()`,
                        `move_to(x, y)`, `get_bounds()`, `zoom_in()`, `zoom_out()`.
            vision_analyzer: An object that can analyze saved images via
                             `analyze_image(path)` and return a structured dict.
        """
        self.microscope = microscope
        self.vision = vision_analyzer

        # Simple memory structure
        self.memory: Dict[str, List] = {
            "positions": [],
            "discoveries": [],
            "actions": [],
        }
        self.visited: List[Tuple[float, float]] = []
        self.uninteresting_streak: int = 0

        # Starting strategy: random exploration
        self.strategy = "random"

        # Where to save temporary frames for analysis
        self.output_dir = Path("outputs/agent")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- Helpers
    def _save_image(self, image: np.ndarray, position: Tuple[float, float], step: int) -> str:
        """Save a single-channel image to PNG for the vision analyzer."""
        # Support shapes: (H, W) or (C, H, W)
        if image.ndim == 3:
            # Take the first channel
            image = image[0]
        # Normalize to 0-255 uint8
        arr = image.astype(np.float32)
        arr = arr - arr.min()
        denom = arr.max() + 1e-8
        arr = (arr / denom * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        x, y = position
        filename = f"step_{step}_x{float(x):.3f}_y{float(y):.3f}.png"
        path = str(self.output_dir / filename)
        img.save(path)
        return path

    def _has_visited(self, pos: Tuple[float, float], tolerance: float = 0.01) -> bool:
        """Check if a position was visited within a tolerance."""
        for (vx, vy) in self.visited:
            if abs(vx - pos[0]) <= tolerance and abs(vy - pos[1]) <= tolerance:
                return True
        return False

    def _random_new_area(self, bounds: Dict[str, float]) -> Tuple[float, float]:
        return (
            random.uniform(bounds["x_min"], bounds["x_max"]),
            random.uniform(bounds["y_min"], bounds["y_max"]),
        )

    # -------------------------------------------------------------- Decision
    def decide_next_action(self, vision_analysis: Dict, current_state: Dict) -> Dict:
        """
        Decision logic.

        Rules:
        - If interesting (score > 7): zoom in and capture more
        - If seen before: skip and move elsewhere
        - If quality is poor: adjust settings
        - If nothing interesting for N steps: jump to random new area
        - If near boundary: turn around

        Returns: action dict, e.g. {"type": "move", "x": 10, "y": 20}
        """
        bounds = current_state["bounds"]
        pos = current_state["position"]

        # Derive a simple score from analysis
        score = 0
        if isinstance(vision_analysis, dict):
            score = max(
                int(vision_analysis.get("image_quality", 0) or 0),
                int(vision_analysis.get("confidence", 0) or 0),
            )
            if vision_analysis.get("interestingness", {}).get("is_interesting"):
                score = max(score, 8)

        # Seen before?
        if self._has_visited(pos, tolerance=0.01):
            # Skip and move elsewhere
            self.uninteresting_streak += 1
            print("😴 Seen this area before, moving on…")
            # Random nearby move
            dx = (bounds["x_max"] - bounds["x_min"]) * random.uniform(-0.1, 0.1)
            dy = (bounds["y_max"] - bounds["y_min"]) * random.uniform(-0.1, 0.1)
            new_x = np.clip(pos[0] + dx, bounds["x_min"], bounds["x_max"])
            new_y = np.clip(pos[1] + dy, bounds["y_min"], bounds["y_max"])
            return {"type": "move", "x": float(new_x), "y": float(new_y)}

        # Poor quality? adjust settings
        quality = int((vision_analysis or {}).get("image_quality", 0) or 0)
        if quality < 4:
            print("🔧 Image quality low. Adjusting settings (zoom out)…")
            try:
                self.microscope.zoom_out(1.2)
            except Exception:
                pass

        # Interesting score
        if score > 7:
            print("✨ Interesting feature detected! Moving closer…")
            try:
                self.microscope.zoom_in(1.2)
            except Exception:
                pass
            # Small inward move
            dx = (bounds["x_max"] - bounds["x_min"]) * random.uniform(-0.05, 0.05)
            dy = (bounds["y_max"] - bounds["y_min"]) * random.uniform(-0.05, 0.05)
            new_x = np.clip(pos[0] + dx, bounds["x_min"], bounds["x_max"])
            new_y = np.clip(pos[1] + dy, bounds["y_min"], bounds["y_max"])
            self.uninteresting_streak = 0
            return {"type": "move", "x": float(new_x), "y": float(new_y)}

        # Nothing interesting streak
        self.uninteresting_streak = self.uninteresting_streak + 1
        if self.uninteresting_streak >= 5:
            print("📍 Exploring new region…")
            new_x, new_y = self._random_new_area(bounds)
            self.uninteresting_streak = 0
            return {"type": "move", "x": float(new_x), "y": float(new_y)}

        # Near boundary? turn around
        margin = 0.05 * (bounds["x_max"] - bounds["x_min"])
        near_edge = (
            pos[0] <= bounds["x_min"] + margin
            or pos[0] >= bounds["x_max"] - margin
            or pos[1] <= bounds["y_min"] + margin
            or pos[1] >= bounds["y_max"] - margin
        )
        if near_edge:
            print("↩️ Near boundary, turning inland…")
            center_x = (bounds["x_max"] + bounds["x_min"]) / 2
            center_y = (bounds["y_max"] + bounds["y_min"]) / 2
            # Step toward center
            new_x = pos[0] + (center_x - pos[0]) * 0.2
            new_y = pos[1] + (center_y - pos[1]) * 0.2
            return {"type": "move", "x": float(new_x), "y": float(new_y)}

        # Default: small random exploration
        print("🔭 Scanning casually, small random step…")
        dx = (bounds["x_max"] - bounds["x_min"]) * random.uniform(-0.08, 0.08)
        dy = (bounds["y_max"] - bounds["y_min"]) * random.uniform(-0.08, 0.08)
        new_x = np.clip(pos[0] + dx, bounds["x_min"], bounds["x_max"])
        new_y = np.clip(pos[1] + dy, bounds["y_min"], bounds["y_max"])
        return {"type": "move", "x": float(new_x), "y": float(new_y)}

    # -------------------------------------------------------------- Explore
    def explore(self, num_steps: int = 100):
        """
        Main loop that runs autonomously.

        Each step:
        1) Capture current image
        2) Analyze with vision AI
        3) Decide next action based on analysis
        4) Execute action
        5) Store results in memory
        6) Print progress update
        """
        print("🧠 Teaching moment: The agent is just a while loop!")
        print("   sense → think → act, repeat until stopped.")

        for step in range(1, num_steps + 1):
            # Sense
            position = self.microscope.get_current_position()
            bounds = self.microscope.get_bounds()
            print(f"🔍 Analyzing position ({position[0]:.3f}, {position[1]:.3f})…")

            try:
                image = self.microscope.capture_image()
            except Exception as e:
                print(f"❌ Capture failed: {e}")
                break

            # Save image for vision module if available
            image_path = None
            try:
                image_path = self._save_image(image, position, step)
            except Exception:
                image_path = None

            # Analyze
            analysis = {}
            if hasattr(self.vision, "analyze_image") and image_path:
                try:
                    analysis = self.vision.analyze_image(image_path) or {}
                except Exception as e:
                    print(f"⚠️ Vision analysis error: {e}. Using simple heuristics.")
                    analysis = {}
            # Fallback simple heuristic
            if not analysis:
                # Use image variance as a crude interest proxy
                var = float(np.var(image))
                quality = int(np.clip(var * 10, 0, 10))
                analysis = {
                    "image_quality": quality,
                    "confidence": quality,
                    "interestingness": {"is_interesting": quality > 7, "reason": "variance proxy"},
                }

            # Think → Decide action
            current_state = {"position": position, "bounds": bounds}
            action = self.decide_next_action(analysis, current_state)

            # Act
            if action.get("type") == "move":
                try:
                    self.microscope.move_to(action["x"], action["y"])
                    print(f"➡️ Moving to ({action['x']:.3f}, {action['y']:.3f})")
                except Exception as e:
                    print(f"❌ Move failed: {e}")
                    break

            # Learn → store in memory
            self.memory["positions"].append(tuple(position))
            self.visited.append(tuple(position))
            self.memory["actions"].append(action)
            self.memory["discoveries"].append({
                "position": tuple(position),
                "analysis": analysis,
                "image_path": image_path,
                "timestamp": time.time(),
            })

            # Progress update
            if analysis.get("interestingness", {}).get("is_interesting"):
                print("✨ Interesting feature detected! Moving closer…")
            else:
                print("😴 Nothing interesting here, moving on…")

        print("✅ Exploration loop complete.")
        # Return simple results for downstream usage
        return {
            "positions": list(self.memory["positions"]),
            "actions": list(self.memory["actions"]),
            "discoveries": list(self.memory["discoveries"]),
        }

