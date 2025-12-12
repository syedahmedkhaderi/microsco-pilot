import os
from pathlib import Path

# Add src to path for imports when running directly
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.microscope import MicroscopeController
from src.vision import VisionAnalyzer, MockVisionAnalyzer
from src.agent import AutonomousAgent

USE_REAL_API = False  # Toggle: set True for final demo with real API

if __name__ == "__main__":
    # Initialize components
    microscope = MicroscopeController()

    if USE_REAL_API:
        # Use environment key if present, otherwise fail fast
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Set it in .env or export it.")
        vision = VisionAnalyzer(api_key=api_key)
    else:
        # Mock mode: no API needed, ideal for development
        vision = MockVisionAnalyzer()

    agent = AutonomousAgent(microscope, vision)

    # Run exploration
    print("🚀 Starting autonomous exploration...")
    results = agent.explore(num_steps=50)
    print(f"✅ Complete! Found {len(results['discoveries'])} interesting features")

