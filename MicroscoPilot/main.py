import os
from pathlib import Path

# Add src to path for imports when running directly
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.microscope import MicroscopeController
from src.vision import VisionAnalyzer
from src.agent import AutonomousAgent

if __name__ == "__main__":
    # Initialize components
    microscope = MicroscopeController()
    # Use environment key if present, otherwise fall back to provided key
    api_key = os.getenv(
        "ANTHROPIC_API_KEY",
        "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",
    )
    vision = VisionAnalyzer(api_key=api_key)
    agent = AutonomousAgent(microscope, vision)

    # Run exploration
    print("🚀 Starting autonomous exploration...")
    results = agent.explore(num_steps=50)
    print(f"✅ Complete! Found {len(results['discoveries'])} interesting features")

