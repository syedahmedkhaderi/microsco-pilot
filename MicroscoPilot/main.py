"""
MicroscoPilot - Main Entry Point

This is the main script to run the autonomous microscopy agent.
It sets up all the components and starts the exploration loop.
"""

import logging
import os
import sys
from pathlib import Path

# Try to load .env file if it exists (for API keys)
# This allows you to store your API key in a .env file instead of exporting it
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads .env file if it exists
except ImportError:
    # python-dotenv not installed, try manual .env loading
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Add src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from src.microscope import MicroscopeWrapper
from src.vision import VisionAnalyzer
from src.memory import DiscoveryMemory
from src.visualizer import Visualizer
from src.agent import MicroscoPilotAgent


def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    This makes it easy to see what's happening during execution.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory
    log_dir = Path("outputs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up both file and console logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_dir / "microscopilot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """
    Main function that runs the autonomous exploration agent.
    
    This function:
    1. Sets up logging
    2. Initializes all components (microscope, vision, memory, visualizer)
    3. Creates the agent
    4. Runs the exploration loop
    """
    print("=" * 60)
    print("MicroscoPilot - Autonomous Microscopy Agent")
    print("=" * 60)
    print()
    
    # Set up logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting MicroscoPilot...")
    
    try:
        # Step 1: Initialize the microscope
        logger.info("Step 1: Initializing microscope...")
        
        # Try to find a dataset file
        # You can modify this path to point to your dataset
        data_path = None
        
        # Check if there's a dataset in the DTMicroscope directory
        dtmicroscope_data = Path("../DTMicroscope/data/AFM")
        if dtmicroscope_data.exists():
            # Look for any .h5 file
            h5_files = list(dtmicroscope_data.glob("*.h5"))
            if h5_files:
                data_path = str(h5_files[0])
                logger.info(f"Found dataset: {data_path}")
        
        microscope = MicroscopeWrapper(data_path=data_path)
        
        # Set up the microscope with a dataset
        # 'Compound_Dataset_1' is a common dataset name in DTMicroscope
        microscope.setup(data_source='Compound_Dataset_1')
        
        logger.info("Microscope initialized successfully!")
        print("✓ Microscope ready")
        print()
        
        # Step 2: Initialize the vision analyzer (Claude API)
        logger.info("Step 2: Initializing Claude Vision API...")
        
        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
            print("Please set it before running:")
            print("  export ANTHROPIC_API_KEY='your-api-key-here'")
            return
        
        vision_analyzer = VisionAnalyzer(api_key=api_key)
        logger.info("Vision analyzer initialized successfully!")
        print("✓ Claude Vision API ready")
        print()
        
        # Step 3: Initialize memory system
        logger.info("Step 3: Initializing memory system...")
        memory = DiscoveryMemory(save_dir="outputs")
        logger.info("Memory system initialized!")
        print("✓ Memory system ready")
        print()
        
        # Step 4: Initialize visualizer
        logger.info("Step 4: Initializing visualizer...")
        visualizer = Visualizer(save_dir="outputs")
        logger.info("Visualizer initialized!")
        print("✓ Visualizer ready")
        print()
        
        # Step 5: Create the agent
        logger.info("Step 5: Creating agent...")
        agent = MicroscoPilotAgent(
            microscope=microscope,
            vision_analyzer=vision_analyzer,
            memory=memory,
            visualizer=visualizer
        )
        logger.info("Agent created!")
        print("✓ Agent ready")
        print()
        
        # Step 6: Run exploration
        logger.info("Step 6: Starting exploration...")
        print("=" * 60)
        print("Starting autonomous exploration...")
        print("The agent will:")
        print("  - Move around the sample")
        print("  - Capture images")
        print("  - Analyze them with Claude Vision API")
        print("  - Track discoveries")
        print("=" * 60)
        print()
        
        agent.run_exploration()
        
        # Final summary
        summary = memory.get_summary()
        print()
        print("=" * 60)
        print("EXPLORATION COMPLETE!")
        print("=" * 60)
        print(f"Total discoveries: {summary['total_discoveries']}")
        print(f"Positions visited: {summary['positions_visited']}")
        print()
        print("Results saved in the 'outputs' directory:")
        print("  - memory.json: All discoveries")
        print("  - exploration_map.png: Map of visited positions")
        print("  - summary.png: Summary visualization")
        print("  - microscopilot.log: Detailed log file")
        print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Exploration interrupted by user")
        print("\nExploration interrupted by user")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the log file for details: outputs/microscopilot.log")
        raise


if __name__ == "__main__":
    main()

