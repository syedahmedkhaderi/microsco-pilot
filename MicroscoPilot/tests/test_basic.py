"""
Basic tests for MicroscoPilot

Simple tests to verify that the basic components work.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from src.memory import DiscoveryMemory
from src.visualizer import Visualizer


def test_memory():
    """Test that memory system works."""
    print("Testing memory system...")
    
    # Create memory instance
    memory = DiscoveryMemory(save_dir="outputs/test")
    
    # Create dummy image data
    dummy_image = np.random.rand(100, 100)
    
    # Add a discovery
    memory.add_discovery(
        position=(0.5, 0.5),
        image_data=dummy_image,
        analysis="Test analysis",
        feature_description="Test feature"
    )
    
    # Check that it was added
    assert len(memory.discoveries) == 1
    assert memory.discoveries[0]['id'] == 0
    
    # Test has_visited
    assert memory.has_visited((0.5, 0.5))
    assert not memory.has_visited((1.0, 1.0))
    
    # Test summary
    summary = memory.get_summary()
    assert summary['total_discoveries'] == 1
    
    # Test save/load
    memory.save("test_memory.json")
    memory2 = DiscoveryMemory(save_dir="outputs/test")
    memory2.load("test_memory.json")
    assert len(memory2.discoveries) == 1
    
    print("✓ Memory system test passed!")


def test_visualizer():
    """Test that visualizer works."""
    print("Testing visualizer...")
    
    # Create visualizer
    visualizer = Visualizer(save_dir="outputs/test")
    
    # Create dummy image
    dummy_image = np.random.rand(100, 100)
    
    # Test plotting
    visualizer.plot_image(dummy_image, (0.5, 0.5), "test_image.png")
    
    # Check that file was created
    assert (Path("outputs/test") / "test_image.png").exists()
    
    print("✓ Visualizer test passed!")


def test_basic():
    """Run all basic tests."""
    print("=" * 60)
    print("Running basic tests...")
    print("=" * 60)
    print()
    
    try:
        test_memory()
        print()
        test_visualizer()
        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_basic()

