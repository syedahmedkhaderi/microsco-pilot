"""
Visualization module

This module creates visualizations of the agent's exploration,
showing where it has been and what it has discovered.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set up logging
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates visualizations of the agent's exploration.
    
    This includes:
    - Map of visited positions
    - Images captured
    - Discovery timeline
    """
    
    def __init__(self, save_dir="outputs"):
        """
        Initialize the visualizer.
        
        Args:
            save_dir (str): Directory to save visualization images
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Visualizer initialized. Save directory: {self.save_dir}")
    
    def plot_exploration_map(self, memory, bounds, filename="exploration_map.png"):
        """
        Create a map showing where the agent has explored.
        
        Args:
            memory (DiscoveryMemory): Memory object with exploration history
            bounds (dict): Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
            filename (str): Name of the output file
        """
        logger.info("Creating exploration map...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract positions
            if memory.positions_visited:
                positions = np.array(memory.positions_visited)
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]
                
                # Plot the path
                ax.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1, label='Path')
                
                # Plot visited points
                ax.scatter(x_coords, y_coords, c=range(len(x_coords)), 
                          cmap='viridis', s=50, alpha=0.7, label='Visited positions')
                
                # Mark start and end
                if len(x_coords) > 0:
                    ax.scatter(x_coords[0], y_coords[0], c='green', 
                             s=200, marker='o', label='Start', zorder=5)
                    ax.scatter(x_coords[-1], y_coords[-1], c='red', 
                             s=200, marker='s', label='Current', zorder=5)
            else:
                # No positions visited yet
                ax.text(0.5, 0.5, 'No exploration yet', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16, color='gray')
            
            # Set bounds
            ax.set_xlim(bounds['x_min'], bounds['x_max'])
            ax.set_ylim(bounds['y_min'], bounds['y_max'])
            
            # Labels and title
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)
            ax.set_title('Exploration Map', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            filepath = self.save_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Exploration map saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to create exploration map: {e}")
            raise
    
    def plot_image(self, image_data, position, filename=None, title=None):
        """
        Plot and save a single microscope image.
        
        Args:
            image_data (numpy.ndarray): Image data
            position (tuple): (x, y) position where image was taken
            filename (str, optional): Output filename. If None, auto-generates.
            title (str, optional): Plot title
        """
        logger.info(f"Plotting image at position {position}...")
        
        try:
            # Handle different array shapes
            if len(image_data.shape) == 3:
                # Multiple channels - take first one
                if image_data.shape[0] < image_data.shape[2]:
                    plot_data = image_data[0]
                else:
                    plot_data = image_data[:, :, 0]
            else:
                plot_data = image_data
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Display the image
            im = ax.imshow(plot_data, cmap='gray', origin='lower')
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Set title
            if title is None:
                title = f"Image at ({position[0]:.4f}, {position[1]:.4f})"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            ax.set_xlabel('X Pixel', fontsize=10)
            ax.set_ylabel('Y Pixel', fontsize=10)
            
            # Generate filename if not provided
            if filename is None:
                filename = f"image_{position[0]:.4f}_{position[1]:.4f}.png"
            
            filepath = self.save_dir / filename
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Image saved to {filepath}")
            
            # Return the filepath so it can be used for analysis
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to plot image: {e}")
            raise
    
    def create_summary_plot(self, memory, filename="summary.png"):
        """
        Create a summary visualization with multiple images.
        
        Args:
            memory (DiscoveryMemory): Memory object
            filename (str): Output filename
        """
        logger.info("Creating summary plot...")
        
        try:
            discoveries = memory.get_recent_discoveries(n=4)
            
            if not discoveries:
                logger.warning("No discoveries to plot in summary")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
            
            for i, discovery in enumerate(discoveries[:4]):
                ax = axes[i]
                
                # Create a placeholder since we don't store full images
                # In a real implementation, you'd load the saved images
                pos = discovery['position']
                ax.text(0.5, 0.5, 
                       f"Discovery #{discovery['id']}\n"
                       f"Position: ({pos['x']:.4f}, {pos['y']:.4f})\n"
                       f"Mean: {discovery['image_mean']:.2f}",
                       transform=ax.transAxes,
                       ha='center', va='center',
                       fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_title(f"Discovery at ({pos['x']:.4f}, {pos['y']:.4f})")
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(len(discoveries), 4):
                axes[i].axis('off')
            
            plt.suptitle('Recent Discoveries Summary', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filepath = self.save_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")
            raise

