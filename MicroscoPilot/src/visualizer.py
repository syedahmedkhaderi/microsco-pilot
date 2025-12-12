"""
Visualization module

This module creates visualizations of the agent's exploration,
showing where it has been and what it has discovered.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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
        
        # Dashboard state
        self.fig = None
        self.ax_image = None
        self.ax_map = None
        self.ax_timeline = None
        self.ax_stats = None
        self._heatmap_res = 50
        self._heatmap = None
        self._bounds = None
        self._start_time = None
        self._strategy = "random"
        self._video_writer = None
        self._video_path = None
        
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

    # --- Real-time Dashboard -------------------------------------------------
    def start_dashboard(self, bounds, strategy="random", title="MicroscoPilot Dashboard"):
        """
        Initialize a 2x2 dashboard figure for real-time updates.

        Panels:
        - Top Left: Current microscope view
        - Top Right: Exploration heatmap with current position and features
        - Bottom Left: Discoveries timeline (time vs cumulative features)
        - Bottom Right: Statistics text panel
        """
        logger.info("Starting real-time dashboard...")
        self._bounds = bounds
        self._strategy = strategy
        self._start_time = plt.datetime.datetime.now() if hasattr(plt, "datetime") else None

        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle(title, fontsize=14, fontweight="bold")

        # Grid layout 2x2
        self.ax_image = self.fig.add_subplot(2, 2, 1)
        self.ax_map = self.fig.add_subplot(2, 2, 2)
        self.ax_timeline = self.fig.add_subplot(2, 2, 3)
        self.ax_stats = self.fig.add_subplot(2, 2, 4)
        self.ax_stats.axis("off")

        # Initialize heatmap grid
        self._heatmap = np.zeros((self._heatmap_res, self._heatmap_res), dtype=np.int32)

        # Configure panels
        self.ax_image.set_title("Current Microscope View")
        self.ax_map.set_title("Exploration Map (heatmap)")
        self.ax_timeline.set_title("Discoveries Timeline")
        self.ax_image.set_xlabel("X Pixel")
        self.ax_image.set_ylabel("Y Pixel")
        self.ax_map.set_xlabel("X")
        self.ax_map.set_ylabel("Y")

        plt.tight_layout()
        plt.pause(0.001)

    def _update_heatmap(self, positions):
        """Update heatmap counts from positions list."""
        if self._heatmap is None or self._bounds is None:
            return
        x_min, x_max = self._bounds['x_min'], self._bounds['x_max']
        y_min, y_max = self._bounds['y_min'], self._bounds['y_max']

        # Reset to avoid unbounded growth if passing all positions each time
        self._heatmap[:] = 0

        # Bin positions
        for x, y in positions:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                ix = int((x - x_min) / (x_max - x_min + 1e-12) * (self._heatmap_res - 1))
                iy = int((y - y_min) / (y_max - y_min + 1e-12) * (self._heatmap_res - 1))
                self._heatmap[iy, ix] += 1

    def _find_feature_positions(self, discoveries):
        """Heuristically decide which discoveries are 'features' and return their positions."""
        stars = []
        for d in discoveries:
            pos = d.get('position')
            x = pos['x'] if isinstance(pos, dict) else pos[0]
            y = pos['y'] if isinstance(pos, dict) else pos[1]
            text_a = str(d.get('analysis', ''))
            text_f = str(d.get('feature_description', ''))
            # Simple heuristic: mark if 'interesting' or common keywords appear
            if any(k in (text_a + text_f).lower() for k in [
                'interesting', 'defect', 'grain', 'feature', 'particle']):
                stars.append((x, y))
        return stars

    def update_dashboard(self, memory, current_image, current_position, strategy=None):
        """
        Update all four panels in real-time.

        Use plt.pause to keep UI responsive.
        """
        if self.fig is None:
            logger.warning("Dashboard not started. Call start_dashboard() first.")
            return

        try:
            # Top-left: current image
            self.ax_image.clear()
            img = current_image[0] if hasattr(current_image, 'ndim') and current_image.ndim == 3 else current_image
            im = self.ax_image.imshow(img, cmap='gray', origin='lower')
            self.fig.colorbar(im, ax=self.ax_image, fraction=0.046, pad=0.04)
            self.ax_image.set_title("Current Microscope View")

            # Mark current position text
            self.ax_image.text(0.02, 0.98, f"pos=({current_position[0]:.3f},{current_position[1]:.3f})",
                               transform=self.ax_image.transAxes, va='top', ha='left',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.6), fontsize=9)

            # Top-right: exploration map heatmap
            self.ax_map.clear()
            positions = list(memory.positions_visited) + [current_position]
            self._update_heatmap(positions)
            cmap = plt.cm.get_cmap('hot')
            hm = self.ax_map.imshow(self._heatmap, origin='lower', cmap=cmap,
                                    extent=[self._bounds['x_min'], self._bounds['x_max'],
                                            self._bounds['y_min'], self._bounds['y_max']])
            self.fig.colorbar(hm, ax=self.ax_map, fraction=0.046, pad=0.04, label='Visit count')
            self.ax_map.set_title("Exploration Map (heatmap)")

            # Plot current position as red dot
            self.ax_map.scatter([current_position[0]], [current_position[1]], c='red', s=60, zorder=5)

            # Mark discovered features with stars
            stars = self._find_feature_positions(memory.discoveries)
            if stars:
                xs, ys = zip(*stars)
                self.ax_map.scatter(xs, ys, marker='*', c='yellow', s=120, edgecolors='black', zorder=6)

            self.ax_map.set_xlim(self._bounds['x_min'], self._bounds['x_max'])
            self.ax_map.set_ylim(self._bounds['y_min'], self._bounds['y_max'])
            self.ax_map.grid(True, alpha=0.2)
            self.ax_map.set_xlabel('X')
            self.ax_map.set_ylabel('Y')

            # Bottom-left: discoveries timeline (cumulative features)
            self.ax_timeline.clear()
            timestamps = []
            cumulative = []
            count = 0
            for d in memory.discoveries:
                # Use index as time if timestamp not parseable
                count += 1 if any(k in str(d.get('analysis', '')).lower() for k in ['interesting', 'defect', 'feature']) or \
                           any(k in str(d.get('feature_description', '')).lower() for k in ['interesting', 'defect', 'feature']) else 0
                cumulative.append(count)
                timestamps.append(len(timestamps))
            if timestamps:
                self.ax_timeline.plot(timestamps, cumulative, color='blue')
            self.ax_timeline.set_xlabel('Step')
            self.ax_timeline.set_ylabel('Features (cumulative)')
            self.ax_timeline.grid(True, alpha=0.3)

            # Bottom-right: statistics
            self.ax_stats.clear()
            self.ax_stats.axis('off')
            total_steps = len(memory.positions_visited)
            areas_explored = len(set(memory.positions_visited))
            features_found = cumulative[-1] if cumulative else 0
            strategy_text = strategy or self._strategy
            # Time elapsed
            time_elapsed = "N/A"
            if self._start_time is not None:
                # Matplotlib's naive fallback, not all environments have plt.datetime
                time_elapsed = "running"
            stats_lines = [
                f"Total steps taken: {total_steps}",
                f"Areas explored: {areas_explored}",
                f"Features found: {features_found}",
                f"Current strategy: {strategy_text}",
                f"Time elapsed: {time_elapsed}",
            ]
            self.ax_stats.text(0.0, 1.0, "Statistics\n" + "\n".join(stats_lines),
                               va='top', ha='left', fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.pause(0.001)

            # Record video frame if enabled
            if self._video_writer is not None:
                try:
                    self._video_writer.grab_frame()
                except Exception as e:
                    logger.warning(f"Video frame capture failed: {e}")
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")

    def save_dashboard(self, filename="dashboard_final.png"):
        """Save current dashboard state as a PNG."""
        if self.fig is None:
            logger.warning("Dashboard not started. Nothing to save.")
            return None
        filepath = self.save_dir / filename
        try:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Dashboard saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")
            return None

    def save_heatmap(self, memory, bounds, filename="heatmap.png", res=200):
        """Save a high-res heatmap image of visited areas."""
        try:
            x_min, x_max = bounds['x_min'], bounds['x_max']
            y_min, y_max = bounds['y_min'], bounds['y_max']
            grid = np.zeros((res, res), dtype=np.int32)
            for x, y in memory.positions_visited:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    ix = int((x - x_min) / (x_max - x_min + 1e-12) * (res - 1))
                    iy = int((y - y_min) / (y_max - y_min + 1e-12) * (res - 1))
                    grid[iy, ix] += 1
            fig, ax = plt.subplots(figsize=(8, 6))
            hm = ax.imshow(grid, origin='lower', cmap='hot',
                           extent=[x_min, x_max, y_min, y_max])
            plt.colorbar(hm, ax=ax, label='Visit count')
            ax.set_title('Exploration Heatmap (high-res)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            filepath = self.save_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"Heatmap saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save heatmap: {e}")
            return None

    def export_data(self, memory, filename="exploration_data.json"):
        """Export memory data and summary as JSON."""
        try:
            path = self.save_dir / filename
            import json
            with open(path, 'w') as f:
                json.dump({
                    'discoveries': memory.discoveries,
                    'summary': memory.get_summary(),
                }, f, indent=2)
            logger.info(f"Exploration data exported to {path}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return None

    # --- Optional video recording ------------------------------------------
    def start_video(self, filename="exploration.mp4", fps=5):
        """Start recording dashboard updates to an MP4 (requires ffmpeg)."""
        if self.fig is None:
            logger.warning("Dashboard not started. Cannot start video.")
            return None
        try:
            Writer = animation.FFMpegWriter
            self._video_writer = Writer(fps=fps)
            self._video_path = str(self.save_dir / filename)
            self._video_writer.setup(self.fig, self._video_path, dpi=150)
            logger.info(f"Video recording started: {self._video_path}")
            return self._video_path
        except Exception as e:
            logger.warning(f"FFmpeg not available or setup failed: {e}")
            self._video_writer = None
            self._video_path = None
            return None

    def stop_video(self):
        """Stop video recording if active."""
        if self._video_writer is not None:
            try:
                self._video_writer.finish()
                logger.info(f"Video saved to {self._video_path}")
            except Exception as e:
                logger.warning(f"Failed to finalize video: {e}")
        self._video_writer = None
        self._video_path = None
    
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

