"""
Microscope wrapper module

This module provides a BEGINNER-FRIENDLY wrapper class around the
DTMicroscope AFM simulator. A wrapper class is a small helper that
hides complex details and exposes a few easy methods.

Coordinate system (top view):
    y ^
      |
      |        (0, sample_size)
      |
      |-------------> x
    (0,0)          (sample_size, 0)
Positions are in the range [0, sample_size] for both x and y.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Try to import DTMicroscope. If missing, we fall back to a simulation mode.
HAS_DTMICROSCOPE = True
try:
    from DTMicroscope.base.afm import AFM_Microscope
except Exception:
    HAS_DTMICROSCOPE = False

# Set up logging for this module
logger = logging.getLogger(__name__)


# --- Helpers -----------------------------------------------------------------

def _clamp(value: float, min_v: float, max_v: float) -> float:
    """Clamp value to [min_v, max_v]."""
    return max(min_v, min(max_v, value))


def _assert_in_range(name: str, value: float, min_v: float, max_v: float):
    """Assert that a value is within a valid range with a clear error."""
    assert isinstance(value, (int, float)), f"{name} must be a number"
    assert min_v <= value <= max_v, f"{name}={value} is outside valid range [{min_v}, {max_v}]"


# --- Simulation fallback -----------------------------------------------------

class _FakeMicroscope:
    """
    Lightweight simulator used when DTMicroscope is not installed.

    Generates synthetic AFM-like images and tracks position/bounds.
    """

    def __init__(self, sample_size: float, image_size: int = 256):
        self.sample_size = sample_size
        self.image_size = image_size
        self.x = sample_size / 2
        self.y = sample_size / 2
        # For compatibility with real microscope attributes
        lin = np.linspace(0, sample_size, image_size)
        self.x_coords = lin
        self.y_coords = lin

    def go_to(self, x: float, y: float):
        self.x = _clamp(float(x), 0.0, self.sample_size)
        self.y = _clamp(float(y), 0.0, self.sample_size)

    def get_scan(self, channels=None, scan_rate=None):
        """
        Generate a synthetic AFM-like image (1 channel, HxW).
        """
        h = w = self.image_size
        # Seed with position so nearby spots look similar
        seed = int((self.x * 1000) + (self.y * 1000)) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        base = rng.normal(loc=0.0, scale=0.15, size=(h, w))
        # Add a smooth bump to imitate a feature
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        sigma = rng.uniform(8, 25)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        bump = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        image = base + bump * rng.uniform(0.5, 1.2)
        # Normalize to 0-1 and add channel dimension (1, H, W)
        image = image - image.min()
        image = image / (image.max() + 1e-8)
        return image[np.newaxis, ...]  # shape (1, H, W)

    def get_dataset_info(self):
        return [
            ("channels", ["synthetic_height"]),
            ("signals", ["height"]),
            ("units", ["arb"]),
            ("scans", [0]),
        ]


# --- Main controller ---------------------------------------------------------

@dataclass
class MicroscopeState:
    x: float
    y: float
    magnification: float
    images_captured: int
    position_history: List[Tuple[float, float]] = field(default_factory=list)


class MicroscopeController:
    """
    Beginner-friendly wrapper around the DTMicroscope AFM simulator.

    This class hides the complex DTMicroscope API and exposes simple methods:
      - move_to(x, y)
      - capture_image()
      - get_current_position()
      - zoom_in() / zoom_out()
      - get_state()

    If DTMicroscope is not installed, it automatically switches to a
    lightweight simulation mode that generates synthetic images.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        sample_size: float = 1.0,
        magnification: float = 1.0,
    ):
        """
        Initialize the microscope controller.

        Args:
            data_path: Path to AFM dataset (.h5). Optional; only needed for real DTMicroscope.
            sample_size: Physical sample span for x and y axes (0 to sample_size).
            magnification: Starting magnification (conceptual scaling for images).

        Examples:
            >>> mc = MicroscopeController()
            >>> mc.move_to(0.2, 0.3)
            >>> img = mc.capture_image()
            >>> pos = mc.get_current_position()
        """
        assert sample_size > 0, "sample_size must be positive"
        assert magnification > 0, "magnification must be positive"

        self.sample_size = float(sample_size)
        self.magnification = float(magnification)
        self.images_captured = 0
        self.position_history: List[Tuple[float, float]] = []
        self.image_settings_history: List[Dict] = []

        self._use_simulation = False

        if HAS_DTMICROSCOPE:
            try:
                self._microscope = AFM_Microscope(data_path=data_path)
                self._microscope.setup_microscope(data_source="Compound_Dataset_1")
                logger.info("DTMicroscope AFM initialized.")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize DTMicroscope ({e}). Falling back to simulation mode."
                )
                self._use_simulation = True
        else:
            logger.warning("DTMicroscope not found. Using simulation mode.")
            self._use_simulation = True

        if self._use_simulation:
            self._microscope = _FakeMicroscope(sample_size=self.sample_size)

        # Initialize position to center
        self._microscope.go_to(self.sample_size / 2, self.sample_size / 2)
        self.position_history.append(self.get_current_position())

    # ------------------------------------------------------------------ Utils
    def _validate_position(self, x: float, y: float) -> Tuple[float, float]:
        """Validate and clamp requested position to valid bounds."""
        _assert_in_range("x", x, 0.0, self.sample_size)
        _assert_in_range("y", y, 0.0, self.sample_size)
        return float(x), float(y)

    # ---------------------------------------------------------- Public API
    def move_to(self, x: float, y: float) -> Tuple[float, float]:
        """
        Move the stage to (x, y). Values are clamped to [0, sample_size].

        Args:
            x: X coordinate in sample units.
            y: Y coordinate in sample units.

        Returns:
            (actual_x, actual_y) after clamping.

        Example:
            >>> mc.move_to(0.1, 0.9)
        """
        x, y = self._validate_position(x, y)

        try:
            self._microscope.go_to(x, y)
            actual_x = _clamp(self._microscope.x, 0.0, self.sample_size)
            actual_y = _clamp(self._microscope.y, 0.0, self.sample_size)
            self.position_history.append((actual_x, actual_y))

            # Warn if clamped
            if actual_x != x or actual_y != y:
                logger.warning(
                    f"Requested ({x:.4f}, {y:.4f}) clamped to ({actual_x:.4f}, {actual_y:.4f})"
                )

            return actual_x, actual_y
        except AssertionError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Failed to move: {e}")
            raise

    def get_current_position(self) -> Tuple[float, float]:
        """
        Return the current (x, y) position.

        Example:
            >>> mc.get_current_position()
            (0.5, 0.5)
        """
        try:
            return float(self._microscope.x), float(self._microscope.y)
        except Exception as e:
            logger.error(f"Failed to read position: {e}")
            raise

    def capture_image(self, channels: Optional[List[str]] = None, scan_rate: float = 0.5):
        """
        Capture an image at the current position.

        Args:
            channels: Image channels to capture (ignored in simulation mode).
            scan_rate: Scanning rate (Hz). Lower = slower but more accurate (used in real mode).

        Returns:
            numpy.ndarray: Image array (channels, H, W).

        Example:
            >>> img = mc.capture_image()
            >>> img.shape
            (1, 256, 256)
        """
        # Basic input validation
        assert scan_rate > 0, "scan_rate must be positive"

        try:
            image = self._microscope.get_scan(channels=channels, scan_rate=scan_rate)
            self.images_captured += 1
            self.image_settings_history.append(
                {
                    "position": self.get_current_position(),
                    "magnification": self.magnification,
                    "scan_rate": scan_rate,
                    "channels": channels or ["default"],
                }
            )
            logger.info(
                f"Captured image #{self.images_captured} at {self.get_current_position()}, "
                f"shape={image.shape}, mag={self.magnification}"
            )
            return image
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            raise

    def zoom_in(self, factor: float = 1.2) -> float:
        """
        Increase magnification by a factor.

        Args:
            factor: Multiplier > 1.0 (e.g., 1.2 increases by 20%).

        Returns:
            New magnification value.
        """
        assert factor > 1.0, "zoom_in factor must be > 1.0"
        self.magnification *= factor
        logger.info(f"Zoomed in. Magnification now {self.magnification:.2f}x")
        return self.magnification

    def zoom_out(self, factor: float = 1.2) -> float:
        """
        Decrease magnification by a factor.

        Args:
            factor: Multiplier > 1.0 (e.g., 1.2 decreases by ~17%).

        Returns:
            New magnification value.
        """
        assert factor > 1.0, "zoom_out factor must be > 1.0"
        self.magnification /= factor
        logger.info(f"Zoomed out. Magnification now {self.magnification:.2f}x")
        return self.magnification

    def get_state(self) -> MicroscopeState:
        """
        Get the current microscope state as a dataclass.

        Returns:
            MicroscopeState with position, magnification, images captured, history.
        """
        x, y = self.get_current_position()
        return MicroscopeState(
            x=x,
            y=y,
            magnification=self.magnification,
            images_captured=self.images_captured,
            position_history=list(self.position_history),
        )

    def get_bounds(self) -> Dict[str, float]:
        """
        Get the valid movement bounds.

        Returns:
            dict with x_min, x_max, y_min, y_max
        """
        return {
            "x_min": 0.0,
            "x_max": self.sample_size,
            "y_min": 0.0,
            "y_max": self.sample_size,
        }

    # ---------------------------------------------------------- Safety helpers
    def validate_and_warn(self, x: float, y: float):
        """
        Validate a position and print warnings for unusual inputs.

        - Warn if near edges.
        - Warn if values are NaN/inf.
        """
        if not np.isfinite(x) or not np.isfinite(y):
            logger.warning("Position contains non-finite values (NaN or inf).")
        if x < 0 or y < 0 or x > self.sample_size or y > self.sample_size:
            logger.warning("Requested move outside bounds; it will be clamped.")
        edge_margin = 0.05 * self.sample_size
        if (
            x < edge_margin
            or y < edge_margin
            or x > self.sample_size - edge_margin
            or y > self.sample_size - edge_margin
        ):
            logger.info("Warning: near sample edge; images may have edge artifacts.")


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Microscope demo runner")
    parser.add_argument("--demo", action="store_true", help="Run a simple demo scan")
    parser.add_argument("--steps", type=int, default=10, help="Number of demo steps")
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/microscope_demo",
        help="Directory to save demo images",
    )
    args = parser.parse_args()

    if not args.demo:
        print("Nothing to do. Use --demo to run a demo scan.")
        raise SystemExit(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mc = MicroscopeController()
    bounds = mc.get_bounds()

    print("🚀 Starting microscope demo (simulation if DTMicroscope unavailable)…")
    for i in range(1, args.steps + 1):
        # Random position in bounds
        x = float(np.random.uniform(bounds["x_min"], bounds["x_max"]))
        y = float(np.random.uniform(bounds["y_min"], bounds["y_max"]))
        mc.validate_and_warn(x, y)
        actual_x, actual_y = mc.move_to(x, y)
        img = mc.capture_image()
        # Take first channel and scale to 0-255
        arr = img[0]
        arr = arr - arr.min()
        arr = (arr / (arr.max() + 1e-8) * 255.0).astype(np.uint8)
        img_path = outdir / f"step_{i:03d}_x{actual_x:.3f}_y{actual_y:.3f}.png"
        Image.fromarray(arr).save(img_path)
        print(f"📸 Saved {img_path}")

    print("✅ Demo complete.")

