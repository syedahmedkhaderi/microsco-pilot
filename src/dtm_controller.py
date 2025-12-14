'''DTMicroscope controller for adaptive scanning'''
import numpy as np
from PIL import Image


class AdaptiveAFMController:
    '''Simplified AFM controller with adjustable parameters'''

    def __init__(self, sample_size=1000):
        '''Initialize controller

        Args:
            sample_size: Sample dimensions in nm
        '''
        self.sample_size = sample_size
        self.position = [0, 0]
        self.scan_params = {
            'speed': 5.0,      # µm/s
            'force': 1.0,      # nN
            'resolution': 256  # pixels
        }
        self.scan_history = []

        # Try to import DTMicroscope, fallback to simulation
        try:
            from DTMicroscope.afm import AFM_Simulator
            self.microscope = AFM_Simulator()
            self.simulation_mode = False
            print("✅ Using DTMicroscope simulator")
        except ImportError:
            self.simulation_mode = True
            print("📦 Using fallback simulation mode")

    def scan_region(self, x, y, size):
        '''Scan a region with current parameters

        Args:
            x, y: Starting position (nm)
            size: Region size (nm)

        Returns:
            image: Scanned topography
            scan_time: Time taken (seconds)
        '''
        # Calculate scan time based on parameters
        pixels = self.scan_params['resolution']
        scan_time = (size / self.scan_params['speed'])

        if self.simulation_mode:
            # Generate synthetic topography
            image = self._generate_region(x, y, size)
        else:
            # Use DTMicroscope
            image = self.microscope.scan(
                x=x, y=y,
                size=size,
                resolution=self.scan_params['resolution'],
                scan_speed=self.scan_params['speed'],
                setpoint_force=self.scan_params['force']
            )

        self.scan_history.append({
            'position': (x, y),
            'params': self.scan_params.copy(),
            'time': scan_time
        })

        return image, scan_time

    def _generate_region(self, x, y, size):
        '''Generate realistic synthetic AFM image'''
        res = self.scan_params['resolution']
        img = np.random.rand(res, res) * 20

        # Add features based on position
        if x < 300:  # Smooth region
            img += 10
        elif x > 700:  # Particle-rich region
            for _ in range(int(10 / (self.scan_params['speed'] / 5))):
                cx, cy = np.random.randint(20, res-20, 2)
                Y, X = np.ogrid[:res, :res]
                mask = (X-cx)**2 + (Y-cy)**2 <= 100
                img[mask] += 80
        else:  # Grain boundaries
            img[res//2-3:res//2+3, :] += 50

        # Quality depends on scan speed
        noise_level = self.scan_params['speed'] * 2
        img += np.random.randn(res, res) * noise_level

        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def update_params(self, new_params):
        '''Update scan parameters dynamically
        
        Args:
            new_params: Dict with new parameter values
        '''
        if 'speed' in new_params:
            self.scan_params['speed'] = float(new_params['speed'])
        if 'resolution' in new_params:
            self.scan_params['resolution'] = int(new_params['resolution'])  # Ensure integer
        if 'force' in new_params:
            self.scan_params['force'] = float(new_params['force'])
        
        print(f"⚙️  Updated: speed={self.scan_params['speed']:.1f} µm/s, "
              f"force={self.scan_params['force']:.1f} nN, "
              f"res={self.scan_params['resolution']}")

    def get_stats(self):
        '''Get scanning statistics'''
        if not self.scan_history:
            return {}

        total_time = sum(h['time'] for h in self.scan_history)
        avg_speed = np.mean([h['params']['speed'] for h in self.scan_history])

        return {
            'total_regions': len(self.scan_history),
            'total_time': total_time,
            'avg_speed': avg_speed
        }