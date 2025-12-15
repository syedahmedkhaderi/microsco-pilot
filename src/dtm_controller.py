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
        self.last_scan_result = None

        # Try to import DTMicroscope and sidpy
        try:
            import sidpy
            from DTMicroscope.base.afm import AFM_Microscope
            self.microscope = AFM_Microscope()
            self.sidpy = sidpy
            self.simulation_mode = False
            print("✅ DTMicroscope AFM_Simulator initialized (Physics Mode)")
        except ImportError as e:
            self.simulation_mode = True
            self.microscope = None
            print(f"📦 DTMicroscope/sidpy not found ({e}), using fallback simulation mode")

    def scan_region(self, x, y, size):
        '''Scan a region with current parameters

        Args:
            x, y: Starting position (nm)
            size: Region size (nm)

        Returns:
            image: Scanned topography (PIL Image)
            scan_time: Time taken (seconds)
        '''
        # Calculate realistic scan time
        pixels = self.scan_params['resolution']
        speed = self.scan_params['speed']
        if speed <= 0: speed = 0.1
        line_time = size / speed
        settling_time = 0.02
        scan_time = (pixels * line_time * 2) + (pixels * settling_time)

        # Generate base synthetic topography (the "sample")
        base_image_pil = self._generate_region(x, y, size)
        base_image = np.array(base_image_pil)

        if self.simulation_mode:
            image = base_image_pil
            self.last_scan_result = {'Height': base_image, 'source': 'synthetic'}
        else:
            # Use DTMicroscope Physics
            # Retry loop to handle potential broadcast errors from DTMicroscope internals
            # (Sometimes fails on specific resolutions due to internal rounding)
            max_retries = 3
            current_res = pixels
            
            for attempt in range(max_retries):
                try:
                    # Ensure resolution matches the current attempt (if we retried)
                    if current_res != base_image.shape[0]:
                        base_image_pil = base_image_pil.resize((current_res, current_res), Image.LANCZOS)
                        base_image = np.array(base_image_pil)
                        
                    # 1. Create sidpy Dataset
                    dset = self.sidpy.Dataset.from_array(base_image, name='Height')
                    dset.data_type = self.sidpy.DataType.IMAGE
                    dset.units = 'nm'
                    dset.quantity = 'Height'
                    
                    # Add dimensions
                    dim_x = np.linspace(0, size, base_image.shape[0])
                    dim_y = np.linspace(0, size, base_image.shape[1])
                    dset.set_dimension(0, self.sidpy.Dimension(dim_x, name='x', units='nm', quantity='x', dimension_type='SPATIAL'))
                    dset.set_dimension(1, self.sidpy.Dimension(dim_y, name='y', units='nm', quantity='y', dimension_type='SPATIAL'))
                    
                    # 2. Setup Microscope
                    data_dict = {'Height': dset}
                    self.microscope.data_dict = data_dict
                    self.microscope.setup_microscope()
                    
                    # 3. Configure Physics Effects (PID)
                    # CALIBRATION: "Drift Simulation" Mode
                    # Simulates thermal drift: Slower scans = More drift = Worse quality.
                    # Faster scans = Less drift = Better quality.
                    # Formula: Base 0.15 Hz + Penalty for low speed.
                    drift_penalty = 0.1 / max(speed, 0.1)
                    scan_rate_hz = 0.15 + drift_penalty
                    
                    modification = [{
                        'effect': 'real_PID', 
                        'kwargs': {
                            'scan_rate': scan_rate_hz, 
                            'sample_rate': 2000,
                            'Kp': 0.9  # Strong gain
                        }
                    }]
                    
                    # 4. Scan
                    result_array = self.microscope.get_scan(channels=['Height'], modification=modification)
                    
                    # Result is (1, H, W), take first channel
                    scanned_data = result_array[0]
                    
                    # 5. Convert back to Image
                    norm_data = (scanned_data - np.min(scanned_data)) / (np.max(scanned_data) - np.min(scanned_data) + 1e-10)
                    image = Image.fromarray((norm_data * 255).astype(np.uint8))
                    
                    # Store result
                    self.last_scan_result = {
                        'Height': scanned_data, 
                        'Input': base_image, 
                        'source': 'DTMicroscope'
                    }
                    
                    # If successful, break retry loop
                    break
                    
                except ValueError as e:
                    if "broadcast" in str(e) and attempt < max_retries - 1:
                        print(f"⚠️  DTMicroscope broadcast error at res={current_res}. Retrying with res={current_res-1}...")
                        current_res -= 1
                        continue
                    else:
                        print(f"⚠️  DTMicroscope error: {e}")
                        break
                        
                except Exception as e:
                    print(f"⚠️  DTMicroscope unexpected error: {e}")
                    break
            
            # Check if we have a result, otherwise fallback
            if self.last_scan_result is None or self.last_scan_result.get('source') != 'DTMicroscope':
                print("⚠️  DTMicroscope scan failed after retries. Falling back.")
                image = base_image_pil
                self.last_scan_result = {'Height': base_image, 'source': 'synthetic_fallback'}

        self.scan_history.append({
            'position': (x, y),
            'params': self.scan_params.copy(),
            'time': scan_time,
            'mode': 'synthetic' if self.simulation_mode else 'DTMicroscope'
        })

        return image, scan_time

    def get_quality_from_scan(self, scan_result):
        '''Extract real quality metrics from DTMicroscope data'''
        if scan_result is None or scan_result.get('source') != 'DTMicroscope':
            return None
            
        metrics = {'source': 'DTMicroscope'}
        
        try:
            # Calculate Tracking Error: Difference between Input (Sample) and Output (Scan)
            if 'Input' in scan_result and 'Height' in scan_result:
                input_img = scan_result['Input']
                output_img = scan_result['Height']
                
                # Normalize both to match ranges (physics might add offset)
                in_norm = (input_img - input_img.mean()) / (input_img.std() + 1e-10)
                out_norm = (output_img - output_img.mean()) / (output_img.std() + 1e-10)
                
                # Error map
                error = np.abs(in_norm - out_norm)
                tracking_error = np.mean(error)
                metrics['tracking_error'] = tracking_error
                
                # SNR (Signal to Noise)
                # Signal = Input variance, Noise = Error variance
                signal_power = np.var(in_norm)
                noise_power = np.var(in_norm - out_norm)
                snr = signal_power / (noise_power + 1e-10)
                metrics['snr'] = snr
                
                # Quality Score (0-10)
                # Low error -> High quality
                # Error 0.0 -> 10
                # Error 1.0 -> 0
                quality = 10.0 * np.exp(-tracking_error)
                metrics['quality'] = float(np.clip(quality, 0, 10))
            else:
                metrics['quality'] = 5.0
                
        except Exception as e:
            print(f"⚠️  Error extracting quality: {e}")
            metrics['quality'] = 5.0
            
        return metrics

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