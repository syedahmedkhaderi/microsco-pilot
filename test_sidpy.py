
import numpy as np
import sidpy
from DTMicroscope.base.afm import AFM_Microscope

print("Creating synthetic data...")
data = np.random.rand(100, 100)
dset = sidpy.Dataset.from_array(data, name='Height')
dset.data_type = sidpy.DataType.IMAGE
print(f"dset.data_type: {dset.data_type}")
print(f"sidpy.DataType.IMAGE: {sidpy.DataType.IMAGE}")
print(f"sidpy.DataType['IMAGE']: {sidpy.DataType['IMAGE']}")
print(f"Equality check: {dset.data_type == sidpy.DataType['IMAGE']}")

# Add dimensions
dset.set_dimension(0, sidpy.Dimension(np.linspace(0, 1000, 100), name='x', units='nm', quantity='x', dimension_type='SPATIAL'))
dset.set_dimension(1, sidpy.Dimension(np.linspace(0, 1000, 100), name='y', units='nm', quantity='y', dimension_type='SPATIAL'))

print("Initializing AFM_Microscope...")
microscope = AFM_Microscope()
data_dict = {'Height': dset}

print("Setting up microscope...")
microscope.data_dict = data_dict
microscope.setup_microscope()

print("Scanning with physics...")
# Apply PID effect
modification = [{'effect': 'real_PID', 'kwargs': {'scan_rate': 2.0, 'sample_rate': 2000}}]
result = microscope.get_scan(channels=['Height'], modification=modification)

print(f"Result shape: {result.shape}")
print("✅ Success!")
