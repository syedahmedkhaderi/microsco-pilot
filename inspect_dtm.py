
import sys
import inspect
import os

print(f"Python executable: {sys.executable}")
print(f"Sys path: {sys.path}")

try:
    import DTMicroscope
    print(f"✅ Imported DTMicroscope from {DTMicroscope.__file__}")
    
    try:
        import DTMicroscope.base
        print("✅ Imported DTMicroscope.base")
        
        try:
            import DTMicroscope.base.afm
            print("✅ Imported DTMicroscope.base.afm")
            
            try:
                from DTMicroscope.base.afm import AFM_Microscope as AFM_Simulator
                print("✅ Found AFM_Microscope class!")
                
                print("\n--- Constructor ---")
                print(inspect.signature(AFM_Simulator.__init__))
                
                print("\n--- scan_raster Signature ---")
                if hasattr(AFM_Simulator, 'scan_raster'):
                    print(inspect.signature(AFM_Simulator.scan_raster))
                    print(AFM_Simulator.scan_raster.__doc__)
                        
                print("\n--- Attributes ---")
                for name, _ in inspect.getmembers(AFM_Simulator):
                     if not name.startswith('_') and not inspect.isfunction(getattr(AFM_Simulator, name)):
                         print(name)
            except ImportError as e:
                 print(f"❌ Failed to import AFM_Microscope: {e}")
                 # List attributes of module
                 print(f"Attributes of afm module: {dir(DTMicroscope.base.afm)}")
                 
        except ImportError as e:
            print(f"❌ Failed to import DTMicroscope.base.afm: {e}")
            
    except ImportError as e:
        print(f"❌ Failed to import DTMicroscope.base: {e}")

except ImportError as e:
    print(f"❌ DTMicroscope NOT found: {e}")
except Exception as e:
    print(f"❌ Error inspecting DTMicroscope: {e}")
