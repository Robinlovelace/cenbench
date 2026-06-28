import os

# ONE-LINE SWITCH: Set to True for rapid testing (using cropped 5,900-edge test datasets), False for full runs
TEST_MODE = True

def get_path(filename):
    """Resolve correct path depending on whether TEST_MODE is active."""
    if TEST_MODE:
        dirname = os.path.dirname(filename)
        base = os.path.basename(filename)
        
        # Route sensor files to their respective projections
        if "pedestrians" in base:
            if "4326" in base:
                return os.path.join(dirname, "test_leuven_telraam_pedestrians_4326.geojson")
            else:
                return os.path.join(dirname, "test_leuven_telraam_pedestrians.geojson")
            
        new_base = f"test_{base}"
        return os.path.join(dirname, new_base) if dirname else new_base
    return filename
