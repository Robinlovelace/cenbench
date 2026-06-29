import os
import yaml

# ONE-LINE SWITCH: Set to True for rapid testing (using cropped 5,900-edge test datasets), False for full runs
TEST_MODE = False

# Load central cities configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cities.yaml")
with open(CONFIG_PATH, "r") as f:
    _cities_config = yaml.safe_load(f)["cities"]

def get_city_config(city_name):
    """Get configuration dict for a given city."""
    if city_name not in _cities_config:
        raise ValueError(f"City '{city_name}' not configured in config/cities.yaml")
    return _cities_config[city_name]

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
