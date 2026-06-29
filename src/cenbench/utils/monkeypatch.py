import pandas as pd
import numpy as np

def apply_patches():
    """Apply monkeypatches to resolve pandas 3.0+ and numpy 2.0+ incompatibilities in madina."""
    
    # 1. Patch pandas Series.__init__ to remove the obsolete 'fastpath' parameter
    _orig_series_init = pd.Series.__init__
    def patched_series_init(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'fastpath'}
        _orig_series_init(self, *args, **filtered_kwargs)
    
    pd.Series.__init__ = patched_series_init

    # 2. Patch numpy.array_split to properly return pandas DataFrame/Series chunks
    # instead of converting the DataFrame into a numpy object array under numpy 2.0+
    _orig_array_split = np.array_split
    def patched_array_split(ary, indices_or_sections, axis=0):
        if isinstance(ary, (pd.DataFrame, pd.Series)):
            split_indices = _orig_array_split(range(len(ary)), indices_or_sections, axis=axis)
            return [ary.iloc[idx] for idx in split_indices]
        return _orig_array_split(ary, indices_or_sections, axis=axis)
    
    np.array_split = patched_array_split
