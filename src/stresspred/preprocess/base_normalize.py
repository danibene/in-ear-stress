import numpy as np


def base_normalize(abs_data, base_data, method="divide"):
    """Normalize values by a baseline
    Parameters
    ----------
    base_data : int, float
        Baseline data.
    abs_data : int, float
        Data to be normalized.
    method : str
        Method for normalization. Can be "divide" or "subtract".
    Returns
    ----------
    norm_data: int, float
        Normalized data.
    """
    base_cent = np.median(base_data)
    if method == "subtract":
        norm_data = abs_data - base_cent
    else:
        # to avoid NaNs
        if base_cent == 0:
            norm_data = 0
        else:
            norm_data = abs_data / base_cent
    return norm_data
