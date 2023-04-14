from torch import Tensor
from numpy import ndarray
import numpy as np

def cast_to_numpy(data):
    if isinstance(data,Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, ndarray):
        return data
    if isinstance(data,list):
        return np.array(data)
    else:
        raise TypeError(f"Expected instance of ['Tensor', 'ndarry', 'list'] but recived {type(data)}")

