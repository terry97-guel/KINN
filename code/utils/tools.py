import numpy as np
import torch
import random
from typing import List, Dict
from torch import Tensor

### Casting ###
def cast_tensor(array):
    if isinstance(array, torch.Tensor): return array
    else: return torch.tensor(array)

def cast_numpy(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    elif isinstance(tensor, torch.Tensor):
        return array32(tensor.detach().cpu())
    else: return array32(tensor)

def cast_dict_numpy(dictionary_:dict):
    return {key:cast_numpy(value) for key,value in dictionary_.items()}

def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np

def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

def array32(array) -> np.ndarray:
    return np.array(array,dtype=np.float32)

### Operation ###
def normalize_tensor(tensor:torch.zeros(16,4),mean:torch.zeros(4),std:torch.ones(4)):
    return (tensor-mean)/std

def unnormalize_tensor(tensor:torch.zeros(16,4),mean:torch.zeros(4),std:torch.ones(4)):
    return tensor*std+mean

def get_int_from_ratio(length, ratio):
    return int(ratio*length)

def expand_1d_like(vec_1d, tensor):
    assert vec_1d.shape[0] == tensor.shape[0]
    batch_size = vec_1d.shape[0]
    expand_shape = (batch_size,) + (1,) * (tensor.dim() - 1)
    repeat_shape = (1,) + tuple(tensor.shape[1:])
    vec_ = vec_1d.reshape(expand_shape).repeat(repeat_shape)
    return vec_

def bmul_1d(vec_1d, tensor):
    vec_ = expand_1d_like(vec_1d, tensor)
    return vec_ * tensor
    

def swap_dim_0_1(x:Tensor):
    dim = x.dim()
    return x.permute(1,0, *range(2,dim))

if __name__ == "__main__":
    a = torch.tensor([1,2])
    b = torch.tensor([ [[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[2,2,2,2],[2,2,2,2],[2,2,2,2]] ])
    
    torch.allclose(expand_1d_like(a,b), b)


### Logging ###
from datetime import datetime

def get_runname():
    now = datetime.now()
    format = "%m%d:%H%M"
    runname = now.strftime(format)
    return runname
    

def print_log_dict(log_dictionary):
    log_strs = ""
    for key, item in log_dictionary.items():
        log_str = "{}:{:.2E}, ".format(key,item)
        log_strs = log_strs + log_str
    print(log_strs, "\n")    


def prefix_dict(log_dictionary:dict, prefix:str):
    return {prefix+"_"+key: value for key,value in log_dictionary.items()}


def average_dict(dict_list:List[Dict]):
    assert len(dict_list) > 0
    
    base_dict = dict_list[0]
    
    avg_dict = {}
    for key in base_dict.keys():
        value_list = []
        for dict_ele in dict_list:
            value_list.append(dict_ele[key])
        avg_dict[key] = np.average(np.stack(value_list))
    
    return avg_dict


### Consturcting ###
from torch import nn

def get_linear_layer(hdim, hidden_actv, bias='full') -> nn.Module:
    hdim = tuple(hdim)
    layers = []
    for hdim_idx in range(0,len(hdim)-1):
        if bias == 'zero':
            layer = nn.Linear(hdim[hdim_idx],hdim[hdim_idx+1],bias=True)
            layer.bias.data = torch.zeros_like(layer.bias.data)
        elif bias == 'none':
            layer = nn.Linear(hdim[hdim_idx],hdim[hdim_idx+1],bias=False)
        elif bias == "full":    
            layer = nn.Linear(hdim[hdim_idx],hdim[hdim_idx+1])
        else: 
            raise LookupError("bias type not found")
        # torch.nn.init.normal_(layer.weight,.0,std)
        torch.nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
        layers.append(hidden_actv())
    return nn.Sequential(*layers)

### Seed ###
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determsinistic = True
    torch.backends.cudnn.benchmark = False
    

def set_wandb(pname=None, runname=None):
    import wandb
    if pname is not None:
        wandb.init(project = pname)
    else:
        wandb.init()
        
    if runname is not None:
        wandb.run.name = runname
        
        
### Visualize ###
def make_circle(radius, center=np.zeros(2)):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius * np.cos(theta) + center[0]
    b = radius * np.sin(theta) + center[1]
    return [a,b]

