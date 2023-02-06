# %%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from utils.tools import get_int_from_ratio, cast_tensor
import os 
import json
import random
from typing import Union
from configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE
from torch import Tensor


class JsonDataset(Dataset):
    def __init__(self,data):
        motor_control = torch.tensor(data["motor_control"],dtype=torch.float32)
        position = torch.tensor(data["position"],dtype=torch.float32)
        
        assert len(motor_control) == len(position)
        
        self.motor_control = motor_control
        self.position = position.unsqueeze(-1)
        
        assert self.position.shape[-2:] == (3,1)
        
                
    def __len__(self):
        return len(self.motor_control)
    
    def __getitem__(self,idx):
        return dict(motor_control = self.motor_control[idx], position = self.position[idx])


    def get_std_mean(self):
        motor_control = self.motor_control
        position = self.position
        
        # average along joints
        axis_mean_position = torch.mean(position, dim=1)

        motor_std, motor_mean = torch.std_mean(motor_control, dim=0)
        pos_std, pos_mean = torch.std_mean(axis_mean_position, dim=0)


        motor_std, motor_mean, pos_std, pos_mean
        
        return (motor_std, motor_mean), (pos_std, pos_mean)


def get_dataset(data_path, data_ratio=1.0):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train_data = data["train"]
    data_len = int(len(train_data['position']) * data_ratio)
    
    motor_control = cast_tensor(train_data['motor_control'])
    position      = cast_tensor(train_data['position'])
    
    motor_control_norm = torch.norm(motor_control, dim = 1)
    idx = torch.argsort(motor_control_norm)[:data_len]
    
    
    train_data['motor_control'] = motor_control[idx]
    train_data['position'] = position[idx]
    
    val_data = data["val"]
    test_data = data["test"]
    ext_data = data["ext"]
    
    return JsonDataset(train_data), JsonDataset(val_data), JsonDataset(test_data), JsonDataset(ext_data)
    
# %%

class Sampler():
    def __init__(self, args:Union[PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE], dataset:JsonDataset):
        self.focus_ratio = args.focus_ratio
        self.dataset = dataset
        
        self.length = len(dataset)
        self.max_iter = self.length//args.batch_size + 1
        self.current_iter = 0
        
        self.indices = torch.chunk(torch.randperm(self.length), self.max_iter)
        self.keep_idx = torch.randint(0, self.length, [get_int_from_ratio(self.length, self.focus_ratio),])
        
        self.loss_restore = torch.zeros(self.length)
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iter:
            self.current_iter = 0
            self.indices = torch.chunk(torch.randperm(len(self)), self.max_iter)
            
            # sort_idx = torch.argsort(self.loss_restore, descending=True)
            # self.keep_idx = sort_idx[:int(len(sort_idx)*self.focus_ratio)]
            raise StopIteration()

        else:
            self.sel_idx = self.indices[self.current_iter]
            # idx = torch.cat([self.sel_idx, self.keep_idx])
            self.current_iter += 1
            return self.dataset[self.sel_idx]
    
    # def update(self, loss:Tensor):
    #     assert (len(self.sel_idx) + len(self.keep_idx)) == len(loss)
        
    #     sel_idx_loss = loss[:len(self.sel_idx)]
    #     self.loss_restore[self.sel_idx] = sel_idx_loss
        
    
    def sample_all(self):
        idx = torch.arange(self.length)    
        return self.dataset[idx]




# %%
temp = torch.chunk(torch.arange(20), 20 // 3 + 1)

temp0 = temp[0]
temp1 = temp[1]

torch.cat([temp0, temp1])

# %%
