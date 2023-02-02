#%%
import sys
from pathlib import Path
BASEDIR = str(Path(__file__).parent.parent)
# BASEDIR = str(Path().absolute().parent)
sys.path.append(BASEDIR)

import torch
import torch.nn as nn

from utils.pyart import pr2t, rpy2r, t2r, t2p, rodrigues, r2quat
from configs.template import PRIMNET_ARGS_TEMPLATE
from torch.nn import init
from utils.tools import get_linear_layer, normalize_tensor, unnormalize_tensor, bmul_1d
from utils.pyart import check_ps




'''
JOINT_INIT_PARAMTERS = {
    "SOROSIM1D": 
        dict(p_offset_std = 0.1, rpy_offset_std = 0.01, axis_std = 0.1),
    "ABAQUS":
        dict(p_offset_std = 0.01, rpy_offset_std = 0.01, axis_std = 0.01),
    "FINGER":
        dict(p_offset_std = 0.1, rpy_offset_std = 0.01, axis_std = 0.1),
        }

'''

#%%
import torch.nn as nn

class ZeroOutLayer(nn.Module):
    def forward(self, act_embed):
        return torch.zeros(act_embed.shape[0], 1).to(act_embed.device)
    

def eye_batch(batch_size, dim = 4):
    return torch.tile(torch.eye(dim),dims=(batch_size,1,1))

class Tjoint(nn.Module):
    def __init__(self,):
        super(Tjoint, self).__init__()        
        self.q_layer = ZeroOutLayer()
    
    def get_q_value(self, act_embed):
        return self.q_layer(act_embed).squeeze(1)
    
    def forward(self,joint_value):
        batch_size   = joint_value.shape[0]; device = joint_value.device

        T_offset = eye_batch(batch_size, dim=4).to(device)
        T_joint  = eye_batch(batch_size, dim=4).to(device)
        return T_offset,T_joint


class Fjoint(nn.Module):
    def __init__(self, args: PRIMNET_ARGS_TEMPLATE):
        super(Fjoint, self).__init__()
        device = args.device
        self.p_offset   = nn.Parameter(torch.Tensor(3,1))
        self.rpy_offset = nn.Parameter(torch.Tensor(3))
        self.axis       = torch.zeros(3,dtype=torch.float32).to(device)
        
        init.normal_(self.p_offset,0,args.p_offset_std)
        init.normal_(self.rpy_offset,0,args.rpy_offset_std)
        
        self.q_layer    = ZeroOutLayer()

    def get_q_value(self, act_embed):
        return self.q_layer(act_embed).squeeze(1)
    
    def forward(self,joint_value):
        batch_size   = joint_value.shape[0]; device = joint_value.device

        T_offset = pr2t(self.p_offset, rpy2r(self.rpy_offset))
        T_joint  = eye_batch(batch_size, dim=4).to(device)
        return T_offset,T_joint

class Pjoint(nn.Module):
    def __init__(self,args: PRIMNET_ARGS_TEMPLATE):
        super(Pjoint, self).__init__()
        
        self.p_offset    = nn.Parameter(torch.Tensor(3,1))
        self.rpy_offset  = nn.Parameter(torch.Tensor(3))
        self.axis        = nn.Parameter(torch.Tensor(3,1))
        
        init.normal_(self.p_offset,0,args.p_offset_std)
        init.normal_(self.rpy_offset,0,args.rpy_offset_std)
        init.normal_(self.axis,0,args.p_offset_std)
        
        hdim = [args.motor_embed_dim, 1]
        self.q_layer = get_linear_layer(hdim, torch.nn.Identity)
            
    def get_q_value(self, act_embed):
        return self.q_layer(act_embed).squeeze(1)
    
    def forward(self, q_value):
        batch_size   = q_value.shape[0]; device = q_value.device
        
        T_offset = pr2t(self.p_offset, rpy2r(self.rpy_offset))
        R        = eye_batch(batch_size, dim=3).to(device)
        # p            = torch.outer(q_value,self.axis)
        p        = bmul_1d(q_value, torch.tile(self.axis, (batch_size,1,1)))
        T_joint  = pr2t(p,R)

        return T_offset,T_joint

# %%
class Rjoint(nn.Module):
    def __init__(self,args: PRIMNET_ARGS_TEMPLATE):
        super(Rjoint, self).__init__()

        self.p_offset   = nn.Parameter(torch.Tensor(3,1))
        self.rpy_offset = nn.Parameter(torch.Tensor(3))
        self.axis       = nn.Parameter(torch.Tensor(3,1))
        
        init.normal_(self.p_offset,0,args.p_offset_std)
        init.normal_(self.rpy_offset,0,args.rpy_offset_std)
        init.normal_(self.axis,0,args.p_offset_std)
        
        hdim = [args.motor_embed_dim,1]
        self.q_layer    = get_linear_layer(hdim, torch.nn.Identity)

    def get_q_value(self, act_embed):
        return self.q_layer(act_embed).squeeze(1)
    
    def forward(self,q_value):
        batch_size = q_value.shape[0]; device = q_value.device
        
        T_offset = pr2t(self.p_offset, rpy2r(self.rpy_offset))
        R        = rodrigues(self.axis,q_value)
        p        = torch.zeros(batch_size,3,1).to(device)
        T_joint  = pr2t(p,R)
        
        return T_offset, T_joint
# %%
class FK_LAYER(nn.Module):
    def __init__(self,args: PRIMNET_ARGS_TEMPLATE):
        super(FK_LAYER, self).__init__()
        pdim = args.pdim
        joint_seqs = args.joint_seqs
        
        joints = []
        for joint_type in joint_seqs:
            if joint_type == "F":
                joints.append(Fjoint(args))
            elif joint_type == "R":
                joints.append(Rjoint(args))
            elif joint_type == "P":
                joints.append(Pjoint(args))
            elif joint_type == "T":
                joints.append(Tjoint(args))
            else: raise LookupError("Joint type not found")
        
        self.joints = torch.nn.ModuleList(joints)
        
            
    def forward_q(self, act_embeds):
        q_values = []
        for joint in self.joints:
            q_value = joint.get_q_value(act_embeds)
            assert len(q_value.shape) == 1
            
            q_values.append(q_value)
        
        return torch.stack(q_values)
        
        
    def forward(self,act_embeds):
        batch_size = act_embeds.shape[0]; device = act_embeds.device
        
        q_values = self.forward_q(act_embeds)
        
        outs = []
        out = eye_batch(batch_size, dim=4).to(device)
        for joint,q_value in zip(self.joints, q_values):
            T_offset, T_joint = joint(q_value)
            out = out @ T_offset @ T_joint
            outs.append(out)
        
        return outs 
        



class ACT_EMBED(nn.Module):
    def __init__(self,args: PRIMNET_ARGS_TEMPLATE):
        super(ACT_EMBED,self).__init__()
        self.args = args
        hdim = list(args.hdim)
        hdim.insert(0, args.motor_dim)
        hdim.append(args.motor_embed_dim)
        # hdim.append(len(args.joint_seqs))

        layers = get_linear_layer(hdim, args.actv)
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self,motor_control):
        q_values = self.layers(motor_control)
        return q_values


class PRIMNET(nn.Module):
    def __init__(self, args:PRIMNET_ARGS_TEMPLATE):
        super(PRIMNET,self).__init__()
        self.args = args
        
        p_offset_std   = args.p_offset_std
        rpy_offset_std = args.rpy_offset_std
        axis_std       = args.axis_std
        
        self.total_joint_number     = len(args.joint_seqs)
        self.primary_joint_number   = args.marker_num
        
        self.ACT_EMBED  = ACT_EMBED(args)
        self.FK_LAYER   = FK_LAYER(args)

        self.register_motor_std_mean(torch.ones(args.motor_dim), torch.zeros(args.motor_dim))
        self.register_position_std_mean(torch.ones(3), torch.zeros(3))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.5,0.9), eps=1e-4)
        
        
    def forward(self, motor_control=torch.zeros(16,4), OUTPUT_NORMALIZE = True):
        device = motor_control.device; batch_size = motor_control.shape[0]
        primary_joint_number  = self.primary_joint_number
        total_joint_number    = self.total_joint_number
        
        # Input Scale
        motor_std, motor_mean = self.get_buffer("motor_std"), self.get_buffer("motor_mean")
        
        motor_control = normalize_tensor(motor_control,motor_mean,motor_std)
        
        # Forward
        act_embeds = self.ACT_EMBED(motor_control)
        joint_transformation_list = self.FK_LAYER(act_embeds)
        
        # Output Scale
        joint_positions = []
        position_mean, position_std = self.get_buffer("position_mean"), self.get_buffer("position_std")
        for joint_transformation in joint_transformation_list:
            joint_position_ = t2p(joint_transformation)
            
            if OUTPUT_NORMALIZE:
                joint_position_ = unnormalize_tensor(joint_position_, mean = position_mean, std= position_std)
                
            joint_positions.append(joint_position_)
            
        return joint_positions
    
    def register_motor_std_mean(self,motor_std, motor_mean):
        self.register_buffer("motor_std",motor_std)
        self.register_buffer("motor_mean",motor_mean)
        
    def register_position_std_mean(self, position_std, position_mean):
        position_std = position_std.reshape(3,1)
        position_mean = position_mean.reshape(3,1)
        self.register_buffer("position_std",position_std)
        self.register_buffer("position_mean",position_mean)
        


# %%
import torch

temp = torch.ones(128, 100) * 0
temp.shape


