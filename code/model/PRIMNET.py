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
from utils.tools import get_linear_layer, normalize_tensor, unnormalize_tensor, bmul_1d, swap_dim_0_1
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
        self.q_layer = get_linear_layer(hdim, torch.nn.Identity, bias='none')
            
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
        self.q_layer    = get_linear_layer(hdim, torch.nn.Identity, bias='none')

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
        
        # Output with Batch first
        return torch.stack(q_values, dim=1)
        
    def forward_kinematics(self, q_values):
        batch_size = q_values.shape[0]; device = q_values.device
        
        outs = []
        out = eye_batch(batch_size, dim=4).to(device)
        for joint,q_value in zip(self.joints, swap_dim_0_1(q_values)):
            assert len(q_value) == batch_size
            
            T_offset, T_joint = joint(q_value)
            out = out @ T_offset @ T_joint
            outs.append(out)
        
        outs = torch.stack(outs, dim=1)
        assert outs.shape==(batch_size,len(self.joints),4,4)
        
        return outs
        
    def forward(self,act_embeds):
        q_values = self.forward_q(act_embeds)
        return self.forward_kinematics(q_values)
    


class ACT_EMBED(nn.Module):
    def __init__(self,args: PRIMNET_ARGS_TEMPLATE):
        super(ACT_EMBED,self).__init__()
        self.args = args
        args = self.args
        hdim = list(args.hdim)
        hdim.insert(0, args.motor_dim)
        hdim.append(args.motor_embed_dim)
        # hdim.append(len(args.joint_seqs))

        self.layers = get_linear_layer(tuple(hdim), args.actv, bias='none')
        # self.layers = get_linear_layer(tuple(hdim), torch.nn.Identity, bias='none')
        
    
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
        self.register_position_std_mean(torch.ones(3,1), torch.zeros(3,1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.5,0.9), eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=args.lrd)
        
    def forward(self, motor_control=torch.zeros(16,4), OUTPUT_NORMALIZE = True, OUTPUT_POSITION = True):
        device = motor_control.device; batch_size = motor_control.shape[0]
        primary_joint_number  = self.primary_joint_number
        total_joint_number    = self.total_joint_number
        
        # Input Scale
        motor_control = self.normalize(motor_control)
        
        # Forward
        act_embeds = self.ACT_EMBED(motor_control)
        joint_se3 = self.FK_LAYER(act_embeds)
        
        if OUTPUT_POSITION:
            return self.t2p(joint_se3, OUTPUT_NORMALIZE)
        else:
            return joint_se3
        
    def forward_q(self, motor_control=torch.zeros(16,4)):
        device = motor_control.device; batch_size = motor_control.shape[0]
        primary_joint_number  = self.primary_joint_number
        total_joint_number    = self.total_joint_number
        
        # Input Scale
        motor_control = self.normalize(motor_control)
        
        # Forward
        act_embeds = self.ACT_EMBED(motor_control)
        q_values = self.FK_LAYER.forward_q(act_embeds)
        
        return q_values
    
    def t2p(self, joint_se3, OUTPUT_NORMALIZE = True):
        # Get position from joint_se3
        joint_positions = []
        # position_mean, position_std = self.get_buffer("position_mean"), self.get_buffer("position_std")
        for joint_se3_ in swap_dim_0_1(joint_se3):
            joint_position_ = t2p(joint_se3_)
            
            # # Output Scale
            # if OUTPUT_NORMALIZE:
            #     joint_position_ = unnormalize_tensor(joint_position_, mean = position_mean, std= position_std)
                
            joint_positions.append(joint_position_)
            
        return torch.stack(joint_positions, dim=1)
    
    def normalize(self, motor_control):
        motor_std, motor_mean = 20*self.get_buffer("motor_std"), self.get_buffer("motor_mean")
        return normalize_tensor(motor_control,motor_mean,motor_std)
    
    def register_motor_std_mean(self,motor_std, motor_mean):
        self.register_buffer("motor_std",motor_std)
        self.register_buffer("motor_mean",motor_mean)
        
    def register_position_std_mean(self, position_std, position_mean):
        assert position_mean.shape == (3,1)
        assert position_std.shape == (3,1)
        
        self.register_buffer("position_std",position_std)
        self.register_buffer("position_mean",position_mean)
        
    def save_weights(self, epoch):
        torch.save(self.state_dict(), f"{self.args.SAVE_WEIGHT_PATH}/epoch_{epoch+1}.pth")


# %%
import torch

temp = torch.ones(128, 100) * 0
temp.shape


