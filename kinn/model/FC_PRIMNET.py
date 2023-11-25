from kinn.configs.template import FC_PRIMNET_ARGS_TEMPLATE
from kinn.utils.tools import get_linear_layer, normalize_tensor, unnormalize_tensor

import torch
from torch import nn

class FC_PRIMNET(nn.Module):
    def __init__(self, args:FC_PRIMNET_ARGS_TEMPLATE):
        super(FC_PRIMNET, self).__init__()
        self.args = args
        
        hdim = list(args.hdim)
        hdim.insert(0, args.motor_dim)
        
        
        self.layers = get_linear_layer(tuple(hdim), args.actv)
        
        self.out = get_linear_layer((hdim[-1], args.marker_num * 3), torch.nn.Identity)
        
        self.register_motor_std_mean(torch.ones(args.motor_dim), torch.zeros(args.motor_dim))
        self.register_position_std_mean(torch.ones(3,1), torch.zeros(3,1))
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=args.lrd)
        
    def forward(self, motor_control, OUTPUT_NORMALIZE = True):
        motor_std, motor_mean = self.get_buffer("motor_std"), self.get_buffer("motor_mean")
        position_mean, position_std = self.get_buffer("position_mean"), self.get_buffer("position_std")
        motor_control = normalize_tensor(motor_control,motor_mean,motor_std)
        
        joint_position = self.out(self.layers(motor_control))
        
        joint_position = joint_position.reshape(-1, self.args.marker_num, 3, 1)
        if OUTPUT_NORMALIZE:
            joint_position = unnormalize_tensor(joint_position, mean=position_mean, std=position_std)
        return joint_position
    
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