# %%
from configs.template import PCC_PRIMNET_ARGS_TEMPLATE
from utils.tools import get_linear_layer, normalize_tensor, unnormalize_tensor,swap_dim_0_1

import torch
from torch import nn

class PCC_PRIMNET(nn.Module):
    def __init__(self, args:PCC_PRIMNET_ARGS_TEMPLATE):
        super(PCC_PRIMNET, self).__init__()
        self.args = args
        
        hdim = list(args.hdim)
        hdim.insert(0, args.motor_dim)
        
        
        self.layers = get_linear_layer(tuple(hdim), args.actv)
        
        self.out = get_linear_layer((hdim[-1], args.marker_num * 3), torch.nn.Identity)
        
        self.register_motor_std_mean(torch.ones(args.motor_dim), torch.zeros(args.motor_dim))
        self.register_position_std_mean(torch.ones(3,1), torch.zeros(3,1))
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=args.lrd)
        
    def forward(self, motor_control, OUTPUT_NORMALIZE = True, OUTPUT_POSITION = True):
        motor_std, motor_mean = self.get_buffer("motor_std"), self.get_buffer("motor_mean")
        motor_control = normalize_tensor(motor_control,motor_mean,motor_std)
        
        out = self.layers(motor_control)
        out = self.out(out)
        
        batch_size = out.shape[0]
        out = out.reshape(batch_size, self.args.marker_num, 3)
        
        phi,theta,radius = out[:,:,0], out[:,:,1], out[:,:,2]

        
        joint_se3 = forward_pcc(phi,theta, radius)

        if OUTPUT_POSITION:
            return self.t2p(joint_se3, OUTPUT_NORMALIZE)
        else:
            return joint_se3

    
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
    
from utils.pyart import r2t,rpy2r,pr2t, t2p
def eye_batch(batch_size, dim = 4):
    return torch.tile(torch.eye(dim),dims=(batch_size,1,1))

def forward_pcc(phi, theta, radius):
    device = phi.device

    batch_size = phi.shape[0]
    section_number = phi.shape[1]
    
    outs = []
    out = eye_batch(batch_size, dim=4).to(device)

    for section in range(section_number):
        phi_ = phi[:,section]
        theta_ = theta[:,section]
        radius_ = radius[:,section]
        
        phi_rpy = torch.stack([torch.zeros_like(phi_), torch.zeros_like(phi_), phi_], dim=1)
        assert phi_rpy.shape == (batch_size,3)
        T1 = r2t(rpy2r(phi_rpy))

        theta_rpy = torch.stack([ -theta_, torch.zeros_like(-theta_), torch.zeros_like(-theta_)],dim = 1)
        assert theta_rpy.shape == (batch_size,3)
        
        P = torch.zeros(batch_size, 3, 1).to(device)
        
        assert (radius_ * (1-torch.cos(theta_))).shape == (batch_size,)
        assert (radius_ * torch.sin(theta_)).shape == (batch_size,)
        
        P[:,1,0] = radius_ * (1-torch.cos(theta_))
        P[:,2,0] = radius_ * torch.sin(theta_)
        
        T2 = pr2t(P,rpy2r(theta_rpy))
        
        T = T1@T2
        out = out @ T
        outs.append(out)
    
    outs = torch.stack(outs, dim=1)
    return outs

# %%
if __name__ == "__main__":
    from configs.PCC_PRIMNET.ABAQUS_32 import ARGS
    args = ARGS()
    model = PCC_PRIMNET(args)

    motor_control = torch.zeros(2,args.motor_dim)

    # %%
    out = model.out(model.layers(motor_control))


    phi,theta,radius = out[:,0], out[:,1], out[:,2]
    phi=phi.reshape(-1,1)
    theta=theta.reshape(-1,1)
    radius=radius.reshape(-1,1)


    forward_pcc(phi,theta,radius)


    # %%
    model(motor_control).shape

    # %%