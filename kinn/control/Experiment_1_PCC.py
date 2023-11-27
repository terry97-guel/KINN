# %%
TRAJ_SIZE = 'small' # 'large' or 'small'
from kinn.utils.tools import set_seed
set_seed(0)
# %%
# dataset = "ELASTICA" 
dataset = "FINGER" 

from pathlib import Path
import os
path = Path().absolute()

if path.name =="PCC":
    os.chdir(path.parent)

BASEDIR = Path().absolute()
BASEDIR

# %%
from kinn.utils.dataloader import get_dataset, Sampler
train_dataset,val_dataset,test_dataset,ext_dataset  = get_dataset(f"../../dataset/{dataset}.json", 1.0)

# %%
from kinn.utils.args import read_ARGS
configs = f"PRIMNET/{dataset}.py"


args = read_ARGS((BASEDIR/'../configs'/configs).absolute())
args

# %%
args.section_number = 6

# %%
import torch

def eye_batch(batch_size, dim = 4):
    return torch.tile(torch.eye(dim),dims=(batch_size,1,1))

# %%
from kinn.utils.pyart import r2t,rpy2r,pr2t, t2p

batch_size = 16

phi    = torch.ones(batch_size, args.section_number)
theta  = torch.ones(batch_size, args.section_number)
radius = torch.ones(batch_size, args.section_number)



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

forward_pcc(phi,theta,radius).shape

# %%
train_sampler = Sampler(args, train_dataset)
val_sampler   = Sampler(args, val_dataset)
test_sampler  = Sampler(args, test_dataset)
ext_sampler   = Sampler(args, ext_dataset)

PI = torch.pi


# %%
def motor_control_to_array(motor_control):
    motor_array = torch.zeros(len(motor_control),4)

    motor_array[:,0] = torch.where(motor_control[:,0]>=0, motor_control[:,0], torch.zeros_like(motor_control[:,0]))
    motor_array[:,1] = torch.where(motor_control[:,0]<0, -motor_control[:,0], torch.zeros_like(motor_control[:,0]))

    motor_array[:,2] = torch.where(motor_control[:,1]>=0, motor_control[:,1], torch.zeros_like(motor_control[:,1]))
    motor_array[:,3] = torch.where(motor_control[:,1]<0, -motor_control[:,1], torch.zeros_like(motor_control[:,1]))

    return motor_array

# %%
x = torch.tensor(.0); y = torch.tensor(.0); z = torch.tensor(.0);
L = torch.tensor(120.0)
R = torch.tensor(5.0)
d = torch.tensor(7.5)


LR = torch.tensor([L,R,d,x,y,z])
LR.requires_grad = True

from torch import optim
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from functools import partial

optimizer = optim.LBFGS([LR],
                        history_size=10,
                        max_iter=4,
                        line_search_fn="strong_wolfe")



def p_loss_fn(x,y):
    loss =  F.l1_loss(x,y, reduction='none')
    loss = torch.mean(loss, dim=list(range(1,loss.ndim)))
    return loss

def closure(LR, motor_control):
    optimizer.zero_grad()
    pred_position = predict(LR, motor_control)
    position_loss =  p_loss_fn(pred_position, target_position[:,-1])
    
    total_loss = torch.mean(position_loss)
    total_loss.backward()
    return total_loss


def predict(LR, motor_control):
    L,R,d,x,y,z = LR
    motor_array = motor_control_to_array(motor_control)
    Lengths = torch.ones_like(motor_array)*L - motor_array/4096 * R * 2*PI
    
    l1,l3,l4,l2 = Lengths[:,0],Lengths[:,1],Lengths[:,2],Lengths[:,3]

    phi = -torch.atan2((l4-l2),(l3-l1))
    theta = torch.sqrt((l3-l1)**2 + (l4-l2)**2) / (2*d)
    l = (l1+l2+l3+l4)/4
    radius = l/theta


    phi    = phi.reshape(-1,1)
    theta  = theta.reshape(-1,1)
    radius = radius.reshape(-1,1)

    phi.shape, theta.shape, radius.shape
    joint_se3 = forward_pcc(phi,theta, radius)

    EE_se3 = joint_se3[:,-1,:,:]
    pred_position = t2p(EE_se3)
    base_position = torch.stack([x,y,z]).reshape(-1,1)
    
    # assert pred_position.shape[1:] == base_position.shape
    pred_position = pred_position + base_position
    return pred_position 


# %%
h_lbfgs = []

for i in range(100):
    for batch in train_sampler:
        target_position = batch["position"]
        motor_control = batch["motor_control"]
        
        optimizer.zero_grad()
    
    # pred_position = f(LR)
    # position_loss =  p_loss_fn(pred_position, target_position[:,-1])
    # objective = torch.mean(position_loss)
    
    
    objective = optimizer.step(partial(closure, LR, motor_control))
    h_lbfgs.append(objective.item())
    
plt.semilogy(h_lbfgs, label='L-BFGS')
plt.legend()
plt.show()

# %%
pred_position = predict(LR, motor_control)
position_loss =  p_loss_fn(pred_position, target_position[:,-1])
objective = torch.mean(position_loss)
objective

# %%
for sampler in [test_sampler, ext_sampler]:

    batch = sampler.sample_all()
    motor_control = batch["motor_control"]
    target_position = batch["position"]


    with torch.no_grad():
        pred_position = predict(LR, motor_control)
        position_loss =  p_loss_fn(pred_position, target_position[:,-1])

        total_loss = torch.mean(position_loss)
        print(float(total_loss))
# %%
# Solve IK

if TRAJ_SIZE.lower()=="small":
    target_trajectory = torch.FloatTensor(
        [[0,0,0],[-0.5,-0.5,0],[0.5,-0.5,0],[0.5,0.5,0],[-0.5,0.5,0],[-0.5,-0.5,0]]
        ) * 30/1000
else:
    target_trajectory = torch.FloatTensor(
        [[0,0,0],[-0.5,-0.5,0],[0.5,-0.5,0],[0.5,0.5,0],[-0.5,0.5,0],[-0.5,-0.5,0]]
        ) * 60/1000

# %%
import numpy as np
def interpolate(a, b, n_points):
    interp_points = torch.linspace(0, 1, n_points+2)[:]
    output = (1 - interp_points)[:, None] * a + interp_points[:, None] * b
    return output


# %%
from functools import partial
motor_control_list = []
position_list = []

scale_rate = 1000

for i in range(len(target_trajectory)):
    motor_actuation = torch.FloatTensor([[1, 0]]).requires_grad_(True)
    target_position_tensor = target_trajectory[i]

    def loss_pred_position(motor_control):
        pred_position = predict(LR, motor_control)
        return p_loss_fn(pred_position.flatten()[:2], target_position_tensor.flatten()[:2])

    optimizer = optim.LBFGS([motor_actuation],
                            history_size=10,
                            max_iter=4,
                            line_search_fn="strong_wolfe")

    # optimizer = optim.LBFGS([motor_actuation],
    #                         history_size=10,
    #                         max_iter=4)

    def closure(motor_actuation):
        optimizer.zero_grad()
        loss = loss_pred_position(motor_actuation*scale_rate )  # Pass motor_control to loss_pred_position
        loss.backward()
        return loss

    loss_list = []
    for i in range(100):
        optimizer.step(partial(closure, motor_actuation))  # Pass the closure function
        # print loss
        loss = closure(motor_actuation)
        loss_list.append(loss.item())
    
    plt.plot(loss_list)

    motor_control = motor_actuation*scale_rate
    # Capture results after optimization
    with torch.no_grad():
        pred_position = predict(LR, motor_control)
        motor_control_list.append(motor_control.detach().numpy().tolist())
        position_list.append(pred_position.detach().numpy().tolist())

        print("here")
# %%
position_array = np.array(position_list).reshape(-1,3)[:,:2]

plt.plot(position_array[:,0], position_array[:,1])
plt.plot(target_trajectory[:,0], target_trajectory[:,1], alpha=0.5)
plt.axis('equal')

# %%
motor_control_list
# %%

# %%
motor_control = torch.FloatTensor([[1000, 0]])
pred_position = predict(LR, motor_control)
pred_position


model = 'PCC'

if TRAJ_SIZE.lower() == 'small':
    name = "small_square"
else:
    name = "large_square"
    
path = Path(f"planned_traj/{name}/{model}_plan.npz")
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path,
          actuation=np.array(motor_control_list),
            target_trajectory=target_trajectory)

data = np.load(path)
# %%
data['actuation'],
data['target_trajectory']
# %%
