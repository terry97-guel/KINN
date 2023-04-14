# Experiment 1 of the submitted paper

# %%

from pathlib import Path
import sys
import os
from path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR


BASEDIR, RUNMODE = get_BASERDIR(__file__)
print("Going up one level")
os.chdir((BASEDIR/"..").__str__())

BASEDIR, RUNMODE = get_BASERDIR(".")
print("Current working directory:", os.getcwd())

sys.path.append(str(BASEDIR))


# %%
## Load mode
model = "FC_PRIMNET"
path = model+"/FINGER"

import argparse
from utils.args import read_ARGS
from model.PRIMNET import PRIMNET
from model.FC_PRIMNET import FC_PRIMNET
import torch
import numpy as np



parser = argparse.ArgumentParser(description= 'parse for DLPG')
parser.add_argument("--configs", default=model+"/FINGER.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET] # [FINGER, ABAQUS]
args= parser.parse_args([])
args = read_ARGS((BASEDIR/'configs'/args.configs).absolute())

if args.MODEL == "PRIMNET":
    soro = PRIMNET(args=args).to("cpu")
elif args.MODEL == 'FC_PRIMNET':
    soro = FC_PRIMNET(args=args).to("cpu")
else: raise TypeError("MODEL not found")

soro.load_state_dict(torch.load(BASEDIR.parent/'results'/path/"weights/epoch_1000.pth"))
# %%
# Reference


# %%
from torch.nn import functional as F

def interpolate(a, b, n_points):
    interp_points = torch.linspace(0, 1, n_points+2)[:]
    output = (1 - interp_points)[:, None] * a + interp_points[:, None] * b
    return output



def p_loss_fn(x,y):
    loss =  F.l1_loss(x,y, reduction='none')
    loss = torch.mean(loss, dim=list(range(1,loss.ndim)))
    return loss


# %%
from torch.optim import LBFGS

# Square

target_trajectory = torch.FloatTensor(
    [[0,0,0],[-0.5,-0.5,0],[0.5,-0.5,0],[0.5,0.5,0],[-0.5,0.5,0],[-0.5,-0.5,0]]
    ) * 70/1000


# Circle
# theta = torch.linspace(0,2*torch.pi, 50)
# target_trajectory = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=1) * 20/1000






from matplotlib import pyplot as plt

from utils.tools import cast_numpy
target_trajectory_np = cast_numpy(target_trajectory)
plt.plot(target_trajectory_np[:,0], target_trajectory_np[:,1])

plt.show()


# %%

actuation = torch.FloatTensor([[0,0]]).requires_grad_(True)

from jacobian import jacobian
from functools import partial
def forward_soro(soro,actuation):
    ps = soro(actuation)
    pos_EE = ps[:,-1,0:2,0]

    return pos_EE

temp = soro(torch.FloatTensor([[1000,0]]))
temp = temp[:,-1,:2,0]

ps = soro(actuation* 50)
# temp = interpolate()

jac, pos_EE = jacobian(partial(forward_soro,soro), actuation)

jac, pos_EE

# %%

from tqdm import tqdm

actuation_list = []
position_list = []
actuation = torch.FloatTensor([[0,0]]).requires_grad_(True)

for i in range(len(target_trajectory)-1):
    target_position_tensor = interpolate(target_trajectory[i], target_trajectory[i+1], 0)

    actuation = torch.FloatTensor([[0,0]]).requires_grad_(True)
    for target_position_ in target_position_tensor:
        # print(target_position_)
        # target_position_ = soro(torch.FloatTensor([[100,0]]))[:,-1,:2,0]
        
        idx = 0
        while True:
            if actuation.isnan().any():
                print("Found NAN!")
                break

            jac, pos_EE = jacobian(partial(forward_soro,soro), actuation)
            J_use = jac[0,:,0,:]

            ik_err = target_position_.detach()[:2].reshape(1,-1)-pos_EE[0]

            # print(torch.linalg.norm(ik_err) * 1e5)
            idx = idx+1
            err = torch.linalg.norm(ik_err)
            print(f"actuation:{actuation}, idx:{idx}: err:{err*1000}", end='\r')
            # print(idx,end="\r")

            if err<1e-4:
                break

            if idx > 2_000 and err>1e-4:
                # print("Ground")
                # actuation = torch.FloatTensor([[0,0]]).requires_grad_(True)
                print("Warning: IK not converged")
                break

            lambda_ = 0.001
            J_n_ctrl = torch.matmul(J_use.T, J_use) + lambda_* torch.eye(2, 2).to(torch.float)
            delta_actuation = torch.matmul(torch.linalg.solve(J_n_ctrl, J_use.T), ik_err.flatten())                        
            if torch.linalg.norm(delta_actuation)< 1e-4:
                delta_actuation = delta_actuation* (1e-4/torch.linalg.norm(delta_actuation))
            actuation = actuation + 10_000 * delta_actuation

        print(err * 1e4)

        position_list.append(pos_EE[0].detach().numpy())
        actuation_list.append(actuation.detach().numpy().tolist())

        position_array = np.array(position_list)
        position_array = position_array.reshape(-1,2)

        plt.plot(position_array[:,0], position_array[:,1])
        plt.plot(target_trajectory_np[:,0], target_trajectory_np[:,1], alpha=0.5)
        # plt.show()

plt.show()
# %%
position_array = np.array(position_list)
position_array = position_array.reshape(-1,2)

plt.plot(position_array[:,0], position_array[:,1])
plt.plot(target_trajectory_np[:,0], target_trajectory_np[:,1], alpha=0.5)


Path("control/planned_traj/temp").mkdir(parents=True, exist_ok=True)

actuation_array = np.array(actuation_list)
np.savez("control/planned_traj/temp/"+model+".npz",
          actuation=actuation_array,
            target_trajectory=target_trajectory)


data = np.load('control/planned_traj/temp/'+model+'.npz')

temp = data['actuation']
temp

print(np.max(temp))

# %%