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
path = "PRIMNET/FINGER"

import argparse
from utils.args import read_ARGS
from model.PRIMNET import PRIMNET
import torch


parser = argparse.ArgumentParser(description= 'parse for DLPG')
parser.add_argument("--configs", default="PRIMNET/FINGER.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET] # [FINGER, ABAQUS]
args= parser.parse_args([])
args = read_ARGS((BASEDIR/'configs'/args.configs).absolute())



soro = PRIMNET(args=args).to("cpu")
soro.load_state_dict(torch.load(BASEDIR.parent/'results'/path/"weights/epoch_200.pth"))
# %%
import numpy as np
PI = np.pi

file_name = (BASEDIR / "control/urdf/ur5e_soro.urdf").__str__()
assert os.path.exists(file_name)

from kinematics.class_structure import CHAIN
chain_ur = CHAIN(file_name = file_name)


from soro_kinematics import update_ur_q, forward_model, r2rpy_np, solve_ik_traj

# %%
# Init values
qs= np.array([0,-90,90,-90,-90, 0]).astype(np.float32) / 180 * PI
motor_control_np = np.array([0,0,0,0]).astype(np.float32)

qs_list = []
motor_list = []
target_position_list = []
target_rpy_list = []
traj_n = 10
scale_rate = 30

RUN1 = True
RUN2 = True
RUN3 = True
VIZ = True


# %%
if RUN1:
    """APPROCAH"""
    grasp_dir = 0

    chain_ur = update_ur_q(chain_ur, qs)
    motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
    
    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().cpu().numpy()
    R_EE_cur = R_plat
    
    grasp_init = 0.0
    rpy_EE_tar_init = r2rpy_np(R_EE_cur)
    p_EE_tar_init =  p_EE_cur
    
    qs_tar = np.array([-0.6, -57.6, 109.6, -179.4, -100.5, 0.0]).astype(np.float32) / 180 * PI
    chain_ur = update_ur_q(chain_ur, qs_tar)
    motor_control = torch.FloatTensor(np.array([0,0,0,1000]).astype(np.float32)).unsqueeze(0).cpu()

    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().cpu().numpy()
    R_EE_cur = R_plat


    grasp_end = -1
    rpy_EE_tar_end = rpy_EE_tar_init
    p_EE_tar_end = p_EE_cur + np.array([0,0,-8]).reshape(3,1)/1000
    
    qs_list_, motor_list_, qs, motor_control_np = \
        solve_ik_traj(chain_ur, qs, soro, motor_control_np, grasp_init, rpy_EE_tar_init, p_EE_tar_init, grasp_end, rpy_EE_tar_end, p_EE_tar_end, grasp_dir, traj_n=10,scale_rate=scale_rate, VIZ=VIZ)

    qs_list.extend(qs_list_)
    motor_list.extend(motor_list_)

    qs_save = qs; motor_control_np_save = motor_control_np
# %%
qs_list = qs_list[:traj_n]
motor_list = motor_list[:traj_n]
if RUN2:
    """SCRAP"""
    qs=qs_save; motor_control_np = motor_control_np_save
    
    grasp_dir = 0

    qs_tar = np.array([-0.6, -57.6, 109.6, -179.4, -100.5, 0.0]).astype(np.float32) / 180 * PI
    chain_ur = update_ur_q(chain_ur, qs_tar)
    motor_control = torch.FloatTensor(np.array([0,0,0,1000]).astype(np.float32)).unsqueeze(0).cpu()

    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().cpu().numpy()
    R_EE_cur = R_plat


    grasp_init = -1
    rpy_EE_tar_init = rpy_EE_tar_init
    p_EE_tar_init = p_EE_cur + np.array([0,0,-8]).reshape(3,1)/1000


    qs_tar = np.array([-3.6, -56.0, 108.3, -177.1, -97.4, 0.0]).astype(np.float32) / 180 * PI
    chain_ur = update_ur_q(chain_ur, qs_tar)
    motor_control = torch.FloatTensor(np.array([0,0,1000,0]).astype(np.float32)).unsqueeze(0).cpu()

    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().cpu().numpy()
    R_EE_cur = R_plat


    grasp_end = 0.6
    rpy_EE_tar_end = rpy_EE_tar_init
    p_EE_tar_end = p_EE_tar_init + np.array([0,-50,0]).reshape(3,1)/1000

    qs_list_, motor_list_, qs, motor_control_np = \
        solve_ik_traj(chain_ur, qs, soro, motor_control_np, grasp_init, rpy_EE_tar_init, p_EE_tar_init, grasp_end, rpy_EE_tar_end, p_EE_tar_end, grasp_dir, traj_n=10,scale_rate=scale_rate, VIZ=VIZ)

    qs_list.extend(qs_list_)
    motor_list.extend(motor_list_)


# %%
qs_array = np.array(qs_list, dtype=np.float32)
motor_array = torch.tensor(motor_list) * scale_rate

np.save((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__(), qs_array)
np.save((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__(), motor_array.detach().cpu().numpy())

# %%
from soro_kinematics import viz_robot, update_ur_q
import numpy as np
import torch

qs_array = np.load((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__())
motor_array = np.load((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__())
motor_array = torch.tensor(motor_array)

for qs_, motor_ in zip(qs_array,motor_array):
    update_ur_q(chain_ur, qs_)
    viz_robot(chain_ur, soro, motor_.unsqueeze(0), render_time=0.5)

print("here")