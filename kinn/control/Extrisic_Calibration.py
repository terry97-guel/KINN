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
path = "PRIMNET/FINGER_FULL"

import argparse
from kinn.utils.args import read_ARGS
from kinn.model.PRIMNET import PRIMNET
import torch


parser = argparse.ArgumentParser(description= 'parse for DLPG')
parser.add_argument("--configs", default="PRIMNET/FINGER.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET] # [FINGER, ABAQUS]
args= parser.parse_args([])
args = read_ARGS((BASEDIR/'configs'/args.configs).absolute())



soro = PRIMNET(args=args).to("cpu")
soro.load_state_dict(torch.load(BASEDIR.parent/'results'/path/"weights/epoch_500.pth"))
# %%
import numpy as np
PI = np.pi

file_name = (BASEDIR / "control/urdf/ur5e_soro.urdf").__str__()
assert os.path.exists(file_name)

from kinematics.class_structure import CHAIN
chain_ur = CHAIN(file_name = file_name)


from soro_kinematics import update_ur_q, forward_model, r2rpy_np, solve_ik_traj, viz_robot, solve_ik, rpy2r_np, pr2t_np
from soro_kinematics_save import *

# %%
qs_list = []
motor_list = []
target_position_list = []
target_rpy_list = []
traj_n = 3
scale_rate = 30

RUN = True
VIZ = True
PLOT = False

# %%
from matplotlib import pyplot as plt
# z_down = 0.545
# z_up = z_down + 50/1000 # 50mm up

rpy_plat_tar = np.array([-3.14079618e+00, -4.37113875e-08, -1.57079625e+00], dtype=np.float32)
R_plat_tar = rpy2r_np(rpy_plat_tar)


class Traj():
    def __init__(self):
        self.pos_list = []
        self.u_list = []
        self.grasp_list = []

    def append(self, pos_array, u_array, grasp_array):
        self.pos_list.append(pos_array)
        self.u_list.append(u_array)
        self.grasp_list.append(grasp_array)

    def get_array(self):
        grasp_array = np.hstack(self.grasp_list)
        pos_array = np.vstack(self.pos_list)
        u_array = np.vstack(self.u_list)

        return grasp_array, pos_array, u_array

# %%
from kinn.utils.pyart import r2rpy_numpy
# qs = np.array([-90, -110, 90, 110, 65, 0], dtype=np.float32) / 180 * PI

qs = np.array([-90, -90, 90, 90, 55, 0], dtype=np.float32) / 180 * PI


chain_ur = update_ur_q(chain_ur, qs)
motor_control_np = np.array([0,0]).astype(np.float32)
motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).cpu()

obj_info_list = []

radius = 0.01

import time
for _ in range(10):
    marker1 = chain_ur.joint[-3]

    marker = make_markers(name=f"maker_{1}",
            type="sphere",  
            pos= marker1.p.flatten().tolist(),
            rot=[0,0,0], 
            size=[radius, radius, radius], 
            color=[1,0,0,1])
    obj_info_list.append(marker) 



    length = 0.05
    link  = make_markers(name=f'line1', 
                            type="sphere",  
                            pos=(marker1.p.flatten() + marker1.R[:,0] * length).tolist(),
                            rot=[0,0,0], 
                            size=[radius, radius, radius], 
                            color=[1,0,0,1])
    obj_info_list.append(link) 

    link  = make_markers(name=f'line2', 
                            type="sphere",  
                            pos=(marker1.p.flatten() + marker1.R[:,1] * length).tolist(),
                            rot=[0,0,0], 
                            size=[radius, radius, radius], 
                            color=[0,1,0,1])
    obj_info_list.append(link) 

    link  = make_markers(name=f'line3', 
                            type="sphere",  
                            pos=(marker1.p.flatten() + marker1.R[:,2] * length).tolist(),
                            rot=[0,0,0], 
                            size=[radius, radius, radius], 
                            color=[0,0,1,1])
    obj_info_list.append(link) 

    front_length = 0.055
    up_length = 0.08
    side_length = 0.04

    pos = marker1.p.flatten() + marker1.R[:,0].flatten() * up_length + marker1.R[:,1].flatten() * front_length + side_length * marker1.R[:,2].flatten()


    link  = make_markers(name=f'line4', 
                            type="sphere",  
                            pos=pos.tolist(),
                            rot=[0,0,0], 
                            size=[radius, radius, radius], 
                            color=[0,0,0,1])
    obj_info_list.append(link) 

    viz_robot(chain_ur, soro, motor_control, obj_info_list)
    time.sleep(0.1)

camera_R = marker1.R
print("R:",camera_R)
camera_p = pos
print("position:",camera_p)

np.save("control/planned_traj/camera_p.npy", camera_p)
# %%
