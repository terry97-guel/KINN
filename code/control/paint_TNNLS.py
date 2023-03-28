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


from soro_kinematics import update_ur_q, forward_model, r2rpy_np, solve_ik_traj, viz_robot

# %%
# Init values
qs= np.array([0,-90,90,-90,-90, 0]).astype(np.float32) / 180 * PI
chain_ur = update_ur_q(chain_ur, qs)
motor_control_np = np.array([0,0,0,0]).astype(np.float32)

qs_list = []
motor_list = []
target_position_list = []
target_rpy_list = []
traj_n = 3
scale_rate = 30

RUN = True
VIZ = True

# %%
from matplotlib import pyplot as plt
z_down = 0.53624987
z_up = z_down + 50/1000 # 50mm up

rpy_EE_tar = np.array([-3.14079618e+00, -4.37113875e-08, -1.57079625e+00], dtype=np.float32)

letter_pos_list = []

# # T
# letter_pos_list.append(np.array([-0-10,-2+10], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-0-10,-40-10], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-0-10-5, -20], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-58-10-5,-13], dtype=np.float32)/1000)

# # N
# letter_pos_list.append(np.array([-58,-48], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-0+6 ,-56], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-0-13 ,-56-13], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-58-13-6,-80-13-6], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-58,-80], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-0+6 ,-87], dtype=np.float32)/1000)

# # N
# letter_pos_list.append(np.array([-58,-100], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-0+6 ,-108], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-0-13 ,-108-13], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-58-13-6,-132-13-6], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-58,-132], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-0+6 ,-139], dtype=np.float32)/1000)


# # L
# letter_pos_list.append(np.array([-0-10-7 ,-160-7], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-58-10-7,-152-7], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-58-10,-152-8], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-58-10,-184-18], dtype=np.float32)/1000)

# S
letter_pos_list.append(np.array([-6, -232], dtype=np.float32)/1000)
letter_pos_list.append(np.array([-0+5, -218+5], dtype=np.float32)/1000)

letter_pos_list.append(np.array([-0, -218], dtype=np.float32)/1000)
letter_pos_list.append(np.array([-13, -201], dtype=np.float32)/1000)

letter_pos_list.append(np.array([-13-5, -201-15], dtype=np.float32)/1000)
letter_pos_list.append(np.array([-26-10+2-5, -214-7-10-13], dtype=np.float32)/1000)

letter_pos_list.append(np.array([-26-10+2-5, -214-7-10-13], dtype=np.float32)/1000)
letter_pos_list.append(np.array([-47-10+2-5, -229-7-10-13], dtype=np.float32)/1000)

letter_pos_list.append(np.array([-47-3-5+10+3-10, -229-3-15], dtype=np.float32)/1000)
letter_pos_list.append(np.array([-58-6-5+10+3-10, -214+5-15], dtype=np.float32)/1000)

letter_pos_list.append(np.array([-58-2-5+10-2-10, -214+6-15], dtype=np.float32)/1000)
letter_pos_list.append(np.array([-53+5-5+10-2-10,-198+5+10-15], dtype=np.float32)/1000)


letter_pos_array = np.vstack(letter_pos_list) - np.array([0,-2], dtype=np.float32)/1000 + np.array([0.82545507,0.09943377])


pos_list = []
grasp_list = [0]
grasp_dir_list  = []


for i in range(0, len(letter_pos_array), 2):
    # if not i in [6]: continue
    letter_pos_1 = letter_pos_array[i]
    letter_pos_2 = letter_pos_array[i+1]
    
    pos_1 = np.array([letter_pos_1[0], letter_pos_1[1], z_up], dtype=np.float32)
    pos_2 = np.array([letter_pos_1[0], letter_pos_1[1], z_down], dtype=np.float32)
    pos_3 = np.array([letter_pos_2[0], letter_pos_2[1], z_down], dtype=np.float32)
    pos_4 = np.array([letter_pos_2[0], letter_pos_2[1], z_up], dtype=np.float32)

    # R = chain_ur.joint[8].R

    grasp_dir = (pos_3-pos_2)
    grasp_dir = grasp_dir/np.linalg.norm(grasp_dir)

    grasp_dir_list.extend([grasp_dir for _ in range(4)])
    grasp_list.extend([-1.0,-1.0,0.2,0.2]) #
    # grasp_list.extend([0,0,0,0])
    pos_list.extend([pos_1, pos_2, pos_3, pos_4])

pos_array = np.vstack(pos_list)
grasp_array = np.array(grasp_list, dtype=np.float32)
grasp_dir_array = np.vstack(grasp_dir_list)


print(pos_array)
# %%
from soro_kinematics import make_markers
p_EE_cur_list = []
obj_info_list = []

assert len(pos_array) == len(grasp_array)-1 == len(grasp_dir_array)
if RUN:
    try: 
        for idx in range(len(pos_array)-1):
            # if idx == 20: 
                # motor_control_np = np.array([0,0,0,0]).astype(np.float32)
            # if idx != 1: # and idx != 5: 
                # continue

            grasp_init = grasp_array[idx]
            grasp_end  = grasp_array[idx+1]

            p_EE_tar_init = pos_array[idx].reshape(3,1)
            p_EE_tar_end  = pos_array[idx+1].reshape(3,1)

            grasp_dir = grasp_dir_array[idx]

            qs_list_, motor_list_, qs, motor_control_np, p_EE_cur_list_ = \
                solve_ik_traj(chain_ur, qs, 
                            soro, motor_control_np, 
                            grasp_init, rpy_EE_tar, p_EE_tar_init, grasp_end, rpy_EE_tar, p_EE_tar_end, grasp_dir, 
                            traj_n=traj_n,scale_rate=scale_rate, step_size = 0.10,VIZ=True)
            
            if VIZ and (idx-1)%4==0:
            # if VIZ:
                for marker_idx, p_EE_cur_ in enumerate(p_EE_cur_list_):
                    
                    radius = 0.01
                    obj_info_list.append(
                        make_markers(name=f"maker_{marker_idx}",
                                    type="sphere",  
                                    pos=p_EE_cur_.flatten().tolist(),
                                    rot=[0,0,0], 
                                    size=[radius, radius, radius], 
                                    color=[1,0,0,1])) 


                motor_control = torch.tensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
                viz_robot(chain_ur, soro, motor_control, obj_info_list, render_time = 1)
                p_EE_cur_list.extend(p_EE_cur_list_)

            qs_list.extend(qs_list_)
            motor_list.extend(motor_list_)
    except KeyboardInterrupt:
        qs_array = np.array(qs_list, dtype=np.float32)
        motor_array = torch.tensor(motor_list) * scale_rate
        p_EE_cur_array = np.array(p_EE_cur_list, dtype=np.float32)

        Path.mkdir((BASEDIR/"control/planned_traj/paint"), parents=True, exist_ok=True)
        np.save((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__(), qs_array)
        np.save((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__(), motor_array.detach().cpu().numpy())
        np.save((BASEDIR/"control/planned_traj/paint/p_EE_cur_array.npy").__str__(), p_EE_cur_array)




    qs_array = np.array(qs_list, dtype=np.float32)
    motor_array = torch.tensor(motor_list) * scale_rate
    p_EE_cur_array = np.array(p_EE_cur_list, dtype=np.float32)

    Path.mkdir((BASEDIR/"control/planned_traj/paint"), parents=True, exist_ok=True)
    np.save((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__(), qs_array)
    np.save((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__(), motor_array.detach().cpu().numpy())
    np.save((BASEDIR/"control/planned_traj/paint/p_EE_cur_array.npy").__str__(), p_EE_cur_array)

    exit()

# %%
from soro_kinematics import viz_robot, update_ur_q, make_markers
import numpy as np
import torch


qs_array       = np.load((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__())
motor_array    = np.load((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__())
p_EE_cur_array = np.load((BASEDIR/"control/planned_traj/paint/p_EE_cur_array.npy").__str__())


motor_array = torch.tensor(motor_array)

# motor_array = motor_array[:10]
# qs_array = qs_array[:10]
# p_EE_cur_array = p_EE_cur_array[:10]

obj_info_list = []
for idx in range(len(qs_array)):
    update_ur_q(chain_ur, qs_array[idx])
    
    if (idx//traj_n)%4==1:    
        radius = 0.01

        marker_idx = (idx//(4*traj_n))*traj_n + idx%traj_n
        p_EE_cur_ = p_EE_cur_array[marker_idx]
        obj_info_list.append(
            make_markers(name=f"maker_{marker_idx}",
                            type="sphere",  
                            pos=p_EE_cur_.flatten().tolist(),
                            rot=[0,0,0], 
                            size=[radius, radius, radius], 
                            color=[1,0,0,1])) 

    motor_control = torch.tensor(motor_array[idx]).unsqueeze(0)
    viz_robot(chain_ur, soro, motor_control, obj_info_list, render_time = 1)

print("here")





# %%





















# # %%
# letter_pos_list.append(np.array([-2,-0, z_up], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-2,-0, z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-40,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-40,-0,z_up], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-20,-0,z_up], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-20,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-13,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-13,-58,z_up], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-48,-58,z_up], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-48,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-56,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-80,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-87,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-87,-0,z_up], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-100,-58,z_up], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-100,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-108,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-132,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-139,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-139,-0,z_up], dtype=np.float32)/1000)

# letter_pos_list.append(np.array([-100,-58,z_up], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-100,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-108,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-132,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-139,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-139,-0,z_up], dtype=np.float32)/1000)


# letter_pos_list.append(np.array([-160,-0,z_up], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-160,-0,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-152,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-184,-58,z_down], dtype=np.float32)/1000)
# letter_pos_list.append(np.array([-184,-58,z_up], dtype=np.float32)/1000)


# letter_pos_list = np.vstack(letter_pos_list) - np.array([-2,0,0], dtype=np.float32)/1000 + np.array([0.82545507,0.09943377, .0])



# grasp_list.insert(0, 0)
# p_init = np.array([0.59898293, 0.13959704, 0.7965294], dtype=np.float32)
# p_tar_array = np.vstack([p_init, letter_pos_list])


# # %%




# # %%
# if RUN1:
#     """APPROCAH"""
#     grasp_dir = 0

#     chain_ur = update_ur_q(chain_ur, qs)
#     motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
    
#     p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
#     p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
#     p_EE_cur = p_EE_cur.detach().cpu().numpy()
#     R_EE_cur = R_plat
    
#     grasp_init = 0.0
#     rpy_EE_tar_init = np.array([-3.14079618e+00, -4.37113875e-08, -1.57079625e+00], dtype=np.float32)
#     p_EE_tar_init =  np.array([0.59898293, 0.13959704, 0.7965294], dtype=np.float32)


#     grasp_end = -1
#     rpy_EE_tar_end = rpy_EE_tar_init
#     p_EE_tar_end = np.array([0.82545507,0.09943377,z_down], dtype=np.float32)
    
#     qs_list_, motor_list_, qs, motor_control_np = \
#         solve_ik_traj(chain_ur, qs, soro, motor_control_np, grasp_init, rpy_EE_tar_init, p_EE_tar_init, grasp_end, rpy_EE_tar_end, p_EE_tar_end, grasp_dir, traj_n=10,scale_rate=scale_rate, VIZ=VIZ)

#     qs_list.extend(qs_list_)
#     motor_list.extend(motor_list_)

#     qs_save = qs; motor_control_np_save = motor_control_np
# # %%
# qs_list = qs_list[:traj_n]
# motor_list = motor_list[:traj_n]
# if RUN2:
#     """SCRAP"""
#     qs=qs_save; motor_control_np = motor_control_np_save
    
#     grasp_dir = 0

#     qs_tar = np.array([-0.6, -57.6, 109.6, -179.4, -100.5, 0.0]).astype(np.float32) / 180 * PI
#     chain_ur = update_ur_q(chain_ur, qs_tar)
#     motor_control = torch.FloatTensor(np.array([0,0,0,1000]).astype(np.float32)).unsqueeze(0).cpu()

#     p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
#     p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
#     p_EE_cur = p_EE_cur.detach().cpu().numpy()
#     R_EE_cur = R_plat


#     grasp_init = -1
#     rpy_EE_tar_init = rpy_EE_tar_init
#     p_EE_tar_init = p_EE_cur + np.array([0,0,-8]).reshape(3,1)/1000


#     qs_tar = np.array([-3.6, -56.0, 108.3, -177.1, -97.4, 0.0]).astype(np.float32) / 180 * PI
#     chain_ur = update_ur_q(chain_ur, qs_tar)
#     motor_control = torch.FloatTensor(np.array([0,0,1000,0]).astype(np.float32)).unsqueeze(0).cpu()

#     p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
#     p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
#     p_EE_cur = p_EE_cur.detach().cpu().numpy()
#     R_EE_cur = R_plat


#     grasp_end = 0.6
#     rpy_EE_tar_end = rpy_EE_tar_init
#     p_EE_tar_end = p_EE_tar_init + np.array([0,-50,0]).reshape(3,1)/1000

#     qs_list_, motor_list_, qs, motor_control_np = \
#         solve_ik_traj(chain_ur, qs, soro, motor_control_np, grasp_init, rpy_EE_tar_init, p_EE_tar_init, grasp_end, rpy_EE_tar_end, p_EE_tar_end, grasp_dir, traj_n=10,scale_rate=scale_rate, VIZ=VIZ)

#     qs_list.extend(qs_list_)
#     motor_list.extend(motor_list_)


# # %%
# qs_array = np.array(qs_list, dtype=np.float32)
# motor_array = torch.tensor(motor_list) * scale_rate

# np.save((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__(), qs_array)
# np.save((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__(), motor_array.detach().cpu().numpy())

# # %%
# from soro_kinematics import viz_robot, update_ur_q
# import numpy as np
# import torch

# qs_array = np.load((BASEDIR/"control/planned_traj/paint/qs_array.npy").__str__())
# motor_array = np.load((BASEDIR/"control/planned_traj/paint/motor_array.npy").__str__())
# motor_array = torch.tensor(motor_array)

# for qs_, motor_ in zip(qs_array,motor_array):
#     update_ur_q(chain_ur, qs_)
#     viz_robot(chain_ur, soro, motor_.unsqueeze(0), render_time=0.5)

# print("here")