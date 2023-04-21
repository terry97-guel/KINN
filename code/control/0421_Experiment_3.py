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
from utils.args import read_ARGS
from model.PRIMNET import PRIMNET
import torch


parser = argparse.ArgumentParser(description= 'parse for DLPG')
parser.add_argument("--configs", default="PRIMNET/FINGER.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET] # [FINGER, ABAQUS]
args= parser.parse_args([])
args = read_ARGS((BASEDIR/'configs'/args.configs).absolute())



soro = PRIMNET(args=args).to("cpu")
soro.load_state_dict(torch.load(BASEDIR.parent/'results'/path/"weights/epoch_1000.pth"))
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
def get_ur_jacobian(chain_ur, p_EE):
    ## get jacobian
    # position jacobian
    p_J_UR = []

    prev_joint = chain_ur.joint[0]
    for joint in chain_ur.joint[1:7]:
        assert joint.type == 'revolute'
        
        J_ = skew_np(prev_joint.R@joint.a)@(p_EE-joint.p)
        p_J_UR.append(J_)
        prev_joint = joint
    p_J_UR = np.hstack(p_J_UR)

    # angular jacobian
    w_J_UR = []
    for joint in chain_ur.joint[1:7]:
        assert joint.type == 'revolute'
        
        J_ = joint.R@joint.a
        w_J_UR.append(J_)
    w_J_UR  = np.hstack(w_J_UR)

    J_UR = np.vstack([p_J_UR, w_J_UR])

    return J_UR

# %%
runname = "bottle_open"

target_position = np.load("control/planned_traj/bottle/center_position.npy")[[1,0,2]]


pos_list = []
pos_list.append(np.array([target_position[0],target_position[1],target_position[2]+0.1]))

grasp_list = []
grasp_list.append(0)

u_list = []
u_list.append(np.array([1,0,0]))

pos_array = np.stack(pos_list).reshape(-1,3,1)
grasp_array = np.stack(grasp_list)
u_array = np.stack(u_list)



# %%
# Init values
qs= np.array([0,-90,90,-90,-90, 0]).astype(np.float32) / 180 * PI
chain_ur = update_ur_q(chain_ur, qs)
motor_control_np = np.array([0,0]).astype(np.float32)


result_motor_control_list = []
result_qs_list = []
result_p_EE_list = []
obj_info_list = [None]

# offset = np.array([30,-30,0]).reshape(3,1)/1000
offset = np.array([0,0,0]).reshape(3,1)/1000



scale_rate = 30

for i in range(len(grasp_array)):

    grasp    = grasp_array[i]
    u        = u_array[i]
    p_EE_tar = pos_array[i]
    
    step_size = 0.1

    
    pbar = tqdm(generator(), leave=True)
    update_number = 0

    for _ in pbar:
        # Update qs
        chain_ur = update_ur_q(chain_ur, qs)
        motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
        p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

        # get Jacobian
        p_J_soro, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control, scale_rate)
        p_EE = p_EE + (R_plat@offset)
        J_UR = get_ur_jacobian(chain_ur, p_EE)
        J_soro = np.vstack([p_J_soro, np.zeros((3,2), dtype=np.float32)])



        # IK error
        p_EE_ik_err = get_p_ik_err(p_EE, p_EE_tar)
        w_plat_ik_err = get_w_ik_err(R_plat, R_plat_tar)


        # Grasp
        p_plat_EE = (p_EE - chain_ur.joint[8].p).astype(np.float32)
        l_grasp = 0.02 * grasp

        assert np.abs(np.linalg.norm(u) - 1) < 1e-3
        assert u.shape == (3,)
        R_ = chain_ur.joint[8].R.astype(np.float32)
        u_ = u.reshape(3,1)
        p_plat_EE_tar = l_grasp * u_

        J_grasp = (R_.T @ p_J_soro)[:-1]

        grasp_err = (R_.T @ (p_plat_EE_tar - p_plat_EE + offset))[:-1]
        pbar.set_description(
            "update_number:{}, \
                p_ik_err:{:.2E},\
                    w_ik_err:{:.2E},\
                        grasp_err:{:.2E}".format(
                                update_number,
                                norm(p_EE_ik_err),
                                norm(w_plat_ik_err),
                                norm(grasp_err),
                            ))


        # Break
        if norm(p_EE_ik_err) < 2e-3 and\
            norm(w_plat_ik_err) < 0.01 and\
                norm(grasp_err) < 5e-3:
            break
            # print("Done")


        # Or Solve & Update
        A = []
        b = []
        A.append(np.hstack([J_UR, J_soro]))
        A.append(np.hstack([np.zeros((len(J_grasp),6),dtype=np.float32), J_grasp]))

        b.append(np.vstack([p_EE_ik_err,w_plat_ik_err]))
        b.append(10*grasp_err)

        A = np.vstack(A).astype(np.float32)
        b = np.vstack(b).astype(np.float32)

        J_use=A; ik_err=b; lambda_rate = 0.01

        lambda_min = 1e-6
        lambda_max = 1e-3

        ik_err_avg = np.mean(abs(ik_err))

        # Damping Term 
        lambda_ = lambda_rate * ik_err_avg + lambda_min 
        lambda_ = np.maximum(lambda_, lambda_max)

        n_ctrl = (J_use).shape[1]
        # Lamda Scheduling 
        J_culumn_sum = abs(np.sum(J_use, axis =0))

        for j in range(len(J_culumn_sum)):
            for i in J_culumn_sum:
                idx_nz = j
                J_use_nz = J_use[:,idx_nz].reshape(1, -1)
                det_J = np.linalg.det(np.matmul(J_use_nz, J_use_nz.T))
                if i >0.1:
                    if det_J > 1e-3:
                        lambda_=1e-4
                    elif det_J < 1e-20:
                        lambda_ = lambda_max
                        
        J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl).astype(np.float32)
        dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)


        step_size = step_size * 0.999
        dq = step_size * dq_raw

        dq = dq.flatten()
        qs = qs + dq[:6]
        motor_control_np = motor_control_np+ dq[6:] * scale_rate

        VIZ = False
        if VIZ:
            radius = 0.01
            obj_info_list[0] = make_markers(name=f"maker_{i}",
                                            type="sphere",  
                                            pos= p_EE_tar.flatten().tolist(),
                                            rot=[0,0,0], 
                                            size=[radius, radius, radius], 
                                            color=[0,0,1,1])


            viz_robot(chain_ur, soro, motor_control, obj_info_list, offset=offset, render_time=0.1)
        pbar.update()
        update_number = update_number + 1


    radius = 0.01
    obj_info_list.append(
        make_markers(name=f"maker_{i}",
                    type="sphere",  
                    pos= p_EE.flatten().tolist(),
                    rot=[0,0,0], 
                    size=[radius, radius, radius], 
                    color=[1,0,0,1])) 
    print(motor_control_np * scale_rate)
    result_motor_control_list.append(motor_control_np * scale_rate)
    result_qs_list.append(qs)
    result_p_EE_list.append(p_EE)

q_last = qs
motor_last = motor_control_np

# %%
# 1ST
runname = "bottle_open"
pos_list = []
dir_list = []
grasp_list = []

target_position = np.load("control/planned_traj/bottle/center_position.npy")[[1,0,2]]
target_position[-1] = target_position[-1] + 0.025
plt.scatter(-target_position[1],target_position[0],c="r")


# 밖으로 빼고
delta = np.array([-0.02, 0.02, 0])
point = target_position - delta
plt.scatter(-point[1],point[0],c="b")

diff_pos = point - target_position
u = diff_pos / np.linalg.norm(diff_pos)
grasp = np.linalg.norm(diff_pos)/0.02

pos_list.append(point)
dir_list.append(u)
grasp_list.append(grasp)
prev_point = point

# 안으로 넣고
center = target_position +  2.0 * delta
plt.scatter(-center[1],center[0],c="g")



resolution = 10
for i in range(resolution):
    if i == 0 :
        point = np.array([center[0]-0.01, target_position[1]+0.005,center[2]])
    elif i == resolution-1:
        point = np.array([target_position[0]-0.005, center[1]+0.01,center[2]])
    else:
        radius = 0.01

        theta = -PI/8 * 8 + i/(resolution-1) * PI * 12/8
        point = center + radius * np.array([np.cos(theta), np.sin(theta), 0])
    
    diff_pos = point - target_position
    u = diff_pos / np.linalg.norm(diff_pos)
    grasp = np.linalg.norm(diff_pos)/0.02
    
    

    # point[-1] = point[-1] + 0.002/resolution * i
    pos_list.append(point)
    dir_list.append(u)
    grasp_list.append(grasp)
    
    plt.scatter(-point[1],point[0],c="y")


plt.axis('equal')


pos_array = np.stack(pos_list).reshape(-1,3,1)
grasp_array = np.stack(grasp_list)
u_array = np.stack(dir_list)



# %%
# Init values
qs = q_last
motor_control_np = motor_last
chain_ur = update_ur_q(chain_ur, qs)



# result_motor_control_list = []
# result_qs_list = []
# result_p_EE_list = []
# obj_info_list = []


scale_rate = 30

for i in range(len(grasp_array)):

    grasp    = grasp_array[i]
    u        = u_array[i]
    p_EE_tar = pos_array[i]
    
    step_size = 0.1

    
    pbar = tqdm(generator(), leave=True)
    update_number = 0

    for _ in pbar:
        # Update qs
        chain_ur = update_ur_q(chain_ur, qs)
        motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
        p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

        # get Jacobian
        p_J_soro, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control, scale_rate)
        J_UR = get_ur_jacobian(chain_ur, p_EE)
        J_soro = np.vstack([p_J_soro, np.zeros((3,2), dtype=np.float32)])

        J_plat = get_ur_jacobian(chain_ur, p_plat)[:2]


        # IK error
        p_EE_ik_err = get_p_ik_err(p_EE, p_EE_tar)
        w_plat_ik_err = get_w_ik_err(R_plat, R_plat_tar)


        # Grasp
        p_plat_EE = (p_EE - chain_ur.joint[8].p).astype(np.float32)
        l_grasp = 0.02 * grasp

        assert np.abs(np.linalg.norm(u) - 1) < 1e-3
        assert u.shape == (3,)
        R_ = chain_ur.joint[8].R.astype(np.float32)
        u_ = u.reshape(3,1)
        p_plat_EE_tar = l_grasp * u_

        J_grasp = (R_.T @ p_J_soro)[:-1]

        grasp_err = (R_.T @ (p_plat_EE_tar - p_plat_EE))[:-1]

        pbar.set_description(
            "update_number:{}, \
                p_ik_err:{:.2E},\
                    w_ik_err:{:.2E},\
                        grasp_err:{:.2E}".format(
                                update_number,
                                norm(p_EE_ik_err),
                                norm(w_plat_ik_err),
                                norm(grasp_err),
                            ))


        # Break
        # if norm(w_plat_ik_err) < 0.01 and\
        #         norm(grasp_err) < 2e-3:
        #     break

        if norm(p_EE_ik_err) < 2e-3 and\
            norm(w_plat_ik_err) < 0.01 and\
                norm(grasp_err) < 5e-3:
            break
            print("Done")


        # Or Solve & Update
        A = []
        b = []

        A.append(np.hstack([J_UR, J_soro]))
        A.append(np.hstack([np.zeros((len(J_grasp),6),dtype=np.float32), J_grasp]))


        b.append(np.vstack([p_EE_ik_err,w_plat_ik_err]))
        b.append(10*grasp_err)

        # 일직선에 있도록
        A.append(np.hstack([J_plat, np.zeros((2,2))]))
        b.append(np.zeros((2,1)))

        A = np.vstack(A).astype(np.float32)
        b = np.vstack(b).astype(np.float32)

        J_use=A; ik_err=b; lambda_rate = 0.01

        lambda_min = 1e-6
        lambda_max = 1e-3

        ik_err_avg = np.mean(abs(ik_err))

        # Damping Term 
        lambda_ = lambda_rate * ik_err_avg + lambda_min 
        lambda_ = np.maximum(lambda_, lambda_max)

        n_ctrl = (J_use).shape[1]
        # Lamda Scheduling 
        J_culumn_sum = abs(np.sum(J_use, axis =0))

        for j in range(len(J_culumn_sum)):
            for i in J_culumn_sum:
                idx_nz = j
                J_use_nz = J_use[:,idx_nz].reshape(1, -1)
                det_J = np.linalg.det(np.matmul(J_use_nz, J_use_nz.T))
                if i >0.1:
                    if det_J > 1e-3:
                        lambda_=1e-4
                    elif det_J < 1e-20:
                        lambda_ = lambda_max
                        
        J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl).astype(np.float32)
        dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)


        step_size = step_size * 0.999
        dq = step_size * dq_raw

        dq = dq.flatten()
        qs = qs + dq[:6]
        motor_control_np = motor_control_np+ dq[6:] * scale_rate

        VIZ = False
        if VIZ:
            radius = 0.01
            obj_info_list[0] = make_markers(name=f"maker_{i}",
                                            type="sphere",  
                                            pos= p_EE_tar.flatten().tolist(),
                                            rot=[0,0,0], 
                                            size=[radius, radius, radius], 
                                            color=[0,0,1,1])

            viz_robot(chain_ur, soro, motor_control, obj_info_list, offset=offset, render_time=0.1)
        pbar.update()
        update_number = update_number + 1


    radius = 0.01
    obj_info_list.append(
        make_markers(name=f"maker_{i}",
                    type="sphere",  
                    pos= p_EE.flatten().tolist(),
                    rot=[0,0,0], 
                    size=[radius, radius, radius], 
                    color=[1,0,0,1])) 
    print(motor_control_np * scale_rate)
    result_motor_control_list.append(motor_control_np * scale_rate)
    result_qs_list.append(qs)
    result_p_EE_list.append(p_EE.copy())

# %%
result_motor_control_array = np.array(result_motor_control_list)
result_qs_array = np.array(result_qs_list)
result_p_EE_array = np.array(result_p_EE_list)

alpha = 1
for p_EE in result_p_EE_array:

    plt.scatter(-p_EE[1], p_EE[0], c='r', alpha = alpha)
    alpha = alpha * 0.9

plt.axis('equal')
plt.show()

alpha = 1
for p_EE in result_p_EE_array:

    plt.scatter(-p_EE[1], p_EE[2], c='r', alpha = alpha)
    alpha = alpha * 0.9

plt.axis('equal')
plt.show()



Path.mkdir((BASEDIR/f"control/planned_traj/{runname}"), parents=True, exist_ok=True)
np.save((   BASEDIR/f"control/planned_traj/{runname}/qs_array.npy").__str__(), result_qs_array)
np.save((   BASEDIR/f"control/planned_traj/{runname}/motor_array.npy").__str__(), result_motor_control_array)
np.save((   BASEDIR/f"control/planned_traj/{runname}/p_EE_cur_array.npy").__str__(), result_p_EE_array)
print("Planning Finished!")



# %%
from soro_kinematics_save import viz_robot, update_ur_q, make_markers
import numpy as np
import torch


qs_array       = np.load((BASEDIR/f"control/planned_traj/{runname}/qs_array.npy").__str__())
motor_array    = np.load((BASEDIR/f"control/planned_traj/{runname}/motor_array.npy").__str__())
p_EE_cur_array = np.load((BASEDIR/f"control/planned_traj/{runname}/p_EE_cur_array.npy").__str__())


motor_array = torch.tensor(motor_array)

# motor_array = motor_array[:10]
# qs_array = qs_array[:10]
# p_EE_cur_array = p_EE_cur_array[:10]

obj_info_list = []
for idx in range(len(qs_array)):
    update_ur_q(chain_ur, qs_array[idx])
    
    # if (idx//traj_n)%4==1:    
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
