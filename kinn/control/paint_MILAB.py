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


from soro_kinematics import update_ur_q, forward_model, r2rpy_np, solve_ik_traj, viz_robot, solve_ik, rpy2r_np

# %%
qs_list = []
motor_list = []
target_position_list = []
target_rpy_list = []
traj_n = 3
scale_rate = 30

RUN = True
VIZ = False
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

p_offset_np =  np.array([0.82545507,0.09943377,0], dtype=np.float32)

# %%
# Get Circle Trajectory
traj = Traj()

def get_circle_traj(traj:Traj,center_x,center_y,r, start_theta, end_theta, resolution=20, z_down = 0.545):
    pos_list = []
    u_list = []

    center_x = center_x/1000
    center_y = center_y/1000
    r = r/1000

    theta = np.linspace(start_theta, end_theta, resolution, dtype=np.float32)
    x_unit = np.cos(theta)
    y_unit =  np.sin(theta)

    pos = np.vstack([r*x_unit,r*y_unit,np.ones_like(x_unit) * z_down]).T
    u = np.vstack([-y_unit, x_unit,np.zeros_like(x_unit)]).T

    pos_list.append(pos)
    u_list.append(u)

    pos_array = np.vstack(pos_list)  + np.array([center_x,center_y,0], dtype=np.float32)
    # pos_array = pos_array.reshape(-1,3,1)
    u_array = np.vstack(u_list)

    grasp_array = np.ones_like(u_array[:,0], dtype=np.float32)

    assert len(pos_array) == len(u_array) == len(grasp_array)
    traj.append(pos_array, u_array, grasp_array)

    return traj



center_x = 55
center_y = 0


r = 2
start_theta = PI/2
end_theta = 3*PI/2

traj = get_circle_traj(traj, center_x,center_y,r,start_theta, end_theta, resolution=20)
grasp_array, pos_array, u_array = traj.get_array()

plt.plot(pos_array[:,0], pos_array[:,1])
plt.axis('equal')


# %%
# Get Stroke Trajectory
traj = Traj()

def linear_traj(traj:Traj,start_x, start_y, end_x, end_y, STROKE, resolution=10, z_down = 0.545):
    pos_list = []
    u_list = []
    grasp_list = []


    start_pos = np.array([start_x/1000,start_y/1000,z_down], dtype=np.float32)
    end_pos = np.array([end_x/1000,end_y/1000,z_down], dtype=np.float32)

    pos_array_ = np.linspace(start_pos, end_pos, resolution, dtype=np.float32)
    pos_list.append(pos_array_)

    u_ = end_pos - start_pos
    u_ = u_/np.linalg.norm(u_)
    u_list.append([u_ for _ in range(int(resolution))])

    if STROKE:
        grasp_array_ = np.linspace(-1,1,resolution, dtype=np.float32)
    else:
        grasp_array_ = -np.ones(resolution, dtype=np.float32)

    grasp_list.append(grasp_array_)


    pos_array = np.vstack(pos_list)
    u_array = np.vstack(u_list)
    grasp_array = np.hstack(grasp_list)

    assert len(pos_array) == len(u_array) == len(grasp_array)
    traj.append(pos_array, u_array, grasp_array)

    return traj


# STROKE = True
STROKE = False


start_x = 55
start_y = 0

end_x = 0
end_y = 0

traj = linear_traj(traj, start_x,start_y, end_x, end_y, STROKE, resolution=10)
grasp_array, pos_array, u_array = traj.get_array()

plt.plot(pos_array[:,0], pos_array[:,1])
plt.axis('equal')


# %%
traj = Traj()

def approach_traj(traj:Traj, start_x, start_y, end_x, end_y, APPROACH, resolution=10, z_down = 0.545):
    pos_list = []
    u_list = []
    grasp_list = []

    z_up = z_down + 50/1000 

    pre_pos = np.array([start_x/1000,start_y/1000,z_up], dtype=np.float32)
    start_pos = np.array([start_x/1000,start_y/1000,z_down], dtype=np.float32)
    end_pos = np.array([end_x/1000,end_y/1000,z_down], dtype=np.float32)
    after_pos = np.array([end_x/1000,end_y/1000,z_up], dtype=np.float32)
    



    if APPROACH:
        # Approach
        thrid_resolution = int(resolution/3)
        grasp_array_ = np.zeros(resolution, dtype=np.float32)
        grasp_array_[thrid_resolution:2*thrid_resolution] =  np.linspace(0,-1, thrid_resolution, dtype=np.float32)
        grasp_array_[2*thrid_resolution:] = -np.ones(resolution-2*thrid_resolution, dtype=np.float32)
        pos_array_ = np.linspace(pre_pos, start_pos, resolution, dtype=np.float32)

    else:
        # Approach
        thrid_resolution = int(resolution/3)
        grasp_array_ = np.ones(resolution, dtype=np.float32)

        grasp_array_[thrid_resolution:2*thrid_resolution] =  np.linspace(1,0, thrid_resolution, dtype=np.float32)
        grasp_array_[2*thrid_resolution:] = np.zeros(resolution-2*thrid_resolution, dtype=np.float32)
        pos_array_ = np.linspace(end_pos, after_pos, resolution, dtype=np.float32)


    pos_list.append(pos_array_)


    u_ = end_pos - start_pos
    u_ = u_/np.linalg.norm(u_)
    u_list.append([u_ for _ in range(int(resolution))])

    grasp_list.append(grasp_array_)


    pos_array = np.vstack(pos_list)
    u_array = np.vstack(u_list)
    grasp_array = np.hstack(grasp_list)

    assert len(pos_array) == len(u_array) == len(grasp_array)
    traj.append(pos_array, u_array, grasp_array)

    return traj


APPROACH = False
# APPROACH = True

start_x = 55
start_y = 0

end_x = 0
end_y = 0


traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH, resolution=10)
grasp_array, pos_array, u_array = traj.get_array()

# make plt 3d plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(pos_array[:,0], pos_array[:,1], pos_array[:,2])
ax.axis('equal')



# %%
def plot_pos(traj:Traj):
    grasp_array, pos_array, u_array = traj.get_array()

    # make plt 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    p_offset_np =  np.array([0.82545507,0.09943377,0], dtype=np.float32)

    pos_array = (pos_array + p_offset_np)
    pos_array = pos_array.reshape(-1,3,1)


    for pos_array_ in traj.pos_list:
        ax.plot(pos_array_[:,0], pos_array_[:,1], pos_array_[:,2])
        ax.axis('equal')
    plt.show()

    # make plt 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(pos_array[:,0], pos_array[:,1], pos_array[:,2])
    ax.axis('equal')
    plt.show()

# %%
# M
runname = "M"
traj = Traj()

# Stroke 1
start_x = 53; start_y = 16-5
end_x = 0; end_y = 16-5
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down = z_down)
traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down = z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down = z_down)

# Stroke 2
start_x = 53; start_y = 18-5
end_x = 4; end_y = -7.5-5
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down = z_down)
traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down = z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down = z_down)

# Stroke 3
start_x = 10; start_y = -7.5-5
end_x = 59; end_y = -31-5
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down = z_down)
traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down = z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down = z_down)

# Stroke 4
start_x = 53; start_y = -27-5
end_x = 0; end_y = -27-5
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down = z_down)
traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down = z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down = z_down)


plot_pos(traj)

# %%
# I
runname = "I"
traj = Traj()

# Stroke 1
start_x = 55; start_y = -62
end_x = 55; end_y=-82
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down=z_down)



# Stroke 2
start_x = 50; start_y = -72
end_x = 5; end_y=-72
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down=z_down)

# Stroke 3
start_x = 0; start_y = -62
end_x = 0; end_y=-82
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down=z_down)


plot_pos(traj)


# %%
# L
runname = "L"
traj = Traj()

# Stroke 1
start_x = 55; start_y = -102
end_x = 4; end_y=-102
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down=z_down)

# Stroke 2
start_x = 0; start_y = -102
end_x = 0; end_y=-140

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10, z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10, z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10, z_down=z_down)


plot_pos(traj)

# %%
# A
runname = "A"

traj = Traj()

# Stroke 1
start_x = 60; start_y = -170
end_x = 0; end_y=-148
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10,z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10,z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10,z_down=z_down)

# Stroke 2
start_x = 60; start_y = -170
end_x = 0; end_y=-192

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10,z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10,z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10,z_down=z_down)

# Stroke 3
start_x = 18; start_y = -155
end_x = 18; end_y=-185

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10,z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10,z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10,z_down=z_down)

plot_pos(traj)

# %%
# B
runname = "B"
traj = Traj()

# Stroke 1
start_x = 52; start_y = -213
end_x = 5; end_y= -213
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10,z_down=z_down)
traj  = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10,z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10,z_down=z_down)

# Stroke 2
start_x = 55; start_y = -213
end_x = 55; end_y= -239
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10,z_down=z_down)
traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=False, resolution=10,z_down=z_down)


center_x = 41
center_y = -239
r = 14
start_theta = 0
end_theta = -PI
z_down = 0.500

traj = get_circle_traj(traj, center_x,center_y,r,start_theta, end_theta, resolution=20,z_down=z_down)


start_x = 27; start_y = -239
end_x = 27; end_y= -200
z_down = 0.505

traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10,z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10,z_down=z_down)

# Stroke 3
start_x = 27; start_y = -239
end_x = 27; end_y= -241
z_down = 0.505

traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=True, resolution=10,z_down=z_down)
traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=False, resolution=10,z_down=z_down)

center_x = 27 - 13.5
center_y = -241
r = 13.5
start_theta = 0
end_theta = -PI
z_down = 0.500

traj = get_circle_traj(traj, center_x,center_y,r,start_theta, end_theta, resolution=20,z_down=z_down)


start_x = 0; start_y = -241
end_x = 0; end_y= -205
z_down = 0.505

traj = linear_traj(traj, start_x, start_y, end_x, end_y, STROKE=True, resolution=10,z_down=z_down)
traj = approach_traj(traj, start_x, start_y, end_x, end_y, APPROACH=False, resolution=10,z_down=z_down)


plot_pos(traj)

# %%
grasp_array, pos_array, u_array = traj.get_array()

pos_array = pos_array + p_offset_np
pos_array = pos_array.reshape(-1,3,1)

def get_ur_jacobian(chain_ur):
    ## get jacobian
    # position jacobian
    p_J_UR = []
    for joint in chain_ur.joint[1:7]:
        assert joint.type == 'revolute'
        
        J_ = skew_np(joint.a)@(p_EE-joint.p)
        p_J_UR.append(J_)
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

# Init values
qs= np.array([0,-90,90,-90,-90, 0]).astype(np.float32) / 180 * PI
chain_ur = update_ur_q(chain_ur, qs)
motor_control_np = np.array([0,0]).astype(np.float32)


from soro_kinematics import *

result_motor_control_list = []
result_qs_list = []
result_p_EE_list = []
obj_info_list = []
for i in range(len(grasp_array)):

    grasp    = grasp_array[i]
    u        = u_array[i]
    p_EE_tar = pos_array[i]
    step_size = 0.2


    pbar = tqdm(generator(), leave=True)
    update_number = 0

    for _ in pbar:
        # Update qs
        chain_ur = update_ur_q(chain_ur, qs)
        motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
        p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

        # get Jacobian
        p_J_soro, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control, scale_rate)
        J_UR = get_ur_jacobian(chain_ur)
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
        if norm(p_EE_ik_err) < 1e-3 and\
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

        # if  (np.linalg.norm(dq_raw[6:] * scale_rate) < 3e-2) and norm(grasp_err) > 0.015:
        #     # qs = np.array([0,-90,90,-90,-90, 0]).astype(np.float32) / 180 * PI
        #     motor_control_np = np.zeros_like(motor_control_np)

        # if norm(p_EE_ik_err) < 3e-3 and update_number > 300:
            # break
            # print("Done")

            
        # if update_number % 100 == 99:
            # motor_control_np = np.zeros_like(motor_control_np)
            # update_number = 0
        step_size = step_size * 0.99
        dq = step_size * dq_raw

        dq = dq.flatten()
        qs = qs + dq[:6]
        motor_control_np = motor_control_np+ dq[6:] * scale_rate

        VIZ = False
        if VIZ:
            viz_robot(chain_ur, soro, motor_control, obj_info_list)
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
        # return motor_control_np * scale_rate


result_motor_control_array = np.array(result_motor_control_list)
result_qs_array = np.array(result_qs_list)
result_p_EE_array = np.array(result_p_EE_list)

Path.mkdir((BASEDIR/f"control/planned_traj/{runname}"), parents=True, exist_ok=True)
np.save((   BASEDIR/f"control/planned_traj/{runname}/qs_array.npy").__str__(), result_qs_array)
np.save((   BASEDIR/f"control/planned_traj/{runname}/motor_array.npy").__str__(), result_motor_control_array)
np.save((   BASEDIR/f"control/planned_traj/{runname}/p_EE_cur_array.npy").__str__(), result_p_EE_array)
print("Planning Finished!")



# %%
from soro_kinematics import viz_robot, update_ur_q, make_markers
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