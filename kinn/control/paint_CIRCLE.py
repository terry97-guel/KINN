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
VIZ = True

# %%
from matplotlib import pyplot as plt
z_down = 0.545
z_up = z_down + 50/1000 # 50mm up

rpy_plat_tar = np.array([-3.14079618e+00, -4.37113875e-08, -1.57079625e+00], dtype=np.float32)
R_plat_tar = rpy2r_np(rpy_plat_tar)

pos_list = []
u_list = []

# Get Circle Trajectory
r = 2/1000
theta = np.linspace(0, -2*PI, 20, dtype=np.float32)
x_unit = np.cos(theta)
y_unit =  np.sin(theta)

u1 = -np.sin(theta)
u2 = np.cos(theta)

pos = np.vstack([r*x_unit,r*y_unit,np.ones_like(x_unit) * z_down]).T
u = np.vstack([-y_unit, x_unit,np.zeros_like(x_unit)]).T

pos_list.append(pos)
u_list.append(u)

pos_array = np.vstack(pos_list) + np.array([0.82545507,0.09943377,0], dtype=np.float32)
pos_array = pos_array.reshape(-1,3,1)
u_array = np.vstack(u_list)


grasp_array = np.ones_like(u_array[:,0], dtype=np.float32)

assert len(pos_array) == len(u_array) == len(grasp_array)
pos_array.shape, u_array.shape, grasp_array.shape

# %%
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

# %%
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

runname = "circle"
Path.mkdir((BASEDIR/f"control/planned_traj/{runname}"), parents=True, exist_ok=True)
np.save((   BASEDIR/f"control/planned_traj/{runname}/qs_array.npy").__str__(), result_qs_array)
np.save((   BASEDIR/f"control/planned_traj/{runname}/motor_array.npy").__str__(), result_motor_control_array)
np.save((   BASEDIR/f"control/planned_traj/{runname}/p_EE_cur_array.npy").__str__(), result_p_EE_array)
print("Planning Finished!")


# %%

# %%





# %%



grasp_dir = u


R_EE_tar = rpy2r_np(rpy_EE_tar)

pbar = tqdm(generator(), leave=True)
update_number = 0
for _ in pbar:
    ## FK UR & SORO
    chain_ur = update_ur_q(chain_ur, qs)
    motor_control = torch.tensor(scale_rate * motor_control_np).unsqueeze(0).cpu()
    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

    # dp_dm, p_EE = get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control, scale_rate)
    dp_dm, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control, scale_rate)
    R_EE_cur = R_plat
    p_EE_cur = p_EE
    
    ## IK for UR
    # get ik_error
    p_ik_err = get_p_ik_err(p_EE_cur, p_EE_tar)
    w_ik_err = get_w_ik_err(R_EE_cur, R_EE_tar)

    ## get jacobian
    # position jacobian
    p_J_UR = []
    for joint in chain_ur.joint[1:7]:
        assert joint.type == 'revolute'
        
        J_ = skew_np(joint.a)@(p_EE_cur-joint.p)
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

    p_J_soro = dp_dm
    J_soro = np.vstack([p_J_soro, np.zeros((3,4), dtype=np.float32)])

    ## Grasp constraint
    p_plat_EE = (p_EE_cur - chain_ur.joint[8].p).astype(np.float32)
    l_grasp = 0.01 * grasp
    assert np.abs(np.linalg.norm(grasp_dir) - 1) < 1e-3
    assert grasp_dir.shape == (3,)
    R_ = chain_ur.joint[8].R.astype(np.float32)
    u = grasp_dir.reshape(3,1)
    p_plat_EE_tar = l_grasp * u
    
    # u = (grasp_dir[0] * R_[:,0] + grasp_dir[1] * R_[:,1] + grasp_dir[2] * R_[:,2] ).reshape(3,1)
    # u = chain_ur.joint[8].R[:,grasp_dir].reshape(3,1).astype(np.float32)
    
    J_grasp = (R_.T @ p_J_soro)[:-1]
    
    grasp_err = (R_.T @ (p_plat_EE_tar - p_plat_EE))[:-1]
    
    ## Motor constraint
    margin = 200/scale_rate
    llimit = (motor_control_np < margin).any()
    J_llimit = np.eye(4, dtype=np.float32)[motor_control_np < margin].astype(np.float32)
    llimit_err = (margin-motor_control_np)[motor_control_np < margin].reshape(-1,1)
    
    ulimit = (motor_control_np > 2000/scale_rate-margin).any()
    J_ulimit = np.eye(4, dtype=np.float32)[motor_control_np > 2000/scale_rate-margin]
    ulimit_err = ((2000/scale_rate-margin)-motor_control_np)[motor_control_np > 2000/scale_rate-margin].reshape(-1,1)

    
    pbar.set_description(
        "update_number:{}, \
            p_ik_err:{:.2E},\
                w_ik_err:{:.2E},\
                    grasp_err:{:.2E},\
                        sph_err:{:.2E}".format(
                            update_number,
                            norm(p_ik_err),
                            norm(w_ik_err),
                            norm(grasp_err),
                        ))
            
    # Break
    if norm(p_ik_err) < 1e-3 and\
        norm(w_ik_err) < 0.01 and\
            norm(grasp_err) < 5e-3:
        break
    # Or Solve & Update
    A = []
    b = []
    A.append(np.hstack([J_UR, J_soro]))
    A.append(np.hstack([np.zeros((len(J_grasp),6),dtype=np.float32), J_grasp]))
    
    b.append(np.vstack([p_ik_err,w_ik_err]))
    b.append(10*grasp_err)
    if llimit:
        oor_motor_num = J_llimit.shape[0]
        A.append(np.hstack([np.zeros((oor_motor_num,6)), J_llimit]))
        b.append(llimit_err)
    if ulimit:
        oor_motor_num = J_ulimit.shape[0]
        A.append(np.hstack([np.zeros((oor_motor_num,6)), J_ulimit]))
        b.append(ulimit_err)
    
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

    if norm(p_ik_err) < 3e-3 and update_number > 300:
        break

        
    # if update_number % 100 == 99:
        # motor_control_np = np.zeros_like(motor_control_np)
        # update_number = 0
    step_size = step_size * 0.99
    dq = step_size * dq_raw
    
    dq = dq.flatten()
    qs = qs + dq[:6]
    motor_control_np = motor_control_np+ dq[6:] * scale_rate
    
    if VIZ:
        viz_robot(chain_ur, soro, motor_control)
    pbar.update()
    update_number = update_number + 1





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