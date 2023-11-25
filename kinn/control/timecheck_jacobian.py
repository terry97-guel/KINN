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
from kinn.utils.args import read_ARGS
from kinn.model.PRIMNET import PRIMNET
import torch


parser = argparse.ArgumentParser(description= 'parse for DLPG')
parser.add_argument("--configs", default="PRIMNET/FINGER.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET] # [FINGER, ABAQUS]
args= parser.parse_args([])
args = read_ARGS((BASEDIR/'configs'/args.configs).absolute())



soro = PRIMNET(args=args).to(args.device)
soro.load_state_dict(torch.load(BASEDIR.parent/'results'/path/"weights/epoch_200.pth"))
device = args.device
# %%
import numpy as np
PI = np.pi

file_name = (BASEDIR / "control/urdf/ur5e_soro.urdf").__str__()
assert os.path.exists(file_name)

from kinematics.class_structure import CHAIN
chain_ur = CHAIN(file_name = file_name)


# %%

def idxs_from_name(chain:CHAIN, names):
    idxs = []
    
    for name in names:
        for idx,joint in enumerate(chain.joint):
            # if joint.name ==
            chain.joint[0].name


def update_ur_q(chain:CHAIN, qs):
    assert len(qs) == 6

    for idx in range(6):
        chain.joint[1+idx].q = float(qs[idx])
    
    chain.fk_chain(1)

    return chain

@ torch.no_grad()
def fk_soro(soro:PRIMNET, motor_control):
    assert motor_control.shape == (1,4)
    
    T_soro = soro.forward(motor_control, OUTPUT_POSITION=False)
    return T_soro[0,-1]


def t2pr_np(T):
    p = T[:3,3].reshape(3,1)
    R = T[:3,:3]
    return p, R

def pr2t_np(p,r):
    T = np.eye(4)
    T[:3,:3] = r
    T[:3,3] = p.flatten()
    return T

def get_EE_hybrid(chain_ur:CHAIN, soro:PRIMNET, qs, motor_control):
    chain_ur = update_ur_q(chain_ur, qs)
    p_plat = chain_ur.joint[-1].p
    R_plat = chain_ur.joint[-1].R

    T_plat = pr2t_np(p_plat, R_plat)
    T_soro = fk_soro(soro, motor_control).detach().cpu().numpy()

    return T_plat @ T_soro


def rpy2r_np(rpys):
    R = np.zeros((3,3)).astype(np.float32)
    r = rpys[0]
    p = rpys[1]
    y = rpys[2]
    R[0,:] = np.array([
        np.cos(y)*np.cos(p),
        -np.sin(y)*np.cos(r) + np.cos(y)*np.sin(p)*np.sin(r),
        np.sin(y)*np.sin(r) + np.cos(y)*np.sin(p)*np.cos(r)
        ])
    R[1,:] = np.array([
        np.sin(y)*np.cos(p),
        np.cos(y)*np.cos(r) + np.sin(y)*np.sin(p)*np.sin(r),
        -np.cos(y)*np.sin(r) + np.sin(y)*np.sin(p)*np.cos(r)
        ])
    R[2,:] = np.array([
        -np.sin(p),
        np.cos(p)*np.sin(r),
        np.cos(p)*np.cos(r)
        ])
    return R

def get_p_ik_err(p_EE_cur, p_EE_tar):
    assert p_EE_cur.shape == p_EE_tar.shape
    return p_EE_tar - p_EE_cur

def get_w_ik_err(R_EE_cur, R_EE_tar):
    assert R_EE_cur.shape == R_EE_tar.shape

    Rerr = R_EE_cur.T @ R_EE_tar

    werr = R_EE_cur @ r2w_np(Rerr)
    return werr


def r2w_np(R):
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w

def skew_np(vec):
    assert vec.shape==(3,1)
    x = vec[0,0]
    y = vec[1,0]
    z = vec[2,0]
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def r2rpy_np(R):
    assert R.shape == (3,3)
    r = np.arctan2(R[2,1], R[2,2])
    p = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    y = np.arctan2(R[1,0], R[0,0])

    rpy = np.array([r,p,y]).reshape(-1)
    return rpy


def t2r_np(T):
    R = T[0:3,0:3]
    return R
    
def t2p_np(T):
    p = T[0:3,3]
    return p.reshape(3,1)

# %%
scale_rate = 70
l_tar = 0.15

from tqdm import tqdm
from kinn.model.PRIMNET import PRIMNET, Fjoint, Tjoint, Rjoint, Pjoint
from jacobian import jacobian
from functools import partial
from numpy.linalg import norm

# Init values
qs= np.array([0,-90,90,-90,-90, 0]).astype(np.float32) / 180 * PI
motor_control_np = np.array([0,0,0,0]).astype(np.float32)

qs_list = []
motor_list = []
target_position_list = []
target_rpy_list = []
traj_n = 10


def forward_model(p_plat, R_plat, soro, motor_control):
    p_plat = torch.FloatTensor(p_plat)
    R_plat = torch.FloatTensor(R_plat)
    p_soro_ = soro(motor_control)[0, -1]
    p_EE = p_plat + R_plat @ p_soro_
    return p_EE

def get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control):
    dp_dm, p_EE = jacobian(partial(forward_model, p_plat, R_plat, soro), motor_control)
    return (
        np.array(dp_dm[:,0,0,:].detach().to(device)).astype(np.float32) * scale_rate,
        np.array(p_EE[0].detach().to(device)).astype(np.float32)
        )

def forward_q(model:PRIMNET, motor_control):
    motor_control = model.normalize(motor_control)

    # Forward
    act_embeds = model.ACT_EMBED.layers(motor_control)
    q_values = model.FK_LAYER.forward_q(act_embeds)[0]
    return q_values

from kinn.utils.pyart import t2r
@ torch.jit.script_if_tracing
def kinematic_grad(soro:PRIMNET, q_values):
    joint_se3 = soro.FK_LAYER.forward_kinematics(q_values)

    joint_position =  soro.t2p(joint_se3, OUTPUT_NORMALIZE=False)[0,:,:,0]
    joint_rotation = t2r(joint_se3[0])

    EE_pos = joint_position[-1]

    dp_dq = torch.zeros(3, len(soro.FK_LAYER.joints))

    for idx,joint in enumerate(soro.FK_LAYER.joints):
        if isinstance(joint, Fjoint):
            continue
        elif isinstance(joint, Tjoint):
            continue
        elif isinstance(joint, Rjoint):
            pos_diff = EE_pos - joint_position[idx]
            dp_dq[:, idx] = torch.cross(joint_rotation[idx] @ joint.axis.data[:,0], pos_diff)
            # print('here')
        elif isinstance(joint, Pjoint):
            pos_diff = EE_pos - joint_position[idx]
            dp_dq[:,idx] = joint_rotation[idx] @ joint.axis.data[:,0]
    
    return dp_dq, joint_position[-1]


def get_hybrid_grad_explicit(p_plat, R_plat, soro:PRIMNET, motor_control):
    dq_dm,q_values_tuple = jacobian(partial(forward_q, soro), motor_control)
    dq_dm = dq_dm[:,0,:].detach().numpy(); q_values = q_values_tuple[0].detach().unsqueeze(0)
    
    with torch.no_grad():
        dp_dq, p_soro= kinematic_grad(soro, q_values)
        
    dp_dq = dp_dq.numpy(); 
    p_soro = p_soro.numpy().reshape(3,1)
    p_EE = p_plat + R_plat @ p_soro
    return R_plat@dp_dq@dq_dm* scale_rate, p_EE


# %%

import time
n = 100
p_plat = np.zeros((3,1), dtype=np.float32)
R_plat = np.eye(3, dtype=np.float32)
motor_control = torch.FloatTensor([[0,0,0,0]]).to(device)

start_time = time.time()
for _ in range(n):
    get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control)
end_time = time.time()
autograd_time = end_time - start_time
print("[AutoGrad]:{:2f}".format(autograd_time))

start_time = time.time()
for _ in range(n):
    get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control)
end_time = time.time()
explicit_time = end_time-start_time
print("[Explicit]:{:2f}".format(explicit_time))
