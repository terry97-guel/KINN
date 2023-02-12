# %%
from pathlib import Path
import sys
import os
from path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR

_, RUNMODE = get_BASERDIR(Path(__file__).parent)

if RUNMODE is JUPYTER:
    os.chdir("..")
elif RUNMODE is DEBUG:
    pass
elif RUNMODE is RUN:
    pass

BASEDIR, _ = get_BASERDIR(Path(__file__).parent)

sys.path.append(str(BASEDIR))
BASEDIR, RUNMODE



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



soro = PRIMNET(args=args).to(args.device)
soro.load_state_dict(torch.load(BASEDIR.parent/'results'/path/"weights/epoch_500.pth"))
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


if RUN:
    """APPROACH"""
    qs = np.array([0,0,0,0,0,0]).astype(np.float32)
    motor_control_np = np.array([0,0,0,0]).astype(np.float32)
    
    grasp = 0.5
    rpy_EE_tar = np.array([PI, 0, 0])
    R_EE_tar = rpy2r_np(rpy_EE_tar)
    p_EE_tar = np.array([0.35,0,0.3]).reshape(3,1)
    
    
    l_tar = 0.15
    K_grasp = 1e9
    
    
    def generator():
        while True:
            yield

    from tqdm import tqdm
    pbar = tqdm(generator())
    for _ in pbar:
        # FK UR & soro
        chain_ur = update_ur_q(chain_ur, qs)
        motor_control = torch.tensor(motor_control_np).unsqueeze(0).to(device)
        T_EE_cur = get_EE_hybrid(chain_ur, soro, qs, motor_control)
        p_EE_cur, R_EE_cur = t2pr_np(T_EE_cur)
        
        
        
        ## IK for UR
        # get ik_error        
        p_ik_err = get_p_ik_err(p_EE_cur, p_EE_tar)
        w_ik_err = get_w_ik_err(R_EE_cur, R_EE_tar)
        

        # get jacobian
        p_J_UR = []
        for joint in chain_ur.joint[1:7]:
            assert joint.type == 'revolute'
            
            J_ = skew_np(joint.a) @ (p_EE_cur - joint.p)
            p_J_UR.append(J_)
        
        p_J_UR = np.hstack(p_J_UR)
        
        w_J_UR = []
        for joint in chain_ur.joint[1:7]:
            assert joint.type == 'revolute'
            
            J_ = joint.R @ joint.a
            w_J_UR.append(J_)
            
        w_J_UR = np.hstack(w_J_UR)
        
        J_UR = np.vstack([p_J_UR, w_J_UR])
        
        def get_EE_hybrid_grad(chain_ur:CHAIN, soro:PRIMNET, qs, motor_control):
            chain_ur = update_ur_q(chain_ur, qs)
            p_plat_ = torch.FloatTensor(chain_ur.joint[-1].p)
            R_plat_ = torch.FloatTensor(chain_ur.joint[-1].R)

            p_soro_ = soro(motor_control)[0, -1]
            p_EE_ = p_plat_ + R_plat_ @ p_soro_
            
            return p_EE_
        
        ## IK for SORO
        from torch.autograd.functional import jacobian
        from functools import partial
        
        p_J_soro = jacobian(partial(get_EE_hybrid_grad, chain_ur, soro, qs), motor_control)[:,0,0,:].detach().cpu().numpy() * 1000
        J_soro = np.vstack([p_J_soro, np.zeros((3,4))])
        
        ## Sphere constraint
        p_plat_EE = p_EE_cur - chain_ur.joint[8].p
        sph_l = np.linalg.norm(p_plat_EE)
        sph_err = np.array([1/2* (l_tar**2-sph_l**2)]).reshape(1,1)
        
        J_sph = p_plat_EE.T @ p_J_soro
        ## Grasp constraint
        grasp = 0.5
        grasp_dir = 0
        
        l_grasp = 0.04 * grasp
        u = chain_ur.joint[8].R[:,grasp_dir].reshape(3,1)
        J_grasp = u.T @ p_J_soro
        
        grasp_err = l_grasp - u.T @ p_plat_EE
        
        ## Motor constraint
        margin = 200
        llimit = (motor_control_np < margin).any()
        J_llimit = np.eye(4)[motor_control_np < margin]
        llimit_err = (margin-motor_control_np)[motor_control_np < margin].reshape(-1,1)
        
        ulimit = (motor_control_np > 2000-margin).any()
        J_ulimit = np.eye(4)[motor_control_np > 2000-margin]
        ulimit_err = ((2000-margin)-motor_control_np)[motor_control_np > 2000-margin].reshape(-1,1)

        # print("J_UR.shape :", J_UR.shape)
        # print("p_J_soro.shape :", p_J_soro.shape)
        # print("J_sph.shape :", J_sph.shape)
        # print("J_grasp.shape :", J_grasp.shape)
        # print("J_llimit.shape :", J_llimit.shape)
        # print("J_ulimit.shape :", J_ulimit.shape)
        
        from numpy.linalg import norm
        
        pbar.set_description(
            "p_ik_err:{:.2E},\
                w_ik_err:{:.2E},\
                    grasp_err:{:.2E},\
                        sph_err:{:.2E}".format(
                            norm(p_ik_err),
                            norm(w_ik_err),
                            norm(grasp_err),
                            norm(sph_err)
                        ))
            
        # Break
        if norm(p_ik_err) < 3e-3 and\
            norm(w_ik_err) < 0.01 and\
                norm(grasp_err) < 0.01 and\
                    norm(sph_err) < 0.01:
            break
        # Or Solve & Update
        A = []
        b = []
        A.append(np.hstack([J_UR, J_soro]))
        A.append(np.hstack([np.zeros((1,6)), J_grasp]))
        A.append(np.hstack([np.zeros((1,6)), J_sph]))
        
        b.append(np.vstack([p_ik_err,w_ik_err]))
        b.append(grasp_err)
        b.append(sph_err)
        if llimit:
            oor_motor_num = J_llimit.shape[0]
            A.append(np.hstack([np.zeros((oor_motor_num,6)), J_llimit]))
            b.append(llimit_err)
        if ulimit:
            oor_motor_num = J_ulimit.shape[0]
            A.append(np.hstack([np.zeros((oor_motor_num,6)), J_ulimit]))
            b.append(ulimit_err)
        
        A = np.vstack(A)
        b = np.vstack(b)
        
        # print(A.shape)
        # print(b.shape)
        
        J_use=A; ik_err=b; step_size=0.1; lambda_rate = 0.01
        
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
                        
        labmda_ = 0.1
        J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl)
        dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)

        dq = step_size * dq_raw
        
        dq = dq.astype(np.float32).flatten()
        qs = qs + dq[:6]
        motor_control_np = motor_control_np+ dq[6:]
    


# %%
    