# %%
from pathlib import Path
import sys
import os
from path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR


BASEDIR, RUNMODE = get_BASERDIR(__file__)
if RUNMODE is JUPYTER:
    print("Going up one level")
    os.chdir((BASEDIR/"..").__str__())
elif RUNMODE is DEBUG:
    os.chdir((BASEDIR/"code").__str__())

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

RUN = False
scale_rate = 50

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
    iternum = 0
    for _ in pbar:
        # FK UR & soro
        chain_ur = update_ur_q(chain_ur, qs)
        motor_control = torch.tensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
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
        
        p_J_soro = jacobian(partial(get_EE_hybrid_grad, chain_ur, soro, qs), motor_control)[:,0,0,:].detach().cpu().numpy() * scale_rate
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
        margin = 200/scale_rate
        llimit = (motor_control_np < margin).any()
        J_llimit = np.eye(4)[motor_control_np < margin]
        llimit_err = (margin-motor_control_np)[motor_control_np < margin].reshape(-1,1)
        
        ulimit = (motor_control_np > 2000/scale_rate-margin).any()
        J_ulimit = np.eye(4)[motor_control_np > 2000/scale_rate-margin]
        ulimit_err = ((2000/scale_rate-margin)-motor_control_np)[motor_control_np > 2000/scale_rate-margin].reshape(-1,1)

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
        motor_control_np = motor_control_np+ dq[6:] * scale_rate



# %%
from tqdm import tqdm
scale_rate = 50
l_tar = 0.15

VIZ = True
RUN = True

if VIZ:
    import rospy
    from rospy import Publisher
    from kinematics.structure_utils import  \
        get_p_chain, get_R_chain, get_rpy_from_R_mat, get_mesh_chain, get_scale, get_link_color, get_viz_ingredients
    from kinematics.rviz import publish_viz_robot, publish_viz_markers
    from visualization_msgs.msg import Marker, MarkerArray
    
    rospy.init_node("VIZ_ROBOT")
    pub_robot     = Publisher(f'viz_robot', MarkerArray, queue_size=10)
    pub_markers   = Publisher(f'viz_markers', MarkerArray, queue_size=10)
    
    def publish_robot(pub_robot:Publisher,chain:CHAIN):
        # update_q_chain(self.chain.joint, q_list, self.ctrl_joint_num)
        chain.fk_chain(1)
        p_list     = get_p_chain(chain.joint)
        R_list     = get_R_chain(chain.joint)
        rpy_list   = get_rpy_from_R_mat(R_list)
        mesh_list  = get_mesh_chain(chain.link)
        scale_list = get_scale(chain.link)
        color_list = get_link_color(chain.link)
        viz_links  =  get_viz_ingredients(p_list, rpy_list, mesh_list, scale_list, color_list)
        viz_trg_robot = publish_viz_robot(viz_links)
        pub_robot.publish(viz_trg_robot)

    def publish_soro(pub_markers:Publisher,chain_ur:CHAIN, soro:PRIMNET, motor_control):
        obs_info_lst = []
        
        black = [0.1,0.1,0.1,0.3]
        white = [0.8,0.8,0.8,1]

        # Visualize Platform
        pos_fr = chain_ur.joint[-2].p.astype(np.float32)
        pos_to = chain_ur.joint[-1].p.astype(np.float32)
        name = "platform"

        obs_info_lst.append(get_cylinder_from_axis(pos_fr, pos_to, 0.08, black, name))

        with torch.no_grad():
            ps_ = soro(motor_control)[0].detach().cpu().numpy()
            p_plat = chain_ur.joint[-1].p.astype(np.float32)
            R_plat = chain_ur.joint[-1].R.astype(np.float32)
            ps_ = np.vstack((np.zeros((1,3,1), dtype=np.float32), ps_))
            ps = p_plat + R_plat @ ps_
            
        i = 0
        for pos_fr, pos_to in zip(ps[:-1], ps[1:]):
            i = i+1
            name = f"soro_joint_{i}"
            obs_info_lst.append(get_cylinder_from_axis(pos_fr, pos_to, 0.02, white, name))

        viz_obj = publish_viz_markers(obs_info_lst)
        pub_markers.publish(viz_obj)

    def make_markers(name, type, pos, rot, size, color): 
        return {"name":name, "type":type, "info":pos+rot+size, "color":color}

    def get_cylinder_from_axis(pos_fr, pos_to, radius, color, name = None):
        pos_fr =pos_fr.flatten(); pos_to=pos_to.flatten()
        pos_del = pos_to - pos_fr
        length = np.linalg.norm(pos_del)

        if length < 10e-7:
            link  = make_markers(name=name, 
                                    type="cylinder",  
                                    pos=pos_fr.tolist(),
                                    rot=[0,0,0], 
                                    size=[radius, radius, length], 
                                    color=color)
            return link

        pos_del_norm = pos_del/length
        r = np.arctan2(-pos_del_norm[1], np.sqrt(pos_del_norm[0]**2 + pos_del_norm[2]**2))
        p = np.arctan2(pos_del_norm[0], pos_del_norm[2])

        R_link = rpy2r_np(np.array([r,p,0]))

        rpy = r2rpy_np(R_link)

        length = np.linalg.norm(pos_to - pos_fr)
        pos = pos_fr + pos_del/2

        link  = make_markers(name=name, 
                                type="cylinder",  
                                pos=pos.tolist(),
                                rot=rpy.tolist(), 
                                size=[radius, radius, length], 
                                color=color)

        return link


def generator():
    while True:
        yield

# %%
from model.PRIMNET import PRIMNET, Fjoint, Tjoint, Rjoint, Pjoint
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

from utils.pyart import t2r
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
"""
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

"""
# %%
    
if True:
    """APPROCAH"""
    grasp_dir = 1
    
    chain_ur = update_ur_q(chain_ur, qs)
    motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
    
    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().to(device)
    R_EE_cur = R_plat
    
    grasp_init = 0.0
    rpy_EE_tar_init = r2rpy_np(R_EE_cur)
    p_EE_tar_init =  p_EE_cur
    
    ## 나중에 지울것 ##
    # qs_tar = np.array([0, -44, 63, -139, -92, 0]).astype(np.float32) / 180 * PI
    # chain_ur = update_ur_q(chain_ur, qs_tar)
    # motor_control = torch.FloatTensor(np.array([0,0,0,1000]).astype(np.float32)).unsqueeze(0).to(device)

    # p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    # p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    # p_EE_cur = p_EE_cur.detach().cpu().numpy()
    # R_EE_cur = R_plat
    ## 나중에 지울것 ##

    grasp_end = -0.8
    rpy_EE_tar_end = np.array([-2.61693358, -0.03022852, -1.58825493], dtype=np.float32)
    p_EE_tar_end = np.array([1.0205003, 0.13381146,0.6415896], dtype=np.float32).reshape(3,1)
    
    traj_n_ = 10
    for (grasp, rpy_EE_tar, p_EE_tar) in \
        tqdm(zip(\
            np.linspace(grasp_init, grasp_end, traj_n),\
            np.linspace(rpy_EE_tar_init, rpy_EE_tar_end, traj_n),
            np.linspace(p_EE_tar_init, p_EE_tar_end, traj_n))):
        
        # print(p_EE_tar)
        
        R_EE_tar = rpy2r_np(rpy_EE_tar)
        
        pbar = tqdm(generator())
        for _ in pbar:
            ## FK UR & SORO
            chain_ur = update_ur_q(chain_ur, qs)
            motor_control = torch.tensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
            p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

            # dp_dm, p_EE = get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control)
            dp_dm, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control)
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

            ## Sphere constraint
            p_plat_EE = (p_EE_cur - chain_ur.joint[8].p).astype(np.float32)
            sph_l = np.linalg.norm(p_plat_EE)
            sph_err = np.array([1/2* (l_tar**2-sph_l**2)], dtype=np.float32).reshape(1,1)
            
            J_sph = p_plat_EE.T @ p_J_soro
            ## Grasp constraint
            grasp = 0.5
            grasp_dir = 0
            
            l_grasp = 0.04 * grasp
            u = chain_ur.joint[8].R[:,grasp_dir].reshape(3,1).astype(np.float32)
            J_grasp = u.T @ p_J_soro
            
            grasp_err = l_grasp - u.T @ p_plat_EE
            
            ## Motor constraint
            margin = 200/scale_rate
            llimit = (motor_control_np < margin).any()
            J_llimit = np.eye(4, dtype=np.float32)[motor_control_np < margin].astype(np.float32)
            llimit_err = (margin-motor_control_np)[motor_control_np < margin].reshape(-1,1)
            
            ulimit = (motor_control_np > 2000/scale_rate-margin).any()
            J_ulimit = np.eye(4, dtype=np.float32)[motor_control_np > 2000/scale_rate-margin]
            ulimit_err = ((2000/scale_rate-margin)-motor_control_np)[motor_control_np > 2000/scale_rate-margin].reshape(-1,1)

            
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
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_grasp]))
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_sph]))
            
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
            
            A = np.vstack(A).astype(np.float32)
            b = np.vstack(b).astype(np.float32)
            
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
                            
            J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl).astype(np.float32)
            dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)

            dq = step_size * dq_raw
            
            dq = dq.flatten()
            qs = qs + dq[:6]
            motor_control_np = motor_control_np+ dq[6:] * scale_rate
            
            if VIZ:
                render_time = 0.1
                frequency = 60
                rate = rospy.Rate(frequency)
                max_rendering = frequency * render_time
                
                rendering = 0
                while not rospy.is_shutdown():
                    if rendering == max_rendering: break

                    publish_robot(pub_robot,chain_ur)
                    publish_soro(pub_markers, chain_ur, soro, motor_control)
                    rendering = rendering + 1
                    rate.sleep()
            
            pbar.update()

# %%
if RUN:
    """SCRAP"""
    grasp_dir = 1
    
    chain_ur = update_ur_q(chain_ur, qs)
    motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
    
    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().to(device)
    R_EE_cur = R_plat
    
    grasp_init = -0.8
    rpy_EE_tar_init = np.array([-2.61693358, -0.03022852, -1.58825493], dtype=np.float32)
    p_EE_tar_init = np.array([1.0205003, 0.13381146,0.6415896], dtype=np.float32).reshape(3,1)
    
    ## 나중에 지울것 ##
    # qs_tar = np.array([0, -33, 58, -150, -92, 0]).astype(np.float32) / 180 * PI
    # chain_ur = update_ur_q(chain_ur, qs_tar)
    # motor_control = torch.FloatTensor(np.array([0,0,1000,0]).astype(np.float32)).unsqueeze(0).to(device)

    # p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    # p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    # p_EE_cur = p_EE_cur.detach().cpu().numpy()
    # R_EE_cur = R_plat
    ## 나중에 지울것 ##

    grasp_end = 0.8
    rpy_EE_tar_end = np.array([-2.52964497, -0.02859196, -1.59082341], dtype=np.float32)
    p_EE_tar_end = np.array([1.0784122,0.09696829,0.57288957], dtype=np.float32).reshape(3,1)
    
    traj_n_ = 10
    for (grasp, rpy_EE_tar, p_EE_tar) in \
        tqdm(zip(\
            np.linspace(grasp_init, grasp_end, traj_n),\
            np.linspace(rpy_EE_tar_init, rpy_EE_tar_end, traj_n),
            np.linspace(p_EE_tar_init, p_EE_tar_end, traj_n))):
        
        # print(p_EE_tar)
        
        R_EE_tar = rpy2r_np(rpy_EE_tar)
        
        pbar = tqdm(generator())
        for _ in pbar:
            ## FK UR & SORO
            chain_ur = update_ur_q(chain_ur, qs)
            motor_control = torch.tensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
            p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

            # dp_dm, p_EE = get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control)
            dp_dm, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control)
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

            ## Sphere constraint
            p_plat_EE = (p_EE_cur - chain_ur.joint[8].p).astype(np.float32)
            sph_l = np.linalg.norm(p_plat_EE)
            sph_err = np.array([1/2* (l_tar**2-sph_l**2)], dtype=np.float32).reshape(1,1)
            
            J_sph = p_plat_EE.T @ p_J_soro
            ## Grasp constraint
            grasp = 0.5
            grasp_dir = 0
            
            l_grasp = 0.04 * grasp
            u = chain_ur.joint[8].R[:,grasp_dir].reshape(3,1).astype(np.float32)
            J_grasp = u.T @ p_J_soro
            
            grasp_err = l_grasp - u.T @ p_plat_EE
            
            ## Motor constraint
            margin = 200/scale_rate
            llimit = (motor_control_np < margin).any()
            J_llimit = np.eye(4, dtype=np.float32)[motor_control_np < margin].astype(np.float32)
            llimit_err = (margin-motor_control_np)[motor_control_np < margin].reshape(-1,1)
            
            ulimit = (motor_control_np > 2000/scale_rate-margin).any()
            J_ulimit = np.eye(4, dtype=np.float32)[motor_control_np > 2000/scale_rate-margin]
            ulimit_err = ((2000/scale_rate-margin)-motor_control_np)[motor_control_np > 2000/scale_rate-margin].reshape(-1,1)

            
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
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_grasp]))
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_sph]))
            
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
            
            A = np.vstack(A).astype(np.float32)
            b = np.vstack(b).astype(np.float32)
            
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
                            
            J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl).astype(np.float32)
            dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)

            dq = step_size * dq_raw
            
            dq = dq.flatten()
            qs = qs + dq[:6]
            motor_control_np = motor_control_np+ dq[6:] * scale_rate
            
            if VIZ:
                render_time = 0.1
                frequency = 60
                rate = rospy.Rate(frequency)
                max_rendering = frequency * render_time
                
                rendering = 0
                while not rospy.is_shutdown():
                    if rendering == max_rendering: break

                    publish_robot(pub_robot, chain_ur)
                    publish_soro(pub_markers, chain_ur, soro, motor_control)
                    rendering = rendering + 1
                    rate.sleep()
            pbar.update()

# %%
if RUN:
    """SCRAP"""
    grasp_dir = 1
    
    chain_ur = update_ur_q(chain_ur, qs)
    motor_control = torch.FloatTensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
    
    p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    p_EE_cur = p_EE_cur.detach().to(device)
    R_EE_cur = R_plat
    
    grasp_init = -0.8
    rpy_EE_tar_init = np.array([-2.61693358, -0.03022852, -1.58825493], dtype=np.float32)
    p_EE_tar_init = np.array([1.0205003, 0.13381146,0.6415896], dtype=np.float32).reshape(3,1)
    
    ## 나중에 지울것 ##
    # qs_tar = np.array([0, -33, 58, -150, -92, 0]).astype(np.float32) / 180 * PI
    # chain_ur = update_ur_q(chain_ur, qs_tar)
    # motor_control = torch.FloatTensor(np.array([0,0,1000,0]).astype(np.float32)).unsqueeze(0).to(device)

    # p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)
    # p_EE_cur = forward_model(p_plat, R_plat, soro, motor_control)
    # p_EE_cur = p_EE_cur.detach().cpu().numpy()
    # R_EE_cur = R_plat
    ## 나중에 지울것 ##

    grasp_end = 0.8
    rpy_EE_tar_end = np.array([-2.52964497, -0.02859196, -1.59082341], dtype=np.float32)
    p_EE_tar_end = np.array([1.0784122,0.09696829,0.57288957], dtype=np.float32).reshape(3,1)
    
    traj_n_ = 10
    for (grasp, rpy_EE_tar, p_EE_tar) in \
        tqdm(zip(\
            np.linspace(grasp_init, grasp_end, traj_n),\
            np.linspace(rpy_EE_tar_init, rpy_EE_tar_end, traj_n),
            np.linspace(p_EE_tar_init, p_EE_tar_end, traj_n))):
        
        # print(p_EE_tar)
        
        R_EE_tar = rpy2r_np(rpy_EE_tar)
        
        pbar = tqdm(generator())
        for _ in pbar:
            ## FK UR & SORO
            chain_ur = update_ur_q(chain_ur, qs)
            motor_control = torch.tensor(scale_rate * motor_control_np).unsqueeze(0).to(device)
            p_plat = chain_ur.joint[-1].p.astype(np.float32); R_plat = chain_ur.joint[-1].R.astype(np.float32)

            # dp_dm, p_EE = get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control)
            dp_dm, p_EE = get_hybrid_grad_explicit(p_plat, R_plat, soro, motor_control)
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

            ## Sphere constraint
            p_plat_EE = (p_EE_cur - chain_ur.joint[8].p).astype(np.float32)
            sph_l = np.linalg.norm(p_plat_EE)
            sph_err = np.array([1/2* (l_tar**2-sph_l**2)], dtype=np.float32).reshape(1,1)
            
            J_sph = p_plat_EE.T @ p_J_soro
            ## Grasp constraint
            grasp = 0.5
            grasp_dir = 0
            
            l_grasp = 0.04 * grasp
            u = chain_ur.joint[8].R[:,grasp_dir].reshape(3,1).astype(np.float32)
            J_grasp = u.T @ p_J_soro
            
            grasp_err = l_grasp - u.T @ p_plat_EE
            
            ## Motor constraint
            margin = 200/scale_rate
            llimit = (motor_control_np < margin).any()
            J_llimit = np.eye(4, dtype=np.float32)[motor_control_np < margin].astype(np.float32)
            llimit_err = (margin-motor_control_np)[motor_control_np < margin].reshape(-1,1)
            
            ulimit = (motor_control_np > 2000/scale_rate-margin).any()
            J_ulimit = np.eye(4, dtype=np.float32)[motor_control_np > 2000/scale_rate-margin]
            ulimit_err = ((2000/scale_rate-margin)-motor_control_np)[motor_control_np > 2000/scale_rate-margin].reshape(-1,1)

            
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
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_grasp]))
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_sph]))
            
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
            
            A = np.vstack(A).astype(np.float32)
            b = np.vstack(b).astype(np.float32)
            
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
                            
            J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl).astype(np.float32)
            dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)

            dq = step_size * dq_raw
            
            dq = dq.flatten()
            qs = qs + dq[:6]
            motor_control_np = motor_control_np+ dq[6:] * scale_rate
            
            if VIZ:
                render_time = 0.1
                frequency = 60
                rate = rospy.Rate(frequency)
                max_rendering = frequency * render_time
                
                rendering = 0
                while not rospy.is_shutdown():
                    if rendering == max_rendering: break

                    publish_robot(chain_ur, pub_robot)
                    rendering = rendering + 1
                    rate.sleep()
            pbar.update()
