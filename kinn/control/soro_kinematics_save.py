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

from kinn.model.PRIMNET import PRIMNET
import torch
import numpy as np
from kinematics.class_structure import CHAIN

PI = np.pi
VIZ = True

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
from tqdm import tqdm

p_offset = np.array([-0.0059, -0.0067, 0]).astype(np.float32)

if VIZ:
    import rospy
    from rospy import Publisher
    from kinematics.structure_utils import  \
        get_p_chain, get_R_chain, get_rpy_from_R_mat, get_mesh_chain, get_scale, get_link_color, get_viz_ingredients
    from kinematics.rviz import publish_viz_robot, publish_viz_markers
    from visualization_msgs.msg import Marker, MarkerArray
    
    import rosnode
    node_names = rosnode.get_node_names()

    if "VIZ_ROBOT" not in node_names:
        rospy.init_node("VIZ_ROBOT")
        pub_robot     = Publisher(f'viz_robot', MarkerArray, queue_size=10)
        pub_soro      = Publisher(f'viz_soro', MarkerArray, queue_size=10)
        pub_marker    = Publisher(f'viz_marker', MarkerArray, queue_size=10)


    def publish_robot(chain:CHAIN):
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

    def publish_soro(chain_ur:CHAIN, soro:PRIMNET, motor_control, offset=np.array([0,0,0])):
        obs_info_lst = []
        
        black = [0.1,0.1,0.1,0.3]
        white = [0.8,0.8,0.8,1]

        # Visualize Platform
        pos_fr = chain_ur.joint[-2].p.astype(np.float32)
        pos_to = chain_ur.joint[-1].p.astype(np.float32)
        name = "platform"

        obs_info_lst.append(get_cylinder_from_axis(pos_fr, pos_to, 0.08, black, name))

        with torch.no_grad():
            # ps_ = soro(motor_control)[0].detach().cpu().numpy()
            
            p_plat = chain_ur.joint[-1].p.astype(np.float32)
            R_plat = chain_ur.joint[-1].R.astype(np.float32)

            ps_ = soro(motor_control)[0].detach().cpu().numpy() - p_offset.reshape(3,1) + (offset).reshape(3,1)
            zero_point = np.array([0.0043,0.0032,0]).astype(np.float32).reshape(1,3,1)  + (offset).reshape(1,3,1)
            ps_ = np.vstack((zero_point, ps_))
            ps = p_plat + R_plat @ ps_
            
        i = 0
        for pos_fr, pos_to in zip(ps[:-1], ps[1:]):
            i = i+1
            name = f"soro_joint_{i}"
            obs_info_lst.append(get_cylinder_from_axis(pos_fr, pos_to, 0.02, white, name))

        viz_obj = publish_viz_markers(obs_info_lst)
        pub_soro.publish(viz_obj)
    
    def publish_markers(obs_info_lst):
        viz_obj = publish_viz_markers(obs_info_lst)
        pub_marker.publish(viz_obj)

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
from kinn.model.PRIMNET import PRIMNET, Fjoint, Tjoint, Rjoint, Pjoint
from jacobian import jacobian
from functools import partial
from numpy.linalg import norm



# p_offset = np.array([0,0,0]).astype(np.float32)

def forward_model(p_plat, R_plat, soro, motor_control):
    p_plat = torch.FloatTensor(p_plat)
    R_plat = torch.FloatTensor(R_plat)
    p_soro_ = soro(motor_control)[0, -1]
    p_EE = p_plat + R_plat @ (p_soro_- p_offset) 
    return p_EE

def get_hybrid_grad_auto(p_plat, R_plat, soro, motor_control, scale_rate):
    dp_dm, p_EE = jacobian(partial(forward_model, p_plat, R_plat, soro), motor_control)
    return (
        np.array(dp_dm[:,0,0,:].detach().cpu()).astype(np.float32) * scale_rate,
        np.array(p_EE[0].detach().cpu()).astype(np.float32)
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


def get_hybrid_grad_explicit(p_plat, R_plat, soro:PRIMNET, motor_control, scale_rate):
    dq_dm,q_values_tuple = jacobian(partial(forward_q, soro), motor_control)
    dq_dm = dq_dm[:,0,:].detach().numpy(); q_values = q_values_tuple[0].detach().unsqueeze(0)
    
    with torch.no_grad():
        dp_dq, p_soro= kinematic_grad(soro, q_values)
        
    dp_dq = dp_dq.numpy(); 
    
    p_soro = p_soro - p_offset
    p_soro = p_soro.numpy().reshape(3,1)
    p_EE = p_plat + R_plat @ p_soro
    return R_plat@dp_dq@dq_dm* scale_rate, p_EE

def solve_ik_traj(chain_ur, qs, 
                  soro, motor_control_np, 
                  grasp_init, rpy_EE_tar_init, p_EE_tar_init, grasp_end, rpy_EE_tar_end, p_EE_tar_end, grasp_dir,
                  traj_n=10,scale_rate=30, step_size=0.1, VIZ=False):
    l_tar = 0.15
    step_size_original = step_size
    qs_list = []
    motor_list = []
    p_EE_cur_list = []
    for (grasp, rpy_EE_tar, p_EE_tar) in \
        tqdm(zip(\
            np.linspace(grasp_init, grasp_end, traj_n),\
            np.linspace(rpy_EE_tar_init, rpy_EE_tar_end, traj_n),
            np.linspace(p_EE_tar_init, p_EE_tar_end, traj_n))):
        
        # print(p_EE_tar)
        
        R_EE_tar = rpy2r_np(rpy_EE_tar)
        
        pbar = tqdm(generator(), leave=True)
        update_number = 0
        step_size = step_size_original
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

            ## Sphere constraint
            p_plat_EE = (p_EE_cur - chain_ur.joint[8].p).astype(np.float32)
            sph_l = np.linalg.norm(p_plat_EE)
            sph_err = np.array([1/2* (l_tar**2-sph_l**2)], dtype=np.float32).reshape(1,1)
            
            J_sph = p_plat_EE.T @ p_J_soro
            ## Grasp constraint
            l_grasp = 0.01 * grasp
            assert np.abs(np.linalg.norm(grasp_dir) - 1) < 1e-3
            assert grasp_dir.shape == (3,)
            R_ = chain_ur.joint[8].R.astype(np.float32)
            # u = (grasp_dir[0] * R_[:,0] + grasp_dir[1] * R_[:,1] + grasp_dir[2] * R_[:,2] ).reshape(3,1)
            u = grasp_dir.reshape(3,1)
            p_plat_EE_tar = l_grasp * u
            
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
                                    norm(sph_err)
                                ))
                    
            # Break
            if norm(p_ik_err) < 1e-3 and\
                norm(w_ik_err) < 0.01 and\
                    norm(grasp_err) < 5e-3 and\
                        norm(sph_err) < 0.01:
                break
            # Or Solve & Update
            A = []
            b = []
            A.append(np.hstack([J_UR, J_soro]))
            A.append(np.hstack([np.zeros((len(J_grasp),6),dtype=np.float32), J_grasp]))
            A.append(np.hstack([np.zeros((1,6),dtype=np.float32), J_sph]))
            
            b.append(np.vstack([p_ik_err,w_ik_err]))
            b.append(10*grasp_err)
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
        p_EE_cur_list.append(p_EE_cur)
        qs_list.append(qs)
        motor_list.append(motor_control_np)
    return qs_list, motor_list, qs, motor_control_np, p_EE_cur_list


def viz_robot(chain_ur, soro, motor_control, obj_info_list=None, render_time = 0.1, offset=np.array([0,0,0])):
    
    frequency = 60
    rate = rospy.Rate(frequency)
    max_rendering = frequency * render_time
    
    rendering = 0
    while not rospy.is_shutdown():
        if rendering == max_rendering: break

        publish_robot(chain_ur)
        publish_soro(chain_ur, soro, motor_control, offset)
        if obj_info_list is not None:
            publish_markers(obj_info_list)
        rendering = rendering + 1
        rate.sleep()

# %%
