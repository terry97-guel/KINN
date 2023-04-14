import numpy as np

from kinematics.class_structure import CHAIN
from kinematics.structure_utils import  \
    get_p_chain, get_R_chain, get_rpy_from_R_mat, get_mesh_chain, get_scale, get_link_color, get_viz_ingredients

ROS = False
if ROS:
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from kinematics.rviz import publish_viz_robot, publish_viz_markers

import torch
from torch import Tensor

class RobotClass:
    def __init__(self, file_name = "code/urdf/soro/ur5e_onrobot.urdf", base_offset=[0,0,0], number = 1):
        if ROS:
            rospy.init_node("Run_Robot")
            self.pub_robot      = rospy.Publisher(f'viz_robot_{number}', MarkerArray, queue_size=10)
            self.pub_sphere     = rospy.Publisher(f'viz_sphere_{number}', MarkerArray, queue_size=10)
        self.chain          = CHAIN(file_name=file_name, base_offset=base_offset)
        self.chain.add_joi_to_robot()
        self.chain.add_link_to_robot()
        self.ctrl_joint_num = 6
    
    def publish_robot(self):
        if ROS:
            # update_q_chain(self.chain.joint, q_list, self.ctrl_joint_num)
            self.chain.fk_chain(1)
            p_list     = get_p_chain(self.chain.joint)
            R_list     = get_R_chain(self.chain.joint)
            rpy_list   = get_rpy_from_R_mat(R_list)
            mesh_list  = get_mesh_chain(self.chain.link)
            scale_list = get_scale(self.chain.link)
            color_list = get_link_color(self.chain.link)
            viz_links  =  get_viz_ingredients(p_list, rpy_list, mesh_list, scale_list, color_list)
            viz_trg_robot = publish_viz_robot(viz_links)
            self.pub_robot.publish(viz_trg_robot)
        else:
            raise 
        
    def publish_markers(self, obj):
        if ROS:
            viz_obj = publish_viz_markers(obj)
            self.pub_sphere.publish(viz_obj)
        else:
            pass    

# %%
def set_qs(robot:RobotClass, qs):
    assert len(qs) == 6

    for idx in range(6):
        robot.chain.joint[1+idx].q = float(qs[idx])
    
    robot.chain.fk_chain(1)

    return robot

def get_qs(robot:RobotClass):
    qs = []
    for joint in robot.chain.joint[1:1+6]:
        qs.append(joint.q)
    return np.array(qs)

def Local_pEE(model, motor_control):
    assert len(motor_control) == 1
    primary_joint_position, auxilary_joint_position =  model(motor_control)
    return primary_joint_position[0,-1]/1000

def Local_pAux(model, motor_control):
    assert len(motor_control) ==1
    primary_joint_position, auxilary_joint_position = model(motor_control)
    return auxilary_joint_position[0,7:]/1000


def get_platform_pr_tensor(robot, device = 'cpu'):
    p_plat = robot.chain.joint[8].p
    R_plat = robot.chain.joint[8].R

    return torch.FloatTensor(p_plat).to(device), torch.FloatTensor(R_plat).to(device)



def Global_pEE(robot, model, motor_control, p_offset = torch.zeros(3,1)):
    p_plat, R_plat = get_platform_pr_tensor(robot)

    return p_plat + R_plat @ (Local_pEE(model, motor_control).unsqueeze(-1) + p_offset)



def Global_pAux(robot, model, motor_control, p_offset = torch.zeros(3,1), R_offset = torch.eye(3)):
    assert len(p_offset.shape) == 2
    device = motor_control.device
    p_plat, R_plat = get_platform_pr_tensor(robot)
    aux_p = Local_pAux(model, motor_control).unsqueeze(-1)
    aux_p = R_offset @ aux_p + p_offset
    
    repeat_size = aux_p.shape[0]
    R_plat = torch.tile(R_plat, dims=[repeat_size,1,1]).to(device)

    return p_plat + torch.bmm(R_plat, aux_p)



def fk_UR_soro(robot, qs, model, motor_control):
    robot = set_qs(robot, qs)
    return robot, Global_pEE(robot, model, motor_control), Global_pAux(robot, model, motor_control)

# %%
from numpy import ndarray

def cast_to_numpy(data):
    if isinstance(data,Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, ndarray):
        return data
    if isinstance(data,list):
        return np.array(data)
    else:
        raise TypeError(f"Expected instance of ['Tensor', 'ndarry', 'list'] but recived {type(data)}")



# %%
def skew(vec):
    assert vec.shape==(3,1)
    x = vec[0,0]
    y = vec[1,0]
    z = vec[2,0]
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def get_p_J_UR(robot, model, motor_control):
    x = Global_pEE(robot, model, motor_control)
    x = cast_to_numpy(x)
    J_UR = []
    for joint in robot.chain.joint[1:1+6]:
        assert joint.type == 'revolute'

        J_ = skew(joint.a) @ (x - joint.p)
        J_UR.append(J_)

    J_UR = np.hstack(J_UR)
    return J_UR

def get_rpy_J_UR(robot):
    J_UR = []
    for joint in robot.chain.joint[1:1+6]:
        assert joint.type == 'revolute'

        J_ = joint.R @ joint.a
        J_UR.append(J_)

    J_UR = np.hstack(J_UR)
    return J_UR

from functools import partial
from torch.autograd.functional import jacobian

def get_J_soro(robot, model, motor_control):
    J_soro =  jacobian(partial(Global_pEE, robot, model), motor_control)[:,0,0,:]
    return cast_to_numpy(J_soro)

def get_J(robot, model, motor_control, SORO=False):
    p_J_UR = get_p_J_UR(robot, model, motor_control)
    rpy_J_UR = get_rpy_J_UR(robot)

    if SORO:
        J_soro = get_J_soro(robot,model, motor_control)
    else:
        J_soro = None
    return p_J_UR, rpy_J_UR, J_soro


def get_p_ik_err(x_EE, target_position):
    target_position = cast_to_numpy(target_position)
    assert target_position.shape == (1,3)
    x_EE = cast_to_numpy(x_EE)

    target_position = cast_to_numpy(target_position).reshape(3,1)

    return target_position - x_EE


def r2w(R):
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


def rpy2r(rpy):
    r_rad = rpy[0]
    p_rad = rpy[1]
    y_rad = rpy[2]

    cos_r = np.cos(r_rad); sin_r = np.sin(r_rad)
    cos_p = np.cos(p_rad); sin_p = np.sin(p_rad)
    cos_y = np.cos(y_rad); sin_y = np.sin(y_rad)

    R = np.array([
        [cos_y*cos_p,   -sin_y*cos_r+cos_y*sin_p*sin_r, sin_y*sin_r+cos_y*sin_p*cos_r],
        [sin_y*cos_p,    cos_y*cos_r+sin_y*sin_p*sin_r,  -cos_y*sin_r+sin_y*sin_p*cos_r],
        [-sin_p,         cos_p*sin_r,                    cos_p*cos_r]
    ])

    return R
    
def r2rpy(R):
    assert R.shape == (3,3)
    r = np.arctan2(R[2,1], R[2,2])
    p = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    y = np.arctan2(R[1,0], R[0,0])

    rpy = np.array([r,p,y]).reshape(-1)
    return rpy


def check_SO3(R):
    test1 = R.shape == (3,3)

    test2 = np.sum(np.abs(R.T @ R)) < 3 + 1e-2
    test3 =  3-1e-2 < np.sum(np.abs(R.T @ R)) 
    return test1 and test2 and test3

def get_rpy_ik_err(R_EE, target_rpy):
    target_rpy = cast_to_numpy(target_rpy)
    assert target_rpy.shape == (3,)
    assert check_SO3(R_EE)

    target_rpy = cast_to_numpy(target_rpy)
    Rerr = R_EE.T @ rpy2r(target_rpy)

    werr = R_EE @ r2w(Rerr)
    return werr


def get_ik_ingredients(robot, model, motor_control, target_position, target_rpy, SORO=True):
    p_J_UR, rpy_J_UR, J_soro = get_J(robot,model,motor_control, SORO)
    x_EE = Global_pEE(robot,model, motor_control)
    p_ik_err = get_p_ik_err(x_EE, target_position)

    R_EE = robot.chain.joint[8].R
    rpy_ik_err = get_rpy_ik_err(R_EE, target_rpy)
    return [p_J_UR, rpy_J_UR, J_soro], [p_ik_err, rpy_ik_err]


def column_size(matrix):
    m = len(matrix[0])
    return m

def damped_ls(J_use, ik_err, step_size=1, lambda_rate = 0.1):
    lambda_min = 1e-6
    lambda_max = 1e-3
    
    ik_err_avg = np.mean(abs(ik_err))
    # Damping Term 
    lambda_ = lambda_rate * ik_err_avg + lambda_min 
    lambda_ = np.maximum(lambda_, lambda_max)

    n_ctrl = column_size(J_use)
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
    J_n_ctrl = np.matmul(J_use.T, J_use) + lambda_* np.eye(n_ctrl, n_ctrl)
    dq_raw = np.matmul(np.linalg.solve(J_n_ctrl, J_use.T), ik_err)

    dq = step_size * dq_raw
    return dq.flatten(), det_J

def move_joints(robot, dq):
    qs = get_qs(robot)
    dq = dq.reshape(qs.shape)
    robot = set_qs(robot,qs+dq)

    return robot
    
def get_platform_pr(robot):
    p_plat = robot.chain.joint[8].p
    R_plat = robot.chain.joint[8].R

    return p_plat, R_plat

# %%
def trim_scale(in_val, th):
    max_abs_val = np.max(abs(in_val.flatten()))
    if max_abs_val > th:
        out_val = in_val/max_abs_val * th 
    else:
        out_val = in_val 

    return out_val

if ROS:
    from soro.soro_rviz import rviz_show_soro
def Hybrid_IK(robot, model, motor_control, target_position, target_rpy, grasp, grasp_dir=0, VERBOSE=True, SHOW=True):
    grasp = max(min(float(grasp),1),-1)
    l_grasp = 0.04 * grasp

    l = 0.15
    K_sph = 1.0
    K_cone = 0.5
    K_grasp = 1e9
    K_soro_ik = 1e6
    K_motor = 1e2
    step = 0

    while True:
        [p_J_UR, rpy_J_UR, J_soro], [p_ik_err, rpy_ik_err] = get_ik_ingredients(robot,model,motor_control,target_position, target_rpy, SORO=True)

        
        A = np.vstack([p_J_UR, rpy_J_UR])
        b = np.vstack([p_ik_err * 1e-1, rpy_ik_err * 1e-1])
        dq_ur,_ =damped_ls(A, b, lambda_rate = 0.01/1e-1)
        dq_ur = trim_scale(dq_ur, th = 40*np.math.pi/180)


        ik_err_size = np.linalg.norm(p_ik_err)
        if step !=0: 
            if ik_err_size < 3e-3 and e_sph < 1e-2 and np.abs(e_grasp)<1e-3 and motor_constaint:
                return robot, motor_control

        x_EE = Global_pEE(robot, model, motor_control)
        x = cast_to_numpy(x_EE) - robot.chain.joint[8].p

        # Sphere constraints
        x_norm  = np.linalg.norm(x)
        e_sph = np.array([1/2 * (l**2 - x_norm**2)])
        e_sph = e_sph.reshape(1,-1)

        A_sph = x.T @ J_soro

        # Grasp constraints
        u = robot.chain.joint[8].R[:,grasp_dir]
        A_grasp = u.T @ J_soro
        e_grasp = l_grasp - u.T@x
        

        A = np.vstack([J_soro     , A_sph, A_grasp])
        b = np.vstack([K_soro_ik * p_ik_err, K_sph*e_sph, K_grasp*e_grasp])
        
        # Motor constraints Lower
        motor_base = 200
        motor_ineq = (motor_control.flatten()<motor_base).any()
        A_motor = np.eye(4)[motor_control.flatten()<motor_base]
        e_motor = cast_to_numpy(motor_base - motor_control[motor_control<motor_base]).reshape(-1,1)
        
        if motor_ineq:
            A = np.vstack([A, A_motor])
            b = np.vstack([b, e_motor * K_motor])
        
        # Motor constraints Upper
        motor_base = 1800
        motor_ineq = (motor_control.flatten()>motor_base).any()
        A_motor = np.eye(4)[motor_control.flatten()>motor_base]
        e_motor = cast_to_numpy(motor_base - motor_control[motor_control>motor_base]).reshape(-1,1)

        if motor_ineq:
            A = np.vstack([A, A_motor])
            b = np.vstack([b, e_motor * K_motor])

        dq_soro,_ = damped_ls(A, b, lambda_rate = 0.01/K_soro_ik)
        dq_soro = trim_scale(dq_soro, th = 30)

        robot = move_joints(robot, dq_ur)
        motor_control = motor_control + torch.FloatTensor(dq_soro)
        
        step = step + 1

        motor_constaint = (motor_control.flatten()>0).all()

        log_string = "Current step: {}, ik_err_size:{:.2E}, e_sph:{:.2E}, e_grasp:{:.2E}, motor_constaint:{}"
        if VERBOSE:
            print(log_string.format(step, float(ik_err_size), float(e_sph), float(e_grasp), motor_constaint), end='\r')
        if SHOW:
            rviz_show_soro(robot, model, motor_control, target_position, render_time=0.1, target_rpy=target_rpy, p_offsets = torch.zeros((1,3,1)))
