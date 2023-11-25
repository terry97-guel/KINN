import numpy as np
from soro.hybrid_robot import rpy2r, r2rpy
def make_markers(name, type, pos, rot, size, color): 
    return {"name":name, "type":type, "info":pos+rot+size, "color":color}
    
def get_joint_cylinder(robot, idx, radius = 0.15, color=[0.5,0,1,1]):

    pos_fr = robot.chain.joint[idx].p.reshape(-1)
    pos_to = robot.chain.joint[idx+1].p.reshape(-1)
    name = f"link_{idx}"
    return get_cylinder_from_axis(pos_fr, pos_to, radius, color, name)

def get_cylinder_from_axis(pos_fr, pos_to, radius, color, name = None):
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

    R_link = rpy2r(np.array([r,p,0]))

    rpy = r2rpy(R_link)

    length = np.linalg.norm(pos_to - pos_fr)
    pos = pos_fr + pos_del/2

    link  = make_markers(name=name, 
                            type="cylinder",  
                            pos=pos.tolist(),
                            rot=rpy.tolist(), 
                            size=[radius, radius, length], 
                            color=color)

    return link


def make_sphere(name, pos, radius, color):
    sphere  = make_markers(name=name, 
                        type="sphere",  
                        pos=list(pos),
                        rot=[0,0,.0], 
                        size=[float(radius)] * 3, 
                        color=color)
    return sphere


from soro.hybrid_robot import RobotClass, Global_pAux, get_platform_pr_tensor
from soro.tools import cast_to_numpy
import rospy
import torch

def rviz_show_soro(
    robot:RobotClass, model, motor_control, target_position,
    render_time = 10, target_rpy = None,
    p_offsets = torch.zeros(4,3,1)):

    assert len(p_offsets.shape) == 3
    assert len(p_offsets) in [1,4]

    obs_info_lst = []
    
    black = [0.1,0.1,0.1,0.3]
    white = [0.8,0.8,0.8,1]
    red   = [0.5,0.0,0.0,0.8]
    green  = [0.0,1.0,0.0,0.8]
    blue  = [0,0,1, 0.8]

    # Visualize Platform
    obs_info_lst.append(get_joint_cylinder(robot=robot, idx=7,  radius = 0.08  , color = black))

    PI = np.pi

    thetas = np.linspace(0, 2*PI, 4+1)[:-1]
    for theta, p_offset in zip(thetas, p_offsets):
        p_plat, R_plat = get_platform_pr_tensor(robot)

        R_local = torch.FloatTensor(rpy2r([0,0,theta]))
        pos_fr = p_plat + R_plat @ p_offset
        auxs = Global_pAux(robot, model, motor_control, p_offset, R_local)

        # Visualize Soft robot
        for idx, aux in enumerate(auxs):
            pos_fr, pos_to = cast_to_numpy(pos_fr.squeeze(-1)), cast_to_numpy(aux.squeeze(-1))

            radius = 0.02
            soro_mesh = get_cylinder_from_axis(pos_fr, pos_to, radius, white, name=f"joint_{idx+1}")    

            obs_info_lst.append(soro_mesh)
            pos_fr = aux


    # Visualize End Effector
    EE_pos = cast_to_numpy(auxs[-1,:,0])
    EE = make_sphere("EE", EE_pos, 0.01, blue)
    obs_info_lst.append(EE)

    # Visualize End Effector rpy
    if target_rpy is not None:
        idx = 8
        R_ = robot.chain.joint[idx].R
        length = 0.05
        radius = 0.01
        pos_fr = robot.chain.joint[idx].p.flatten()
        
        colors = [red, green, blue]
        for idx, d_vec in enumerate(R_.T):
            pos_to = pos_fr + d_vec * length
            
            EE_rpy = get_cylinder_from_axis(pos_fr, pos_to, radius, colors[idx], name=f"EE_rpy_{idx+1}")    
            obs_info_lst.append(EE_rpy)
        

    # Visualize IK Target
    IK_TAR = make_sphere("IK_TAR", target_position[0], 0.03, red)
    obs_info_lst.append(IK_TAR)

    # Visualize IK rpy Target
    if target_rpy is not None:
        target_rpy = cast_to_numpy(target_rpy)
        R_ = rpy2r(target_rpy)
        length = 0.08
        pos_fr = target_position[0]
        
        colors = [red, green, blue]
        for idx, d_vec in enumerate(R_):
            pos_to = target_position[0] + d_vec * length
            
            
            IK_TAR_ROT = get_cylinder_from_axis(pos_fr, pos_to, radius, colors[idx], name=f"IK_TAR_ROT_{idx+1}")    
            obs_info_lst.append(IK_TAR_ROT)

    frequency = 60
    rate = rospy.Rate(frequency)
    max_rendering = frequency * render_time

    rendering=0
    while not rospy.is_shutdown():
        if rendering == max_rendering:
            break
        
        robot.publish_robot()
        robot.publish_markers(obs_info_lst)
        rendering +=1
        rate.sleep()