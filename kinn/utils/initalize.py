import torch
from model.PRIMNET import Tjoint, PRIMNET
import numpy as np
from utils.pyart import rodriguesNoBatch
from utils.pyart import angle_axis_numpy, angle_numpy, pr2t, r2t, rpy2r, t2p
from utils.tools import unnormalize_tensor, normalize_tensor
from configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE

# TPOSE = {
#     "ABAQUS":
#         [[-3.6081e+01, -4.5336e+01, -1.4693e-37],
#          [-4.0160e+01, -4.8704e+01,  2.9617e+02],
#          [-5.9577e+01, -6.7365e+01,  5.9097e+02],
#          [-9.5752e+01, -1.0279e+02,  8.8284e+02],
#          [-1.4831e+02, -1.5461e+02,  1.1705e+03],
#          [-2.1341e+02, -2.1897e+02,  1.4545e+03],
#          [-2.8393e+02, -2.8877e+02,  1.7373e+03]],
    
#     "FINGER":
#          [0,0,120],
    
#     "SOROSIM1D":
#         [[00.0000,  0, 10],
#          [13.3333,  0, 10],
#          [26.6667,  0, 10],
#          [40.0000,  0, 10],
#          [53.3333,  0, 10],
#          [66.6667,  0, 10],
#          [80.0000,  0, 10],
#          [93.3333,  0, 10],
#          [106.6667, 0, 10],
#          [120.0000, 0, 10]],
# }


def INITALZE_EVEN_JOINTS(model:PRIMNET, args:PRIMNET_ARGS_TEMPLATE):
    device = torch.device(args.device)

    if args.OUTPUT_NORMALIZE:
        position_std, position_mean = model.get_buffer("position_std"), model.get_buffer("position_mean")
    
    ## Set p_offset, rpy_offset ###
    Tpose = torch.FloatTensor(args.TPOSE).to(device)
    
    start_pose     = torch.tensor([0,0,0]).reshape(3,1).to(device)
    p_offset       = torch.zeros(3)
    R              = torch.eye(3).to(device)
    joints = model.FK_LAYER.joints

    assert len(args.joint_seqs) // args.marker_num
    
    joint_idxs_chunk = np.split(np.arange(len(args.joint_seqs)), args.marker_num)
    
    for chunk_idx, joint_idxs in enumerate(joint_idxs_chunk):
        end_pose = Tpose[chunk_idx].reshape(3,1)
        
        if args.OUTPUT_NORMALIZE:
            end_pose = normalize_tensor(end_pose, mean = position_mean, std = position_std)

        # Count number of joints execpet Tjoint
        joint_count = 0
        for joint_idx in joint_idxs:
            if not isinstance(joints[joint_idx], Tjoint):
                joint_count += 1
        
        diff_pose = (end_pose - start_pose)/joint_count
        
        # Set p_offset, rpy_offset
        for joint_idx in joint_idxs:
            joint = joints[joint_idx]
            if not isinstance(joint, Tjoint):
                p_offset = R.T @ (diff_pose)
                
                joint.p_offset.data = p_offset
                rpy_offset = joint.rpy_offset.data
                R = R @ rpy2r(rpy_offset)
        
        start_pose = end_pose

    # Asserting Tpose has been set up correctly
    ## 1st    
    joint_se3 = model.FK_LAYER.forward_kinematics(q_values=torch.zeros(16,len(args.joint_seqs)))
    t_position = model.t2p(joint_se3, False)

    aux_joints = len(args.joint_seqs) // args.marker_num
    # assertion of proper initalization
    for i in range(len(args.joint_seqs)):
        if (i+1)%aux_joints == 0:
            prim_idx = i//aux_joints            
            t_position_EE = t_position[0, i].flatten()
            t_position_GT = Tpose[prim_idx]
            if args.OUTPUT_NORMALIZE:
                t_position_GT = normalize_tensor(t_position_GT.reshape(3,1), mean = position_mean, std = position_std).flatten()

            assert t_position_EE.shape == t_position_GT.shape
            
            similarity = torch.cosine_similarity(t_position_EE, t_position_GT, dim=0)
            # assert similarity > 0.95

            from torch.linalg import norm as torch_norm
            
            similarity = (torch_norm(t_position_EE) - torch_norm(t_position_GT))/torch_norm(t_position_GT)
            # assert similarity < 0.01
            
    
    
    ## 2nd
    q_values = model.forward_q(motor_control = torch.zeros(16,args.motor_dim))
    t_position = model(motor_control = torch.zeros(16,args.motor_dim))
    
    aux_joints = len(args.joint_seqs) // args.marker_num
    # assertion of proper initalization
    for i in range(len(args.joint_seqs)):
        if (i+1)%aux_joints == 0:
            prim_idx = i//aux_joints            
            t_position_EE = t_position[0, i].flatten()
            t_position_GT = Tpose[prim_idx]

            assert t_position_EE.shape == t_position_GT.shape
            
            similarity = torch.cosine_similarity(t_position_EE, t_position_GT, dim=0)
            # assert similarity > 0.9

            from torch.linalg import norm as torch_norm
            
            similarity = (torch_norm(t_position_EE) - torch_norm(t_position_GT))/torch_norm(t_position_GT)
            # assert similarity < 0.1

    return model



'''
def SET_TPOSE(model, args):
    Tpose = np.array(TPOSE[args.DATASET])
    
    if args.DATASET in ["ABAQUS"]:
        dt = np.array([0,0,1])
    elif args.DATASET in ["FINGER"]:
        dt = np.array([0,0.05,0.95])
        # dt = np.array([0,0.01,0.99])
        
        
    Tpose_Dict = dict(NORMAL_VECTOR=[],RADIUS=[],THETA=[],PHIS=[])
    
    
    for point_idx in range(len(Tpose)-1):
        P1 = Tpose[point_idx]
        P2 = Tpose[point_idx+1]

        δP = P2-P1
        dη = np.cross(dt,δP)
        dη = dη/np.linalg.norm(dη)

        dM = np.cross(δP,dη)
        dM = dM/np.linalg.norm(dM)
        
        scale_t = np.dot(δP,dt)/2/np.dot(dM,dt)
        assert scale_t > 0
        # print(scale_t)
        
        
        C = (P1+P2)/2 - scale_t * dM

        CP1 = P1-C; CP2 = P2-C
        assert np.linalg.norm(CP1)-np.linalg.norm(CP2)<1e-5
        radius = np.linalg.norm(CP1)
        
        θ = angle_numpy(CP1,CP2)
        assert θ>0
        
        diff = CP2-rodriguesNoBatch(CP1,dη,θ)
        assert np.linalg.norm(diff)<1e-5
        
        Tpose_Dict['NORMAL_VECTOR'].append(list(dη))
        Tpose_Dict['RADIUS'].append(float(radius))
        Tpose_Dict['THETA'].append(float(θ))
    
        if point_idx==0:            
            R_START = torch.tensor(np.stack([-dη/np.linalg.norm(dη),-CP1/np.linalg.norm(CP1),dt/np.linalg.norm(dt)]).T).unsqueeze(0)
            p_START = torch.tensor(Tpose[0]).unsqueeze(0)
            T_START = pr2t(p_START, R_START)
            
            Tpose_Dict["START_T"] = (T_START[0]).tolist()
            Tpose_Dict["PHIS"].append(float(0))
            dη_prev = dη
        else:
            phi = angle_axis_numpy(axis = dt, before=dη_prev, after = dη)
            Tpose_Dict["PHIS"].append(float(phi))
    
        # Next iter
        dt = rodriguesNoBatch(dt,dη,θ)
    
    
    model.set_Tpose_Dict(Tpose_Dict)
    
    Phi = torch.zeros(16,6)
    Theta = torch.zeros(16,6)
    Radius = torch.zeros(16,6)
    primary_joint_position = model.forward_PCC(Phi,Theta,Radius)
    
    if model.Tpose_Dict_flag:
        print("FK result: \n", primary_joint_position[0])
        print("IK target: \n", Tpose)
        print("IK Error : \n", primary_joint_position[0] - Tpose)
'''