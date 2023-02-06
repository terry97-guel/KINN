from configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE
from model.PRIMNET import PRIMNET
from model.FC_PRIMNET import FC_PRIMNET
from utils.tools import swap_dim_0_1, cast_dict_numpy
from utils.dataloader import Sampler
from torch.nn import functional as F
import torch

# p_loss_fn = F.mse_loss

def p_loss_fn(x,y):
    loss =  F.smooth_l1_loss(x,y, reduction='none')
    loss = torch.mean(loss, dim=list(range(1,loss.ndim)))
    return loss

def update_primnet(model:PRIMNET, batch, args:PRIMNET_ARGS_TEMPLATE, TRAIN = True, sampler:Sampler = None):
    target_position = batch["position"]
    
    motor_control = batch["motor_control"]
    
    joint_position = model.forward(motor_control, OUTPUT_NORMALIZE = args.OUTPUT_NORMALIZE)
    
    aux_joints = len(args.joint_seqs) // args.marker_num
    # Position loss & Vector loss
    position_loss = 0.0
    vector_loss = 0.0
    for i in range(len(args.joint_seqs)):
        joint_position_ = joint_position[:,i]
        
        if (i+1)%aux_joints == 0:
            target_position_ = target_position[:, i//aux_joints, :]
            
            assert joint_position_.shape == target_position_.shape
            position_loss = position_loss + p_loss_fn(joint_position_, target_position_)
        
        if i< len(args.joint_seqs)-1:
            next_joint_position_ = joint_position[:,i+1]
            
            assert joint_position_.shape == next_joint_position_.shape
            
            vector_loss = vector_loss + torch.cosine_similarity(joint_position_,next_joint_position_).squeeze(-1)

    assert position_loss.shape == vector_loss.shape
    total_loss = position_loss + vector_loss * args.w_vec
    if TRAIN:
        # sampler.update(position_loss)
        assert total_loss.ndim == 1
        total_loss = torch.mean(total_loss)
        total_loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        
    return cast_dict_numpy(dict(
        total_loss = torch.mean(total_loss),        
        vector_loss = torch.mean(vector_loss),
        position_loss = torch.mean(position_loss)
        ))
    

def update_fc_primnet(model:FC_PRIMNET,batch, args:FC_PRIMNET_ARGS_TEMPLATE, TRAIN=True, sampler:Sampler=None):
    target_position = batch["position"]
    motor_control = batch["motor_control"]
    
    joint_position = model.forward(motor_control, OUTPUT_NORMALIZE = args.OUTPUT_NORMALIZE)
    
    position_loss = 0,0
    
    position_loss = p_loss_fn(joint_position, target_position)
    
    if TRAIN:
        # sampler.update(position_loss)
        position_loss = torch.mean(position_loss)
        position_loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        
    return cast_dict_numpy(dict(
        position_loss = torch.mean(position_loss)
    ))

def update_pcc_primnet(model,batch):
    pass

