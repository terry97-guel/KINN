from configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE
from model.PRIMNET import PRIMNET

def update_primnet(model:PRIMNET, batch, train_sampler, args:PRIMNET_ARGS_TEMPLATE, TRAIN = True):
    target_position = batch["position"]
    
    motor_control = batch["motor_control"]
    
    joints_position_list = model.forward(motor_control, OUTPUT_NORMALIZE = args.OUTPUT_NORMALIZE)
    
    aux_joints = len(args.joint_seqs) // args.pdim
    # Position loss
    position_loss = 0.0
    for i in range(len(joints_position_list)):
        if i%aux_joints == 0:
            
            position_loss += pos_loss

    # Vector loss
    vector_loss = 0.0
    
        
    
    
    def pos_loss():
        return
    pass

def update_fc_primnet(model, batch):
    pass

def update_pcc_primnet(model,batch):
    pass

