from dataclasses import dataclass
from torch import tensor, Tensor
from torch import nn


@dataclass
class ARGS():
    # MODEL
    MODEL:str = "PRIMNET"
    EVEN_JOINTS:bool = True
    
    # LOG
    WANDB:bool = True
    pname:str = "PRIMNET_v2.1"                      # WANDB project Name
    runname:str = "ABAQUS64"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "ABAQUS"                         # DATASET
    TPOSE:tuple = (
        (-3.6081e+01, -4.5336e+01, -1.4693e-37),
        (-4.0160e+01, -4.8704e+01,  2.9617e+02),
        (-5.9577e+01, -6.7365e+01,  5.9097e+02),
        (-9.5752e+01, -1.0279e+02,  8.8284e+02),
        (-1.4831e+02, -1.5461e+02,  1.1705e+03),
        (-2.1341e+02, -2.1897e+02,  1.4545e+03),
        (-2.8393e+02, -2.8877e+02,  1.7373e+03)
        )
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                      # Path to load weight

    # LOGS
    SAVE_PERIOD:int = 1
    TEST_PERIOD:int = 1
    
    # INITALIZATION
    # INIT_FC_LAYER = nn.init.xavier_normal_
    # JOINT_INITALIZE = nn.init.uniform_
    EVEN_JOINT:bool = True
    p_offset_std:float = 0.1
    rpy_offset_std:float = 0.01
    axis_std:float = 0.1
    
    # NORMALIZATION
    OUTPUT_NORMALIZE:bool = False
    
    # SEED
    seed:int = 0

    # DIMENSION
    hdim:tuple = (64,64)
    motor_embed_dim:int = 64
    
    # TRAINING
    lr:float = 0.0015
    lrd:float = 0.95
    wd:float = 0.0
    w_vec:float = 0.0
    epochs:int = 500
    focus_ratio:float = 0.0
    data_ratio:float = 1.0
    n_workers:int = 2
    batch_size:int = 128
    
    # ARCHITECTURE
    actv =  nn.ReLU # nn.ReLU, nn.LeakyReLU, nn.Mish
    marker_num:int = 7
    motor_dim:int = 3
    joint_seqs:tuple = (
        "F",
        "R",
        "P",
        "R",
        "R",
        "P",
        "R"
    ) * marker_num
    
    # DEVICE
    device = "cpu"
    

