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
    runname:str = "FINGER64"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "FINGER"                         # DATASET
    TPOSE:tuple = (
        (0,0,0.120),
        )
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                      # Path to load weight

    # LOGS
    SAVE_PERIOD:int = 1
    TEST_PERIOD:int = 1
    
    # INITALIZATION
    INIT_FC_LAYER = nn.init.xavier_normal_
    JOINT_INITALIZE = nn.init.uniform_
    EVEN_JOINT:bool = True
    p_offset_std:float = 0.1
    rpy_offset_std:float = 0.01
    axis_std:float = 0.1
    
    # NORMALIZATION
    OUTPUT_NORMALIZE:bool = False
    
    # SEED
    seed:int = 1

    # DIMENSION
    hdim:tuple = (64,64)
    motor_embed_dim:int = 4
    
    # TRAINING
    lr:float = 0.0015
    lrd:float = 0.95
    wd:float = 0.0
    w_vec:float = 1e-3
    epochs:int = 500
    focus_ratio:float = 0.0
    data_ratio:float = 1.0
    n_workers:int = 2
    batch_size:int = 64
    
    # ARCHITECTURE
    actv =  nn.ReLU # nn.ReLU, nn.LeakyReLU, nn.Mish
    joint_seqs:tuple = (
        "F",
        "R",
        "P",
        "R",
        "R",
        "P",
        "R"
    )
    marker_num:int = 1
    motor_dim:int = 4
    
    # DEVICE
    device = "cpu"
    

