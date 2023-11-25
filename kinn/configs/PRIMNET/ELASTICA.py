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
    pname:str = "PRIMNET_v2.2"                      # WANDB project Name
    runname:str = "ELASTICA"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "ELASTICA"                         # DATASET
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
    seed:int = 0

    # DIMENSION
    hdim:tuple = (16,16)
    motor_embed_dim:int = 4
    
    # TRAINING
    lr:float = 0.0015
    lrd:float = 0.95
    wd:float = 0.0
    w_vec:float = 1e-1
    epochs:int = 2000
    focus_ratio:float = 0.2
    data_ratio:float = 1.0
    n_workers:int = 2
    batch_size:int = 64
    
    # ARCHITECTURE
    actv =  nn.Mish # nn.ReLU, nn.LeakyReLU, nn.Mish
    joint_seqs:tuple = (
        "T",
        "R",
        "R",
        "R",
        "R",
        "T",
        "P"
    )
    marker_num:int = 1
    motor_dim:int = 2
    
    # DEVICE
    device = "cpu"
    

