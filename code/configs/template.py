from dataclasses import dataclass
from torch import tensor, Tensor
from torch import nn


@dataclass
class PRIMNET_ARGS_TEMPLATE():
    # MODEL
    MODEL:str = "PRIMNET"
    EVEN_JOINTS:bool = True
    
    # LOG
    WANDB:bool = True
    pname:str = "PRIMNET_v2.1"                      # WANDB project Name
    runname:str = "PRIMNET"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "FINGER"                         # DATASET
    TPOSE:tuple = ((0,0,120))
    
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
    OUTPUT_NORMALIZE:bool = True
    
    # SEED
    seed:int = 0

    # DIMENSION
    hdim:tuple = (128,128,128)
    motor_embed_dim:int = 32
    pdim:int = 1
    
    # TRAINING
    lr:float = 0.0015
    lrd:float = 0.95
    wd:float = 0.0
    w_vec:float = 100.0
    epochs:int = 150
    focus_ratio:float = 0.5
    data_ratio:float = 1.0
    n_workers:int = 2
    batch_size:int = 64
    
    # ARCHITECTURE
    actv =  nn.Mish # nn.ReLU, nn.LeakyReLU, nn.Mish
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
    
    
    
    
@dataclass
class FC_PRIMNET_ARGS_TEMPLATE():
    # MODEL
    MODEL:str = "FC_PRIMNET"
    EVEN_JOINTS:bool = True
    
    # LOG
    WANDB:bool = True
    pname:str = "PRIMNET_v2.1"                      # WANDB project Name
    runname:str = "PRIMNET"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "FINGER"                         # DATASET
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                      # Path to load weight

    # LOGS
    SAVE_PERIOD:int = 1
    TEST_PERIOD:int = 1
    
    # INITALIZATION
    INIT_FC_LAYER = nn.init.xavier_normal_
    
    # NORMALIZATION
    INPUT_NORMALIZE:bool = True
    OUTPUT_NORMALIZE:bool = True
    
    # SEED
    seed:int = 0
    
    # TRAINING
    hdim:tuple = (128,128,128)
    lr:float = 0.0015
    lrd:float = 0.95
    w_vec:float = 100.0
    epochs:int = 150
    focus_ratio:float = 0.5
    data_ratio:float = 1.0
    n_workers:int = 2
    
    # ARCHITECTURE
    actv =  nn.Mish # nn.ReLU, nn.LeakyReLU, nn.Mish
    adim:int = 32
   
    
@dataclass
class PCC_PRIMNET_ARGS_TEMPLATE():
    # MODEL
    MODEL:str = "FC_PRIMNET"
    EVEN_JOINTS:bool = True
    
    # LOG
    WANDB:bool = True
    pname:str = "PRIMNET_v2.1"                      # WANDB project Name
    runname:str = "PRIMNET"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "FINGER"                         # DATASET
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                      # Path to load weight

    # LOGS
    SAVE_PERIOD:int = 1
    TEST_PERIOD:int = 1
    
    # INITALIZATION
    INIT_FC_LAYER = nn.init.xavier_normal_
    
    # NORMALIZATION
    INPUT_NORMALIZE:bool = True
    OUTPUT_NORMALIZE:bool = True
    
    # SEED
    seed:int = 0
    
    # TRAINING
    hdim:tuple = (128,128,128)
    lr:float = 0.0015
    lrd:float = 0.95
    w_vec:float = 100.0
    epochs:int = 150
    focus_ratio:float = 0.5
    data_ratio:float = 1.0
    n_workers:int = 2
    
    # ARCHITECTURE
    actv =  nn.Mish # nn.ReLU, nn.LeakyReLU, nn.Mish
    adim:int = 32