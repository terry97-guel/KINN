from dataclasses import dataclass
from torch import tensor, Tensor
from torch import nn


@dataclass
class ARGS():
    # MODEL
    MODEL:str = "FC_PRIMNET"
    
    # LOG
    WANDB:bool = True
    pname:str = "PRIMNET_v2.2"                      # WANDB project Name
    runname:str = "ABAQUS32"                         # WANDB runname. If unspecified, set to datetime.
    
    # DATASET
    DATASET: str = "ABAQUS"                         # DATASET
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                      # Path to load weight

    # LOGS
    SAVE_PERIOD:int = 1
    TEST_PERIOD:int = 1
    
    # INITALIZATION
    INIT_FC_LAYER = nn.init.xavier_normal_
    
    # NORMALIZATION
    OUTPUT_NORMALIZE:bool = False
    
    # SEED
    seed:int = 0

    # DIMENSION
    hdim:tuple = (32,32)
    motor_embed_dim:int = 64
    
    # TRAINING
    lr:float = 0.0015
    lrd:float = 0.95
    wd:float = 0.0
    epochs:int = 500
    focus_ratio:float = 0.0
    data_ratio:float = 1.0
    n_workers:int = 2
    batch_size:int = 128
    
    # ARCHITECTURE
    actv =  nn.Mish # nn.ReLU, nn.LeakyReLU, nn.Mish
    marker_num:int = 7
    motor_dim:int = 3

    # DEVICE
    device = "cpu"
    

