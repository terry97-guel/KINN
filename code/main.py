# %%
from pathlib import Path
import sys

try: 
    get_ipython().__class__.__name__
    BASEDIR = Path().absolute()
except: BASEDIR = Path(__file__).parent

sys.path.append(str(BASEDIR))
import torch
from torch import nn
import argparse

import numpy as np
from utils import dataloader
from utils.initalize import INITALZE_EVEN_JOINTS
from utils.update import  update_primnet, update_fc_primnet, update_pcc_primnet


import os
import random
from pathlib import Path
import wandb
import time
import json

from utils.tools import set_seed, set_wandb, print_log_dict, prefix_dict, average_dict
from utils.path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR
from utils.args import read_ARGS
from utils.logger import CSVLogger,ask_and_make_folder
from utils.tools import cast_numpy

from configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE
from model.PRIMNET import PRIMNET
from model.FC_PRIMNET import FC_PRIMNET
from model.PCC_PRIMNET import PCC_PRIMNET
from typing import Union

from utils.dataloader import get_dataset, Sampler
# %%
def main(args:Union[PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE]):
    ### SEED ###
    set_seed(args.seed)

    ### WANDB ###
    if args.WANDB:
        runname = f"[{args.MODEL}]"+args.runname
        set_wandb(pname = args.pname, runname=runname)

    ### Logger ###
    csvlogger = CSVLogger(args.SAVE_RESULT_PATH/"log.csv")
    
    ### DATALOADER ###
    train_dataset,val_dataset,test_dataset,ext_dataset  = get_dataset(f"dataset/{args.DATASET}.json", args.data_ratio)
    
    print("train_dataset:",len(train_dataset))
    print("val_dataset:",len(val_dataset))
    print("test_dataset:",len(test_dataset))
    print("ext_dataset:",len(ext_dataset))
    
    (motor_std, motor_mean), (pos_std, pos_mean) = train_dataset.get_std_mean()

    ### Declare Instance ###
    # Model
    if args.MODEL == "PRIMNET":
        model = PRIMNET(args=args).to(args.device)
        model.register_motor_std_mean(motor_std, motor_mean)
        model.register_position_std_mean(pos_std, pos_mean)
        
        if args.EVEN_JOINT:
            model = INITALZE_EVEN_JOINTS(model, args)
            
    elif args.MODEL == "FC_PRIMNET":
        model = FC_PRIMNET(args=args).to(args.device)
        model.register_motor_std_mean(motor_std, motor_mean)
        model.register_position_std_mean(pos_std, pos_mean)
        
        
    elif args.MODEL == "PCC_PRIMNET":
        model = PCC_PRIMNET(args=args).to(args.device)
        model.register_motor_std_mean(motor_std, motor_mean)
        model.register_position_std_mean(pos_std, pos_mean)
    else:
        raise LookupError(f"model should be one of ['PRIMNET', 'FC_PRIMNET', 'PCC_PRIMNET'] \n, Found {args.MODEL}")
    
    # Sampler
    train_sampler = Sampler(args, train_dataset)
    val_sampler   = Sampler(args, val_dataset)
    test_sampler  = Sampler(args, test_dataset)
    ext_sampler   = Sampler(args, ext_dataset)
    
    # Load Weight
    if args.LOAD_WEIGHTPATH is None:
        print("No Weights Found!! Starting From Scratch")
    elif Path.is_file(Path(args.LOAD_WEIGHTPATH)):
        model.load_state_dict(torch.load(args.LOAD_WEIGHTPATH),strict=True)
    else:
        raise FileNotFoundError()
    
    # Train Loop
    from tqdm import tqdm
    train_dict_list = []
    for epoch in tqdm(range(args.epochs)):
        for batch in train_sampler:
            if args.MODEL == "PRIMNET":
                train_log_dict = update_primnet(model, batch, args, TRAIN = True, sampler = train_sampler)
            elif args.MODEL == "FC_PRIMNET":
                train_log_dict = update_fc_primnet(model, batch, args, TRAIN = True, sampler = train_sampler)
            elif args.MODEL == "PCC_PRIMNET":
                train_log_dict = update_pcc_primnet(model, batch, args, TRAIN = True, sampler = train_sampler)
            else: raise LookupError(f"model should be one of ['PRIMNET', 'FC_PRIMNET', 'PCC_PRIMNET'] \n, Found {args.MODEL}")
            train_log_dict = prefix_dict(train_log_dict,"train")

            train_dict_list.append(train_log_dict)
        
        train_log_dict = average_dict(train_dict_list)
        if args.WANDB: wandb.log(train_log_dict, step=epoch+1)
        
        # Evaluation - val_dataset
        batch = val_sampler.sample_all()
        if args.MODEL == "PRIMNET":
            val_log_dict = update_primnet(model, batch, args, TRAIN = False)
        elif args.MODEL == "FC_PRIMNET":
            val_log_dict = update_fc_primnet(model, batch, args, TRAIN = False)
        elif args.MODEL == "PCC_PRIMNET":
            val_log_dict = update_pcc_primnet(model, batch, args, TRAIN = False)
        else: raise LookupError(f"model should be one of ['PRIMNET', 'FC_PRIMNET', 'PCC_PRIMNET'] \n, Found {args.MODEL}")
        val_log_dict = prefix_dict(val_log_dict,"val")
        
            
        if args.WANDB: wandb.log(val_log_dict, step=epoch+1)
        if RUNMODE is not RUN:
            print_log_dict(val_log_dict)
        
        # Evaluation - test_dataset
        batch = test_sampler.sample_all()
        if args.MODEL == "PRIMNET":
            test_log_dict = update_primnet(model, batch, args, TRAIN = False)
        elif args.MODEL == "FC_PRIMNET":
            test_log_dict = update_fc_primnet(model, batch, args, TRAIN = False)
        elif args.MODEL == "PCC_PRIMNET":
            test_log_dict = update_pcc_primnet(model, batch, args, TRAIN = False)
        else: raise LookupError(f"model should be one of ['PRIMNET', 'FC_PRIMNET', 'PCC_PRIMNET'] \n, Found {args.MODEL}")
        test_log_dict = prefix_dict(test_log_dict,"test")

        if args.WANDB: wandb.log(test_log_dict, step=epoch+1)

        # Evaluation - ext_dataset
        batch = ext_sampler.sample_all()
        if args.MODEL == "PRIMNET":
            ext_log_dict = update_primnet(model, batch, args, TRAIN = False)
        elif args.MODEL == "FC_PRIMNET":
            ext_log_dict = update_fc_primnet(model, batch, args, TRAIN = False)
        elif args.MODEL == "PCC_PRIMNET":
            ext_log_dict = update_pcc_primnet(model, batch, args, TRAIN = False)
        else: raise LookupError(f"model should be one of ['PRIMNET', 'FC_PRIMNET', 'PCC_PRIMNET'] \n, Found {args.MODEL}")
        ext_log_dict = prefix_dict(ext_log_dict,"ext")
        if RUNMODE is not RUN:
            print_log_dict(ext_log_dict)
        
        if args.WANDB: wandb.log(ext_log_dict, step=epoch+1)
        csvlogger.log(train_log_dict, val_log_dict, test_log_dict, ext_log_dict)
    
        model.save_weights(epoch)

if __name__ == "__main__":
    BASEDIR, RUNMODE = get_BASERDIR(__file__)

    parser = argparse.ArgumentParser(description= 'parse for DLPG')
    parser.add_argument("--configs", default="PRIMNET/FINGER.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET] # [FINGER, ABAQUS_32]
    args= parser.parse_args()

    ARGS = read_ARGS((BASEDIR/'configs'/args.configs).absolute())
    
    ### Save ARGS ###
    import shutil
    SAVEPATH = Path(BASEDIR).parent/"results"/ARGS.MODEL/ARGS.runname
    
    ask_and_make_folder(SAVEPATH, ask = (RUNMODE is RUN))
    shutil.copyfile((BASEDIR/'configs'/args.configs).absolute(), SAVEPATH/"args.py")
    

    ### Set Path ###
    ARGS.SAVE_RESULT_PATH = Path(BASEDIR).parent/"results"/ARGS.MODEL/ARGS.runname
    ARGS.SAVE_WEIGHT_PATH = Path(BASEDIR).parent/"results"/ARGS.MODEL/ARGS.runname/"weights"

    Path.mkdir(ARGS.SAVE_WEIGHT_PATH,exist_ok=True, parents=True)
    Path.mkdir(ARGS.SAVE_RESULT_PATH,exist_ok=True, parents=True)
    
    # if RUNMODE is not RUN:
    #     ARGS.WANDB = False
    
    main(ARGS)


