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
from kinn.utils import dataloader
from kinn.utils.initalize import INITALZE_EVEN_JOINTS
from kinn.utils.update import  update_primnet, update_fc_primnet, update_pcc_primnet


import os
import random
from pathlib import Path
import wandb
import time
import json
import shutil

from kinn.utils.tools import set_seed, set_wandb, print_log_dict, prefix_dict, average_dict
from kinn.utils.path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR
from kinn.utils.args import read_ARGS
from kinn.utils.logger import CSVLogger,ask_and_make_folder
from kinn.utils.tools import cast_numpy

from kinn.configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE
from kinn.model.PRIMNET import PRIMNET
from kinn.model.FC_PRIMNET import FC_PRIMNET
from kinn.model.PCC_PRIMNET import PCC_PRIMNET
from typing import Union

from kinn.utils.dataloader import get_dataset, Sampler
from dataclasses import dataclass
from collections import OrderedDict
import ray

# %%
@dataclass
class Agent_State_Template():    
    state_dict: OrderedDict = None
    joint_seqs: tuple = None
    score = -np.inf

@ ray.remote(num_cpus=16)
def Ray_agent_main(args:Union[PRIMNET_ARGS_TEMPLATE,FC_PRIMNET_ARGS_TEMPLATE,PCC_PRIMNET_ARGS_TEMPLATE]) -> Agent_State_Template:
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
        if args.OUTPUT_NORMALIZE:
            model.register_position_std_mean(pos_std, pos_mean)
        
        if args.EVEN_JOINT:
            model = INITALZE_EVEN_JOINTS(model, args)
            
    elif args.MODEL == "FC_PRIMNET":
        model = FC_PRIMNET(args=args).to(args.device)
        model.register_motor_std_mean(motor_std, motor_mean)
        if args.OUTPUT_NORMALIZE:
            model.register_position_std_mean(pos_std, pos_mean)
        
        
    elif args.MODEL == "PCC_PRIMNET":
        model = PCC_PRIMNET(args=args).to(args.device)
        model.register_motor_std_mean(motor_std, motor_mean)
        if args.OUTPUT_NORMALIZE:
            model.register_position_std_mean(pos_std, pos_mean)
    else:
        raise LookupError(f"model should be one of ['PRIMNET', 'FC_PRIMNET', 'PCC_PRIMNET'] \n, Found {args.MODEL}")
    
    # Sampler
    train_sampler = Sampler(args, train_dataset)
    val_sampler   = Sampler(args, val_dataset)
    test_sampler  = Sampler(args, test_dataset)
    ext_sampler   = Sampler(args, ext_dataset)
    
    
    agent_state = Agent_State_Template()
    
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
        # print_log_dict(val_log_dict)
        
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
        # print_log_dict(ext_log_dict)
        
        if args.WANDB: wandb.log(ext_log_dict, step=epoch+1)
        csvlogger.log(dict(joint_seqs=args.joint_seqs), train_log_dict, val_log_dict, test_log_dict, ext_log_dict)
    
        # Update agent_state
        score = -val_log_dict["val_position_loss"]
        if agent_state.score < score:
            agent_state.state_dict = model.state_dict()
            agent_state.joint_seqs = model.args.joint_seqs
            agent_state.score = score
    
    return agent_state
        





@dataclass
class KAS_Result_Template():
    best_joint_seqs: tuple = ()
    best_score: float = -np.inf
    archive: tuple[tuple] = ()
    state_dict: OrderedDict = None
    last_joint_seqs: tuple[tuple] = ()
    last_scores: tuple[float] = ()
    
def get_random_joint_seqs(joint_seqs_length):
    joint_seqs = []
    for _ in range(joint_seqs_length):
        joint = random.choice(["P","R","F","T"])
        joint_seqs.append(joint)
    
    return tuple(joint_seqs)


def main(generation_num, agent_num, configs, epochs):
    # SEED
    set_seed(0)

    # Path
    BASEDIR, RUNMODE = get_BASERDIR(__file__)
    KAS_Result = KAS_Result_Template()
    
    ### Logger ###
    ARGS = read_ARGS((BASEDIR/'configs'/configs).absolute())
    csvlogger = CSVLogger(Path(BASEDIR).parent/"results"/"KAS"/ARGS.MODEL/ARGS.runname/"log.csv")
    
    
    WANDB = True
    if WANDB:
        set_wandb(pname = "KAS", runname=f"{configs}_{generation_num}_{agent_num}")
    ray.init(num_gpus=2)
    
    # Hyperparameters
    max_trial_limit = 100
    topKRate = 0.5
    eliteRate = 0.1
    joint_seqs_length = len(ARGS.joint_seqs)
    
    
    # For 1st generation
    n_generation = 1
    KAS_Result.archive = (tuple(['T' for _ in range(joint_seqs_length)]),)
    
    
    
    workers = []
    for n_agent in range(1, agent_num+1):
        ARGS = read_ARGS((BASEDIR/'configs'/configs).absolute())
        ARGS.epochs = epochs
        ARGS.WANDB = False
        SAVEPATH = Path(BASEDIR).parent/"results"/"KAS"/ARGS.MODEL/ARGS.runname/f"generation_{n_generation}"/f"agent_{n_agent}"
        ask_and_make_folder(SAVEPATH, ask = False)

        ### Set Path ###
        ARGS.SAVE_RESULT_PATH = SAVEPATH
        ARGS.SAVE_WEIGHT_PATH = SAVEPATH/"weights"

        Path.mkdir(ARGS.SAVE_WEIGHT_PATH,exist_ok=True, parents=True)
        Path.mkdir(ARGS.SAVE_RESULT_PATH,exist_ok=True, parents=True)


        for _ in range(max_trial_limit):
            
            joint_seqs = get_random_joint_seqs(joint_seqs_length)
            if joint_seqs not in KAS_Result.archive:
                # Set ARGS for Agent
                ARGS.joint_seqs = joint_seqs
                
                # Update Archive
                archive = list(KAS_Result.archive)
                archive.append(joint_seqs)
                KAS_Result.archive = tuple(archive)
                
                # Copy ARGS
                # shutil.copyfile((BASEDIR/'configs'/configs).absolute(), SAVEPATH/"args.py")
                
                # Add Worker
                workers.append(Ray_agent_main.remote(ARGS))
                break

        # Get Result
        
        def evaluate_agent(workers, KAS_Result: KAS_Result_Template, ARGS_list=None):
            results = ray.get(workers)
            # print(results)
            
            mean_score = np.mean([result.score for result in results])
            min_score = np.min([result.score for result in results])
            max_score = np.max([result.score for result in results])
            
            last_joint_seqs = []
            last_scores = []
            for result in results:
                if KAS_Result.best_score < result.score:
                    KAS_Result.best_score = result.score
                    KAS_Result.best_joint_seqs = result.joint_seqs
                    KAS_Result.state_dict = result.state_dict

                last_joint_seqs.append(result.joint_seqs)
                last_scores.append(result.score)
                
            KAS_Result.last_joint_seqs = tuple(last_joint_seqs)
            KAS_Result.last_scores = tuple(last_scores)
            
            
            log_dict = dict(best_joint_seqs = KAS_Result.best_joint_seqs, best_score = KAS_Result.best_score, mean_score = mean_score, min_score = min_score, max_score = max_score)
            csvlogger.log(log_dict)
            
            time.sleep(1)
            print("Current Best Joint Seqs: ", KAS_Result.best_joint_seqs, "Current Best Score: ", KAS_Result.best_score)
            
            WEIGHT_PATH = Path(BASEDIR).parent/"results"/"KAS"/ARGS.MODEL/ARGS.runname
            
            # Save state_dict
            state_dict = KAS_Result.state_dict
            state_dict['joint_seqs'] = KAS_Result.best_joint_seqs
            state_dict['scores'] = KAS_Result.best_score
            
            torch.save(KAS_Result.state_dict, WEIGHT_PATH/"best_state_dict.pth")
            
            # Save ARGS
            # shutil.copyfile((BASEDIR/'configs'/configs).absolute(), SAVEPATH/"args.py")
            
            if WANDB: wandb.log({"best_score": KAS_Result.best_score, "mean_score": mean_score, "min_score": min_score, "max_score": max_score}, step=n_generation)
            
            return KAS_Result
        
    KAS_Result = evaluate_agent(workers, KAS_Result)
    # results = ray.get(workers)
        
    
    # For 2nd ~ generation
    
    for n_generation in range(2, generation_num+1):
        print("Generation: ", n_generation)
        topKAgentNum, eliteAgentNum = int(agent_num*topKRate), int(agent_num*eliteRate)
        topKAgentIdx, eliteKAgentIdx = np.argsort(KAS_Result.last_scores)[-topKAgentNum:], np.argsort(KAS_Result.last_scores)[-eliteAgentNum:]
        topKAgent, eliteKAgent = [(KAS_Result.last_joint_seqs[i]) for i in topKAgentIdx], [(KAS_Result.last_joint_seqs[i]) for i in eliteKAgentIdx]

        n_agent = 1
        workers = []
        ARGS_list = []
        while n_agent <= agent_num:
            ARGS = read_ARGS((BASEDIR/'configs'/configs).absolute())
            ARGS.epochs = epochs
            ARGS.WANDB = False
            SAVEPATH = Path(BASEDIR).parent/"results"/"KAS"/ARGS.MODEL/ARGS.runname/f"generation_{n_generation}"/f"agent_{n_agent}"
            ask_and_make_folder(SAVEPATH, ask = False)

            ### Set Path ###
            ARGS.SAVE_RESULT_PATH = SAVEPATH
            ARGS.SAVE_WEIGHT_PATH = SAVEPATH/"weights"

            Path.mkdir(ARGS.SAVE_WEIGHT_PATH,exist_ok=True, parents=True)
            Path.mkdir(ARGS.SAVE_RESULT_PATH,exist_ok=True, parents=True)

            if n_generation == 5:
                print("here")


            if len(eliteKAgent) > 0:
                joint_seqs = eliteKAgent.pop()
                
                if joint_seqs not in KAS_Result.archive:
                    # Set ARGS for Agent
                    ARGS.joint_seqs = joint_seqs
                    
                    # Update Archive
                    archive = list(KAS_Result.archive)
                    archive.append(joint_seqs)
                    KAS_Result.archive = tuple(archive)
                    
                    # Copy ARGS
                    # shutil.copyfile((BASEDIR/'configs'/configs).absolute(), SAVEPATH/"args.py")
                    
                    # Add Worker
                    workers.append(Ray_agent_main.remote(ARGS))
                    ARGS_list.append(ARGS)
                    n_agent = n_agent + 1
                    continue
            
            else:
                for i in range(max_trial_limit):
                    parent1_idx,parent2_idx = random.sample(range(topKAgentNum),k=2)
                    parent1, parent2 = topKAgent[parent1_idx], topKAgent[parent2_idx]
                    bisection_point = random.randint(0,joint_seqs_length-1)
                    
                    joint_seqs = [parent1[i] if i <= bisection_point else parent2[i] for i in range(joint_seqs_length)]

                    if joint_seqs not in KAS_Result.archive:
                        # if joint_seqs == ('T','T','T','T','T','T','T'):
                        #     print('here')
                        
                        # Set ARGS for Agent
                        ARGS.joint_seqs = joint_seqs
                        
                        # Update Archive
                        archive = list(KAS_Result.archive)
                        archive.append(joint_seqs)
                        KAS_Result.archive = tuple(archive)
                        
                        # Copy ARGS
                        # shutil.copyfile((BASEDIR/'configs'/configs).absolute(), SAVEPATH/"args.py")
                        
                        # Add Worker
                        workers.append(Ray_agent_main.remote(ARGS))
                        ARGS_list.append(ARGS)
                        n_agent = n_agent + 1
                        break
                    if i == max_trial_limit-1:
                        print("############ GA has reached max trial limit ###############")
                        print(KAS_Result.best_joint_seqs)
                        return 1

        KAS_Result = evaluate_agent(workers, KAS_Result, ARGS_list)
            
    print("############ GA has finished ###############")            
    print(KAS_Result.best_joint_seqs)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'parse for DLPG')
    parser.add_argument("--generation_num", default=100, type=int)
    parser.add_argument("--agent_num", default=50, type=int)
    parser.add_argument("--configs", default="PRIMNET/ABAQUS_32.py",type=str) # [FC_PRIMNET, PRIMNET, PCC_PRIMNET, PRIMNET_FULL] # [FINGER, ABAQUS_32]
    parser.add_argument("--epochs", default=1000, type=int)
    args= parser.parse_args()

    main(generation_num=args.generation_num, agent_num=args.agent_num, configs=args.configs, epochs=args.epochs)

    
    



# %%
# import torch

# import os

# state_dict = torch.load("../results/KAS/PRIMNET/ABAQUS32/best_state_dict.pth")
# print(state_dict['joint_seqs'])