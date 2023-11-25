# %%
from pathlib import Path
import sys
import os

from enum import Enum
import enum


class Runmode(Enum):
    JUPYTER = enum.auto()
    DEBUG = enum.auto
    RUN = enum.auto()

    
JUPYTER = Runmode.JUPYTER
DEBUG = Runmode.DEBUG
RUN = Runmode.RUN


def get_BASERDIR(path):
    BASEDIR = Path(path).parent
    
    try: 
        get_ipython().__class__.__name__
        print("Running on Jupyter...")
        RUNMODE = JUPYTER
        return BASEDIR, RUNMODE
    except: 
        gettrace = getattr(sys, 'gettrace', None)
            
        if gettrace() is None:
            print('Run Mode...')
            RUNMODE = RUN
            
        elif gettrace():
            print('Debugging Mode...')
            RUNMODE = DEBUG
            
        else:
            print("Can't Recognize Running Configuration")
            sys.exit(0)
    
        return BASEDIR, RUNMODE


# def chdir_level(path,level):
#     BASEDIR_, RUNMODE_= get_BASERDIR(path)
    
#     for _ in range(level):
#         BASEDIR_ = BASEDIR_.parent
    
#     os.chdir(BASEDIR_)
#     sys.path.append(".")
#     return BASEDIR_, RUNMODE_s