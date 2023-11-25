import csv
import os


class CSVLogger:
    def __init__(self, path):
        self.path = str(path)
        if os.path.exists(self.path):
            os.remove(path)
        
        self.clear = True
        
    def create(self,data:dict):
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()

    def log(self, *data:dict):
        logging = {}
        for data_ in data:
            
            # logging.update({f"{key}":float(value) for key, value in data_.items()})
            for key,value in data_.items():
                if isinstance(value,tuple) or isinstance(value,list):
                    logging.update({f"{key}":value})
                else:
                    logging.update({f"{key}":float(value)})
            
        if self.clear:
            self.create(logging)
            self.clear = False
        
        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=logging.keys())
            writer.writerow(logging)

from pathlib import Path
import os
import shutil


def ask_and_make_folder(path:Path, ask = True):
    if ask:
        if path.exists():
            print(f"Save Directory already exists! Delete {path.__str__()}?")
            print("d to delete, o to override")
            delete_folder = input()
            if delete_folder == 'd':
                shutil.rmtree(path.absolute())
                Path.mkdir(path, parents=True)
            elif delete_folder == 'o':
                print("OverRidding...!!")
            else:
                print("Exitting...")
                exit(1)
        else:    
            Path.mkdir(path, parents=True)
    else:
        if path.exists():
            print(f"Save Directory already exists! Deleting {path.__str__()}...")
            shutil.rmtree(path.absolute())
            Path.mkdir(path, parents=True)
        else:
            Path.mkdir(path, parents=True)