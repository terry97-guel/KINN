
# convert data for temporary purposes
# %%

import json
from pathlib import Path

datapath = Path("dataset/FINGERN.json")

with open(datapath, "r") as f:
    data = json.load(f)
    
print(data)

# %%

transformation_matrix = np.array(
    [
        [ 0,-1, 0],
        [ 0, 0, 1],
        [-1, 0, 0]
    ]
)

np.array([-1,0,0])@transformation_matrix, np.array([0,1,0])@transformation_matrix,np.array([0,0,-1])@transformation_matrix

# %%
import numpy as np

for key in data.keys():
    partial_data = data[key]['position'] 
    partial_data
    temp = np.array(partial_data)
    transformed_partial_data = temp@transformation_matrix
    data[key]['position']  = (transformed_partial_data).tolist()


# %%
from matplotlib import pyplot as plt

for key in data.keys():
    partial_data = np.array(data[key]['position'])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(partial_data[:,0,0], partial_data[:,0,1], partial_data[:,0,2])
    plt.show()
    

# %%

# save the result_json as "FINGER.json"
with open("dataset/FINGER.json", "w") as f:
    json.dump(data, f)
    
# %%
