# %%
import json


data = json.load(open("ELASTICA_RAW.json","r"))
# %%
data.keys()

# %%
import numpy as np
np.random.seed(0)

motor_control = np.array(data['motor_control'])
position =  np.array(data['position'])

motor_control.shape, position.shape


# %%
motor_control_norm = np.linalg.norm(motor_control, axis = 1)



motor_control_norm.shape


# %%
idx = np.argsort(motor_control_norm)

idx

# %%
motor_control[idx[0]], motor_control[idx[-1]]


# %%
interpolation_data_number = int(len(idx) * 0.7)

interpolation_idx = idx[:interpolation_data_number]
extrapolation_idx = idx[interpolation_data_number:]

interpolation_idx.shape, extrapolation_idx.shape
# %%
# random shuffle interpolation_idx
np.random.shuffle(interpolation_idx)

train_data_number = int(len(interpolation_idx) * 0.8)
val_data_number = int(len(interpolation_idx) * 0.1)
test_data_number = int(len(interpolation_idx) * 0.1)

train_idx = interpolation_idx[:train_data_number]
val_idx = interpolation_idx[train_data_number:train_data_number+val_data_number]
test_idx = interpolation_idx[train_data_number+val_data_number:]

# %%
save_data = {}
scale = 0.1
save_data['train'] = dict(motor_control=motor_control[train_idx].tolist(),       position=(scale * position[train_idx]         ).reshape(-1,1,3).tolist())
save_data['val'] = dict(motor_control=motor_control[val_idx].tolist(),           position=(scale * position[val_idx]           ).reshape(-1,1,3).tolist())
save_data['test'] = dict(motor_control=motor_control[test_idx].tolist(),         position=(scale * position[test_idx]          ).reshape(-1,1,3).tolist())
save_data['ext'] = dict(motor_control=motor_control[extrapolation_idx].tolist(), position=(scale * position[extrapolation_idx] ).reshape(-1,1,3).tolist())


# %%
# save to json
json.dump(save_data, open("ELASTICA.json","w"))

# %%
pass