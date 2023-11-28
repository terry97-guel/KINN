# %%
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

def get_results(path):
    data = np.load(path)

    target_trajectory = data['target_trajectory'] * 1000
    actuation = data['actuation']
    result = data['result'] * 1000

    print(result.shape, target_trajectory.shape, actuation.shape)
    return result, target_trajectory, actuation


LP_result, LP_target_trajectory, LP_actuation = get_results("planned_traj/large_square_1/PRIMNET_res.npz")
SP_result, SP_target_trajectory, SP_actuation = get_results("planned_traj/small_square_1/PRIMNET_res.npz")
LF_result, LF_target_trajectory, LF_actuation = get_results("planned_traj/large_square_1/FC_PRIMNET_res.npz")
SF_result, SF_target_trajectory, SF_actuation = get_results("planned_traj/small_square_1/FC_PRIMNET_res.npz")
LC_result, LC_target_trajectory, LC_actuation = get_results("planned_traj/large_square/PCC_res.npz")
SC_result, SC_target_trajectory, SC_actuation = get_results("planned_traj/small_square/PCC_res.npz")



# %%
# Plot the result
import matplotlib.pyplot as plt

def plot_result(result, target_trajectory, lim):
    fig = plt.figure(figsize=(8,8))
    plt.plot(result[1:,0], result[1:,1], label="result", c='k')
    plt.plot(target_trajectory[1:,0], target_trajectory[1:,1], '--', label="target", c='r')
    plt.axis('equal')
    
    
    fontsize = 18
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    lim = lim * 1000
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.show()


# %%
plot_result(LP_result, LP_target_trajectory, lim=0.065)
plot_result(LF_result, LF_target_trajectory, lim=0.065)
plot_result(LC_result, LP_target_trajectory, lim=0.065)


# %%
plot_result(SP_result, SP_target_trajectory, lim=0.065)
plot_result(SF_result, SF_target_trajectory, lim=0.065)
plot_result(SC_result, SF_target_trajectory, lim=0.065)




# %%

fig,ax = plt.subplots(2,3,figsize=(10,10))

ax[0,0].plot(LP_result[1:,0], LP_result[1:,1], label="result", c='k')
ax[0,0].plot(LP_target_trajectory[1:,0], LP_target_trajectory[1:,1], '--', label="target", c='r')

ax[0,1].plot(LF_result[1:,0], LF_result[1:,1], label="result", c='k')
ax[0,1].plot(LF_target_trajectory[1:,0], LF_target_trajectory[1:,1], '--', label="target", c='r')

ax[0,2].plot(LP_result[1:,0], LP_result[1:,1], label="result", c='k')
ax[0,2].plot(LP_target_trajectory[1:,0], LP_target_trajectory[1:,1], '--', label="target", c='r')


ax[1,0].plot(SP_result[1:,0], SP_result[1:,1], label="result", c='k')
ax[1,0].plot(SP_target_trajectory[1:,0], SP_target_trajectory[1:,1], '--', label="target", c='r')

ax[1,1].plot(SF_result[1:,0], SF_result[1:,1], label="result", c='k')
ax[1,1].plot(SF_target_trajectory[1:,0], SF_target_trajectory[1:,1], '--', label="target", c='r')

ax[1,2].plot(SC_result[1:,0], SC_result[1:,1], label="result", c='k')
ax[1,2].plot(SC_target_trajectory[1:,0], SC_target_trajectory[1:,1], '--', label="target", c='r')

import matplotlib.pyplot as plt

# Your existing code for subplots

# Setting equal aspect ratio for all subplots
for i in range(2):
    for j in range(3):
        ax[i, j].set_aspect('equal')

# Adding titles along the columns
column_titles = ['KINN', 'PureNN', 'PCC']
for j, title in enumerate(column_titles):
    ax[0, j].set_title(title)

# Adding labels and legends (assuming you want similar settings for all subplots)
for i in range(2):
    for j in range(3):
        ax[i, j].set_xlabel('X Label')
        ax[i, j].set_ylabel('Y Label')
        # ax[i, j].legend()

# Show the plot
plt.tight_layout()
plt.show()


# %%
def remove_even_idx(array):
    length = len(array)
    return array[1:length:2]

# LP_result = remove_even_idx(LP_result)
# LF_result = remove_even_idx(LF_result)
# SP_result = remove_even_idx(SP_result)
# SF_result = remove_even_idx(SF_result)


import torch
def p_loss_fn(pred, target):
    if pred.shape == target.shape:
        target = target[1:]
        pred = pred[1:]
    else:
        target = target[1:]
        pred = remove_even_idx(pred)
        assert pred.shape == target.shape
    
    pred = pred[:,:2]
    target = target[:,:2]
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    return torch.nn.L1Loss()(pred, target)

print("Proposed Model/ Small Square:", p_loss_fn(SP_result, SP_target_trajectory))
print("Proposed Model/ Large Square:", p_loss_fn(LP_result, LP_target_trajectory))
print("FC Model/ Small Square:", p_loss_fn(SF_result, SF_target_trajectory))
print("FC Model/ Large Square:", p_loss_fn(LF_result, LF_target_trajectory))
print("PCC Model/ Small Square:", p_loss_fn(SC_result, SC_target_trajectory))
print("PCC Model/ Large Square:", p_loss_fn(LC_result, LC_target_trajectory))


# %%
# plot SF_result

