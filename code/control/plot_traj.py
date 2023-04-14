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
LF_result, LF_target_trajectory, LF_actuation = get_results("planned_traj/large_square_1/FC_PRIMNET_res.npz")
SP_result, SP_target_trajectory, SP_actuation = get_results("planned_traj/small_square_1/PRIMNET_res.npz")
SF_result, SF_target_trajectory, SF_actuation = get_results("planned_traj/small_square_1/FC_PRIMNET_res.npz")



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
plot_result(LP_result, LP_target_trajectory, lim=0.042)
plot_result(LF_result, LF_target_trajectory, lim=0.065)


# %%
plot_result(SP_result, SP_target_trajectory, lim=0.022)
plot_result(SF_result, SF_target_trajectory, lim=0.022)




# %%

fig,ax = plt.subplots(2,2,figsize=(10,10))

ax[0,0].plot(LP_result[1:,0], LP_result[1:,1], label="result", c='k')
ax[0,0].plot(LP_target_trajectory[1:,0], LP_target_trajectory[1:,1], '--', label="target", c='r')

ax[0,1].plot(LF_result[1:,0], LF_result[1:,1], label="result", c='k')
ax[0,1].plot(LF_target_trajectory[1:,0], LF_target_trajectory[1:,1], '--', label="target", c='r')


ax[1,0].plot(SP_result[1:,0], SP_result[1:,1], label="result", c='k')
ax[1,0].plot(SP_target_trajectory[1:,0], SP_target_trajectory[1:,1], '--', label="target", c='r')

ax[1,1].plot(SF_result[1:,0], SF_result[1:,1], label="result", c='k')
ax[1,1].plot(SF_target_trajectory[1:,0], SF_target_trajectory[1:,1], '--', label="target", c='r')


