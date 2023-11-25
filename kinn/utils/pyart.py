import torch

def check_qs(q):
    assert len(q.shape) == 1

def check_rpys(rpys):
    assert rpys.shape[-1] == 3
    assert len(rpys.shape) in [1,2]

def check_Ts(T):
    assert T.shape[-2:] == (4,4)
    assert len(T.shape) in [2,3]
    
def check_ps(p):
    assert p.shape[-2:] == (3,1)
    assert len(p.shape) in [2,3]

def check_Rs(R):
    assert R.shape[-2:] == (3,3)
    assert len(R.shape) in [2,3]

def check_ws(ws):
    assert ws.shape[-2:] == (3,1)
    assert len(ws.shape) in [2,3]

def is_qs_batch_tensor(qs):
    '''
    qs: (b) or (1)
    '''
    check_qs(qs)
    if len(qs) == 1:
        return False
    else:
        return True


def is_rpys_batch_tensor(rpys):
    '''
    rpys: (b,3) or (3)
    '''
    check_rpys(rpys)
        
    if len(rpys.shape) == 1:
        return False
    else:
        return True

def is_Ts_batch_tensor(Ts):
    '''
    T : (b,4,4) or (4,4)
    '''
    check_Ts(Ts)    
    if len(Ts.shape) == 2:
        return False
    else:
        return True


def is_ps_batch_tensor(ps):
    '''
    p : (b,3,1) or (3,1)
    '''
    check_ps(ps)
    
    if len(ps.shape) == 2:
        return False
    else:
        return True
    
def is_Rs_batch_tensor(Rs):
    '''
    R : (b,3,3) or (3,3)
    '''
    check_Rs(Rs)
    
    if len(Rs.shape) == 2:
        return False
    else:
        return True

def is_ws_batch_tensor(ws):
    '''
    ws(Axis) : (b,3,1) or (3,1)
    '''
    check_ws(ws)
    
    if len(ws.shape) == 2:
        return False
    else:
        return True


def t2pr(T=torch.zeros(16,4,4)):
    if is_Ts_batch_tensor(T):
        ps = T[:,0:3,3]
        Rs = T[:,0:3,0:3]
        return (ps,Rs)
    else:
        ps = T[0:3,3]
        Rs = T[0:3,0:3]
        
def t2r(T=torch.zeros(16,4,4)):
    if is_Ts_batch_tensor(T):
        Rs = T[:,0:3,0:3]
        return Rs
    else:
        Rs = T[0:3,0:3]
        return Rs
    
def t2p(Ts=torch.zeros(16,4,4)):
    if is_Ts_batch_tensor(Ts):
        ps = Ts[:,0:3,3]
        return ps.reshape(-1,3,1)
    else:
        ps = Ts[0:3,3]
        return ps.reshape(3,1)



def pr2t(ps=torch.zeros(16,3,1),Rs=torch.zeros(16,3,3)):
    assert ps.device == Rs.device
    
    device = ps.device

    ps_is_batch_tensor = is_ps_batch_tensor(ps)
    Rs_is_batch_tensor = is_Rs_batch_tensor(Rs)

    if ps_is_batch_tensor and Rs_is_batch_tensor:
        batch_size = ps.shape[0]
        
        Ts = torch.tile(torch.eye(4), [batch_size,1,1]).to(device)
        Ts[:,0:3,0:3] = Rs
        Ts[:,0:3,3] =  ps.reshape(batch_size,3)
        Ts[:,3,3] = 1
        return Ts
    
    elif not ps_is_batch_tensor and Rs_is_batch_tensor:
        batch_size = Rs.shape[0]
        
        Ts = torch.tile(torch.eye(4), [batch_size,1,1]).to(device)
        Ts[:,0:3,0:3] = Rs
        Ts[:,0:3,3] =  torch.tile(ps.reshape(3), [batch_size,1,1])
        Ts[:,3,3] = 1
        return Ts
    
    elif ps_is_batch_tensor and not Rs_is_batch_tensor:
        batch_size = ps.shape[0]
        
        Ts = torch.tile(torch.eye(4), [batch_size,1,1]).to(device)
        Ts[:,0:3,0:3] = torch.tile(Rs, [batch_size,1,1])
        Ts[:,0:3,3] =  ps.reshape(batch_size,3)
        Ts[:,3,3] = 1
        return Ts
        
    else:
        Ts = torch.eye(4).to(device)
        Ts[0:3,0:3] = Rs
        Ts[0:3,3] =  ps.reshape(3)
        Ts[3,3] = 1
        return Ts

def p2t(ps=torch.zeros(16,3)):
    device = ps.device
    if is_ps_batch_tensor(ps):
        batch_size = ps.shape[0]
        Ts = torch.tile(torch.eye(4), [batch_size,1,1]).to(device)
        Ts[:,0:3,3] =  ps[:]
        Ts[:,3,3] = 1
        return Ts
    else:
        Ts = torch.eye(4).to(device)
        Ts[0:3,3] =  ps
        Ts[3,3] = 1
        return Ts
    

def r2t(Rs=torch.zeros(16,3,3)):
    device = Rs.device
    if is_Rs_batch_tensor(Rs):
        batch_size = Rs.shape[0]
        Ts = torch.tile(torch.eye(4), [batch_size,1,1]).to(device)
        Ts[:,0:3,0:3] = Rs[:]
        Ts[:,3,3] = 1
        return Ts
    
    else:
        Ts = torch.eye(4).to(device)
        Ts[0:3,0:3] = Rs
        Ts[3,3] = 1
        return Ts

def rpy2r(rpys=torch.zeros(16,3)): # [radian]
    if is_rpys_batch_tensor(rpys):
        device = rpys.device
        Rs = torch.zeros(rpys.size()[0],3,3,dtype=torch.float).to(device)
        rs = rpys[:,0]
        ps = rpys[:,1]
        ys = rpys[:,2]
        Rs[:,0,:] = torch.vstack([
            torch.cos(ys)*torch.cos(ps),
            -torch.sin(ys)*torch.cos(rs) + torch.cos(ys)*torch.sin(ps)*torch.sin(rs),
            torch.sin(ys)*torch.sin(rs) + torch.cos(ys)*torch.sin(ps)*torch.cos(rs)
            ]).transpose(0,1)
        Rs[:,1,:] = torch.vstack([
            torch.sin(ys)*torch.cos(ps),
            torch.cos(ys)*torch.cos(rs) + torch.sin(ys)*torch.sin(ps)*torch.sin(rs),
            -torch.cos(ys)*torch.sin(rs) + torch.sin(ys)*torch.sin(ps)*torch.cos(rs)
            ]).transpose(0,1)
        Rs[:,2,:] = torch.vstack([
            -torch.sin(ps),
            torch.cos(ps)*torch.sin(rs),
            torch.cos(ps)*torch.cos(rs)
            ]).transpose(0,1)    
        return Rs
    else:
        device = rpys.device
        Rs = torch.zeros(3,3,dtype=torch.float).to(device)
        r = rpys[0]
        p = rpys[1]
        y = rpys[2]
        Rs[0,:] = torch.tensor([
            torch.cos(y)*torch.cos(p),
            -torch.sin(y)*torch.cos(r) + torch.cos(y)*torch.sin(p)*torch.sin(r),
            torch.sin(y)*torch.sin(r) + torch.cos(y)*torch.sin(p)*torch.cos(r)
            ])
        Rs[1,:] = torch.tensor([
            torch.sin(y)*torch.cos(p),
            torch.cos(y)*torch.cos(r) + torch.sin(y)*torch.sin(p)*torch.sin(r),
            -torch.cos(y)*torch.sin(r) + torch.sin(y)*torch.sin(p)*torch.cos(r)
            ])
        Rs[2,:] = torch.tensor([
            -torch.sin(p),
            torch.cos(p)*torch.sin(r),
            torch.cos(p)*torch.cos(r)
            ])
        return Rs

def r2rpy(Rs): # [radian]
    device = Rs.device
    
    r = torch.atan2(Rs[:,2,1],Rs[:,2,2])
    p = torch.atan2(-Rs[:,2,0],torch.sqrt(Rs[:,2,1]*Rs[:,2,1]+Rs[:,2,2]*Rs[:,2,2]))
    y = torch.atan2(Rs[:,1,0],Rs[:,0,0])
    rpys = torch.stack([r,p,y],dim=-1)
    return rpys

def skew(ps=torch.zeros(16,3)):
    device = ps.device
    check_ps(ps)
    
    # ps : Bx3x1 or 3x1
    if is_ps_batch_tensor(ps):
        skew_ps = torch.zeros(ps.size()[0],3,3,dtype=torch.float).to(device)
        zeros = torch.zeros(ps.size()[0],dtype=torch.float).to(device)
        skew_ps[:,0,:] = torch.vstack([zeros, -ps[:,2], ps[:,1]]).transpose(0,1)
        skew_ps[:,1,:] = torch.vstack([ps[:,2], zeros, -ps[:,0]]).transpose(0,1)
        skew_ps[:,2,:] = torch.vstack([-ps[:,1], ps[:,0],zeros]).transpose(0,1)
        return skew_ps
    else:
        skew_ps = torch.zeros(3,3,dtype=torch.float).to(device)
        skew_ps[0,:] = torch.tensor([0, -ps[2], ps[1]])
        skew_ps[1,:] = torch.tensor([ps[2], 0, -ps[0]])
        skew_ps[2,:] = torch.tensor([-ps[1], ps[0],0])
        return skew_ps
        
def rodrigues(w=torch.rand([3,1]), qs=torch.zeros(16),VERBOSE=False):
    assert w.device == qs.device
    
    device     = qs.device
    eps        = 1e-10
    q_is_batch_tensor = is_qs_batch_tensor(qs)
    
    if  q_is_batch_tensor:
        batch_size = qs.shape[0]

        if torch.norm(w) < eps:
            Rs = torch.tile(torch.eye(3),(batch_size,1,1)).to(device)
            return Rs
        if abs(torch.norm(w)-1) > eps:
            if VERBOSE:
                print("Warning: [rodirgues] >> joint twist not normalized")

        theta  = torch.norm(w)
        w      = w/theta        # [3]
        qs     = qs*theta       # [N]
        w_skew = skew(w.unsqueeze(0)).squeeze(0) # []
        Rs = torch.tensordot(
            torch.ones_like(qs).unsqueeze(0),torch.eye(3).unsqueeze(0).to(device),dims=([0],[0])
        ) \
        + torch.tensordot(
            torch.sin(qs).unsqueeze(0),w_skew.unsqueeze(0),dims=([0],[0])
        )\
        + torch.tensordot(
            (1-torch.cos(qs)).unsqueeze(0),(w_skew@w_skew).unsqueeze(0),dims=([0],[0])
        )
        return Rs
    
    else:
        if torch.norm(w) < eps:
            Rs = (torch.eye(3)).to(device)
            return Rs
        if abs(torch.norm(w)-1) > eps:
            if VERBOSE:
                print("Warning: [rodirgues] >> joint twist not normalized")

        theta  = torch.norm(w)
        w      = w/theta        # [3]
        qs     = qs*theta       # [N]
        w_skew = skew(w) # []
        Rs = torch.eye(3).to(device) \
        + torch.sin(qs) * w_skew \
        + (1-torch.cos(qs)) * (w_skew@w_skew)
        return Rs
        

def rpy2quat(rpys=torch.zeros(16,3)):
    assert len(rpys.shape)==2
    roll, pitch, yaw = rpys[:,0],rpys[:,1],rpys[:,2]
    qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
    qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
    qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    quat = torch.stack([qx, qy, qz, qw],dim=-1)
    return quat

def r2quat(Rs=torch.tile(torch.eye(3),[16,1,1])):
    rpys = r2rpy(Rs)
    quat = rpy2quat(rpys)
    return quat


# %%
import numpy as np
def rpy2quat_numpy(rpy=np.zeros(3)):
    # Notation from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    assert rpy.shape == (3,)
    cr,cp,cy = np.cos(rpy/2);
    sr,sp,sy = np.sin(rpy/2);
    # Construct quaternion
    q0 = cr*cp*cy - sr*sp*sy
    q1 = sr*cp*cy + cr*sp*sy
    q2 = cr*sp*cy - sr*cp*sy
    q3 = cr*cp*sy + sr*sp*cy
    
    quat = np.stack([q0,q1,q2,q3])
    return quat


#%%
def quat2rpy_numpy(quat = np.array([1,0,0,0.0])):
    assert quat.shape == (4,)
    q0,q1,q2,q3 = quat
    
    aSinInput = 2*(q1*q3 + q2*q0)
    if   aSinInput>1:  aSinInput = 1
    elif aSinInput<-1: aSinInput = -1
    
    r = np.arctan2( -2*(q2*q3 - q1*q0), q0**2 - q1**2 - q2**2 + q3**2 )
    p = np.arcsin(aSinInput)
    y = np.arctan2(-2*(q1*q2-q3*q0), q0**2+q1**2-q2**2-q3**2)
    rpy = np.stack([r,p,y])
    
    return rpy
    
def r2rpy_numpy(Rs): # [radian]
    r = np.arctan2(Rs[2,1],Rs[2,2])
    p = np.arctan2(-Rs[2,0],np.sqrt(Rs[2,1]*Rs[2,1]+Rs[2,2]*Rs[2,2]))
    y = np.arctan2(Rs[1,0],Rs[0,0])
    rpys = np.stack([r,p,y])
    return rpys

def rpy2r_numpy(rpys=np.zeros(3)): # [radian]
    Rs = np.zeros((3,3))
    rs = rpys[0]
    ps = rpys[1]
    ys = rpys[2]
    Rs[0,:] = np.stack([
        np.cos(ys)*np.cos(ps),
        -np.sin(ys)*np.cos(rs) + np.cos(ys)*np.sin(ps)*np.sin(rs),
        np.sin(ys)*np.sin(rs) + np.cos(ys)*np.sin(ps)*np.cos(rs)
        ])
    Rs[1,:] = np.stack([
        np.sin(ys)*np.cos(ps),
        np.cos(ys)*np.cos(rs) + np.sin(ys)*np.sin(ps)*np.sin(rs),
        -np.cos(ys)*np.sin(rs) + np.sin(ys)*np.sin(ps)*np.cos(rs)
        ])
    Rs[2,:] = np.stack([
        -np.sin(ps),
        np.cos(ps)*np.sin(rs),
        np.cos(ps)*np.cos(rs)
        ])    
    return Rs

def pr2t_numpy(p=np.zeros(3),r=np.eye(3)):
    T = np.eye(4)
    T[:3,:3] = r
    T[:3,3] = p
    return T

def t2pr_numpy(T=np.eye(4)):
    r = T[:3,:3]
    p = T[:3,3]
    return p,r

def t2p_numpy(T=np.eye(4)):
    p = T[:3,3]
    return p

def t2r_numpy(T=np.eye(4)):
    r = T[:3,:3]
    return r

def angle_numpy(a=np.array([1,0,0]), b=np.array([1,1,0])):
    return np.arccos(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)))

def angle_axis_numpy(axis=np.array([1,0,0]), before=np.array([0,1,0]), after=np.array([0,1,1])):
    Normal_Vector = np.cross(axis, before)
    alpha = angle_numpy(Normal_Vector, after)
    return np.pi/2 -alpha 

# %%
def rodriguesNoBatch(v,k,θ):
    """
    Rotating vector v about vector k with θ degrees
    """
    v_rot = v*np.cos(θ) \
          + np.cross(k,v)*np.sin(θ) \
          + np.dot(k,v)*(1-np.cos(θ))*k
    return v_rot


# %%
if __name__=="__main__":
    # %%
    ## UnitTest: r2rpy, rpy2r
    from kinn.utils.pyart import rpy2r, r2rpy
    rpys = torch.rand(16,3) 
    Rs = rpy2r(rpys)
    recon_rpys = r2rpy(Rs)
    diff =recon_rpys-rpys
    torch.where(diff<1e-7,torch.zeros_like(diff),diff)
    
    