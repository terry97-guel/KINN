import numpy as np
import math

# cast

def column_v(x,y,z):
    return np.array([[x, y, z]]).T

def column(input):
    arr = np.array([])
    for i in input.split(): 
        i = float(i)
        arr = np.append(arr, np.array([i]), axis=0)
    arr = np.array([arr])
    return arr.T


def rodrigues(a, q):
    norm_a=np.linalg.norm(a)
    if norm_a <1e-10:
        R = np.eye(3)
    else:
        a = a/norm_a
        th = norm_a*q
        a_hat = np.array([[0,  -a[2], a[1]],
                          [a[2],  0, -a[0]],
                          [-a[1], a[0], 0]], dtype=object)
        R = np.eye(3) + a_hat * np.sin(th) + np.linalg.matrix_power(a_hat, 2) * (1-np.cos(th))
    return R 




# Transform 

def rot_e():
    e = np.array([[1, 	       0, 	      0],
             	  [0,          1,         0],
             	  [0,          0,         1]])
    return e


def rot_x(rad):
    roll = np.array([[1, 	       0, 	         0],
             		 [0, np.cos(rad), -np.sin(rad)],
             		 [0, np.sin(rad),  np.cos(rad)]])
    return roll 


def rot_y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad)],
                      [0,		    1, 	         0],
                      [-np.sin(rad),0, np.cos(rad)]])
    return pitch


def rot_z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0],
         	        [np.sin(rad),  np.cos(rad),  0],
              		[0, 			         0,  1]])
    return yaw 


def Rotation_E(): 
    e = np.array([[1, 	       0, 	      0,    0],
             	  [0,          1,         0,    0],
             	  [0,          0,         1,    0],
             	  [0,		   0,	      0,    0]])
    return e


def Rotation_X(rad):
    roll = np.array([[1, 	       0, 	      0,    0],
             		 [0, np.cos(rad), -np.sin(rad), 0],
             		 [0, np.sin(rad),  np.cos(rad), 0],
             		 [0,		   0,	      0,    0]])
    return roll 


def Rotation_Y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad), 0],
              		  [0,		    1, 	         0, 0],
              		  [-np.sin(rad),0, np.cos(rad), 0],
              		  [0, 		    0, 	         0, 0]])
    return pitch


def Rotation_Z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0, 0],
         	        [np.sin(rad),  np.cos(rad),  0, 0],
              		[0, 			         0,  1, 0],
             		[0, 			         0,  0, 0]])
    return yaw 
'''
def rpy2r(rpy):
    r_rad = rpy[0]
    p_rad = rpy[1]
    y_rad = rpy[2]

    cos_r = np.cos(r_rad); sin_r = np.sin(r_rad)
    cos_p = np.cos(p_rad); sin_p = np.sin(p_rad)
    cos_y = np.cos(y_rad); sin_y = np.sin(y_rad)

    R = np.array([
        [cos_y*cos_p,   -sin_y*cos_r+cos_y*sin_p*sin_r, sin_y*sin_r+cos_y*sin_p*cos_r],
        [sin_y*cos_p,    cos_y*cos_r+sin_y*sin_p*sin_r,  -cos_y*sin_r+sin_y*sin_p*cos_r],
        [-sin_p,         cos_p*sin_r,                    cos_p*cos_r]
    ])

    return R
def r2rpy(R):
    assert R.shape == (3,3)
    r = np.arctan2(R[2,1], R[2,2])
    p = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    y = np.arctan2(R[1,0], R[0,0])

    rpy = np.array([r,p,y]).reshape(-1)
    return rpy

'''
def Translation(x , y, z):
    Position = np.array([[0, 0, 0, x],
                         [0, 0, 0, y],
                         [0, 0, 0, z],
                         [0, 0, 0, 1]])
    return Position


def HT_matrix(Rotation, Position):
    Homogeneous_Transform = Rotation + Position
    return Homogeneous_Transform


def pr2t(position, rotation): 
    position_4diag  = np.array([[0, 0, 0, position[0]],
                                [0, 0, 0, position[1]],
                                [0, 0, 0, position[2]], 
                                [0, 0, 0, 1]], dtype=object)
    rotation_4diag  = np.append(rotation,[[0],[0],[0]], axis=1)
    rotation_4diag_ = np.append(rotation_4diag, [[0, 0, 0, 1]], axis=0)
    ht_matrix = position_4diag + rotation_4diag_ 
    return ht_matrix


def t2p(ht_matrix):
    return ht_matrix[:-1, -1]


def t2r(ht_matrix):
    return ht_matrix[:-1, :-1]

def t2pr(ht_matrix):
    return t2p(ht_matrix), t2r(ht_matrix)

def make_rotation(rad=0):
    for idx, rad_num in enumerate(rad.split()):
        if idx == 0 and float(rad_num) !=0:
            idx0 = rot_x(float(rad_num))
        elif idx==0 and float(rad_num) == 0: 
            idx0 = rot_e()
        if idx == 1 and float(rad_num) !=0:
            idx1 = rot_y(float(rad_num))
        elif idx==1 and float(rad_num) == 0: 
            idx1 = rot_e()
        if idx == 2 and float(rad_num) !=0:
            idx2 = rot_z(float(rad_num))
        elif idx==2 and float(rad_num)==0: 
            idx2 = rot_e()
    rot = idx2.dot(idx1).dot(idx0) 
    return rot

def decompose_rotation_matrix(R):
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])



# urdf
def get_color_from_urdf(color_value):
    color_value_list = []
    for color in color_value.split():
        color_value_list.append(float(color))
    return color_value_list