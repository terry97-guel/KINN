import numpy as np 
from kinematics.tools import \
    column_v, rodrigues, column, rot_e, Rotation_E, pr2t, t2p, t2r, make_rotation, get_color_from_urdf
from kinematics.parser import PARSER
from kinematics.config import CONFIG
from typing import List

class CAPSULE():
    def __init__(self, 
                 scale=[], 
                 cap_radius = float(0), 
                 cap_height = float(0), 
                 cap_p      = column_v(0,0,0), 
                 cap_R      = rot_e(), 
                 cap_center_p = column_v(0, 0, 0), 
                 cap_size     = column_v(0,0,0), 
                 cap_t_offset =np.eye(4)):

        self.scale    = scale 
        self.radius   = cap_radius
        self.height   = cap_height
        self.p        = cap_p
        self.R        = cap_R
        self.center_p = cap_center_p
        self.size     = cap_size
        self.T_offset = cap_t_offset
    

class JOINT():
    def __init__(self, 
                 name   = str('joint'), 
                 id     = int(0), 
                 mother = 0, 
                 child  =[0], 
                 q      = float(0), 
                 a = column_v(1,1,1), 
                 b = column_v(0,0,0),
                 p = column_v(0,0,0), 
                 R = rot_e(), 
                 R_offset = Rotation_E(), 
                 dq = float(0), 
                 type =''):

        self.name     = name 
        self.id       = id 
        self.mother   = mother
        self.child    = child 
        self.q        = q               # joint value
        self.a        = a               # joint axis
        self.b        = b               # joint offset in poisition
        self.p        = p               # joint transformation in position
        self.R        = R               # joint transformation in rotation
        self.R_offset = R_offset        # joint offset in rotation
        self.dq       = dq              # time derivate of joint value
        self.type     = type            # joint type


class LINK(CAPSULE):
    def __init__(self, 
                 name       = str('link'), 
                 joint_id   = int(0), 
                 joint_name = str('mother_joint'), 
                 P_offset     = column_v(0,0,0), 
                 R_offset     = rot_e(), 
                 cap_center_p = column_v(0, 0, 0),
                 scale= [] , 
                 cap_radius   = float(0), 
                 cap_height   = float(0), 
                 cap_t_offset = np.eye(4),
                 mesh_path    = str('path'), 
                 cap_p = column_v(0,0,0), 
                 cap_R = rot_e(), 
                 color = "0.2 0.2 0.2 1", 
                 collision_type = "mesh", 
                 cap_size       = column_v(0,0,0)):
        CAPSULE.__init__(self, scale, cap_radius, cap_height, cap_p, cap_R, cap_center_p, cap_size, cap_t_offset)

        self.name           = name 
        self.joint_id       = joint_id 
        self.joint_name     = joint_name 
        self.P_offset       = P_offset
        self.R_offset       = R_offset 
        self.mesh_path      = mesh_path 
        self.color          = color 
        self.collision_type = collision_type

        # Capsule 
        self.cap = CAPSULE()
        self.cap.scale  = scale
        self.cap.radius = cap_radius 
        self.cap.height = cap_height
        self.cap.p      = cap_p
        self.cap.R      = cap_R
        self.cap.size   = cap_size #In case of box type
        self.cap.T_offset = cap_t_offset #In case of mesh type
        self.cap.center_p = cap_center_p #In case of mesh type

class CHAIN:
    def __init__(self, file_name="ir_gazebo/script/structure/utils/ur5e_onrobot.urdf",
                       base_offset=[0.1, 0, 0.5], 
                       verbose=False):
        self.parser = PARSER(file_name)
        self.base_offset = base_offset
        self.verbose = verbose
        joint = self.get_joint()
        self.joint = joint
        link = self.get_link()
        self.link = link
        self.fk_chain(1)
        
    def fk_chain(self, idx_to):
        if idx_to == 0:                                         # This line seems to never be activated! May be idx_to == 1?
            joint_to = self.joint[idx_to]
            link_to  = self.link[idx_to]
            p = joint_to.b
            r = joint_to.R
            T_capsule = np.matmul(pr2t(p, r) ,link_to.T_offset)
            link_to.cap.center_p = t2p(T_capsule)
            return None 
        idx_fr = self.joint[idx_to-1].mother 
        if idx_fr != 0:
            link_fr  = self.link[idx_fr-1]
            try:
                link_to  = self.link[idx_to-1]
            except Exception:
                print("Here")
            joint_fr = self.joint[idx_fr-1]
            joint_to = self.joint[idx_to-1]

            if joint_to.type in ["revolute","fixed","continuous"]:
                joint_to.p = (np.matmul(joint_fr.R, joint_to.b) + joint_fr.p).astype(float)
                link_to.cap.p = joint_to.p + link_fr.P_offset

                if (joint_to.a == column_v(1,1,1)).all() or joint_to.type=="fixed":
                    joint_to.R = np.matmul(joint_fr.R, joint_to.R_offset)
                else:   
                    joint_fr_after = np.matmul(joint_fr.R, joint_to.R_offset)
                    joint_to.R = np.matmul(joint_fr_after, rodrigues(joint_to.a, joint_to.q)).astype(float)
                    
            else: 
                raise TypeError(f'Found joint type {joint_to.type}. \n Expected one of ["revolute", "fixed", "prismatic", "continuous"]')
            

            T_capsule = np.matmul(pr2t(joint_to.p, joint_to.R) ,link_to.T_offset)
            link_to.cap.center_p = t2p(T_capsule)
            link_to.cap.R = t2r(T_capsule)

        for child_idx in self.joint[idx_to-1].child:
            self.fk_chain(child_idx)

    # Add joint to robot 
    def get_joint(self) -> List[JOINT]: 
        joints = []
        # Other joints      
        for idx, joint in enumerate(self.parser.joint): 
            if idx == 0: 
                joints.append(JOINT(name=joint['name'], 
                                            id=joint['id'], 
                                            mother = self.parser.get_mother(joint['parent']), 
                                            child=self.parser.get_child_joint_tree(joint["child"]), q=0, 
                                            a=column(joint['axis']), 
                                            b=column(joint['xyz']), 
                                            p=column_v(self.base_offset[0], 
                                                       self.base_offset[1],
                                                       self.base_offset[2]), # Attached position offset from the world 
                                            R=np.eye(3), 
                                            R_offset=make_rotation(rad=joint['rpy']),
                                            type=joint['type']))          
            
            else: 
                joints.append(JOINT(name=joint['name'], 
                                            id=joint['id'], 
                                            mother = self.parser.get_mother(joint['parent']), 
                                            child=self.parser.get_child_joint_tree(joint["child"]), q=0, 
                                            a=column(joint['axis']), 
                                            b=column(joint['xyz']), 
                                            p=column_v(0, 0, 0), 
                                            R=np.eye(3), 
                                            R_offset=make_rotation(rad=joint['rpy']),
                                            type=joint['type']))   
                                             

        """ Verbose Function """
        if self.verbose:
            for joi in joints: 
                print("Name: {}, ID: {}, Mother_joint: {}, Child_joint: {}, Type: {}, Axis: {}, B_offset:{}".
                    format(joi.name.ljust(35), str(joi.id).ljust(2), str(joi.mother).ljust(2), 
                           str(joi.child).ljust(11), joi.type.ljust(10), str(joi.a.T).ljust(16), 
                           str(joi.b.T).ljust(5)))
                print("Current Position: {}\nCurrent Rotation: \n{}\n".format(joi.p.T, joi.R))
        
        return joints         

    def get_link(self):
        links = []
        """ CONNECT LINK TREE """
        for idx, link in enumerate(self.parser.link):
            if idx <= 6:
                capsule = CONFIG["capsule"][idx]
            
            links.append(LINK(name=link['name'], 
                                        joint_id   = self.parser.get_link_joint_id(link['name']), 
                                        joint_name = self.parser.get_link_joint_name(link['name']), 
                                        P_offset   = column(link["P_offset"]), 
                                        R_offset   = make_rotation(rad=link["R_offset"]), 
                                        scale      = link['scale'], 
                                        mesh_path  = link["filepath"], 
                                        collision_type = link["collision_type"], 
                                        cap_p        = column_v(0.0, 0.0, 0.0), 
                                        cap_R        = np.eye(3),
                                        cap_radius   = capsule["radius"],
                                        cap_size     = capsule["size"],
                                        cap_t_offset = capsule["T_offset"],
                                        cap_center_p = column_v(0.0, 0.0, 0.0),
                                        cap_height   = capsule["height"],
                                        color        = get_color_from_urdf(link['color'])))

        # CONNECT ALL JOINT/LINKS TO THE ROBOT 
        if self.verbose:
            print('len_link', len(links))
            for link in links:
                print("Name: {}, Connected Joint ID: {}, Connected Joint name: {}, \nMesh Path: {}, \
                    \nCollision Type: {}, Radius: {}, Heihgt: {}, Cap_Position: {}, \nCenter Position: {}\n".
                    format(link.name.ljust(30), str(link.joint_id).ljust(2), link.joint_name.ljust(30), 
                           link.mesh_path, link.collision_type, str(link.radius).ljust(10), str(link.height).ljust(10), 
                           link.cap.p.T, link.cap.center_p))

        return links
