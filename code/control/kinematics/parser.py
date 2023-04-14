from collections import defaultdict
import xml.etree.ElementTree as ET
import copy
from kinematics.tools import column_v



class PARSER():
    def __init__(self, _file_name="/ur5e_onrobot.urdf"): #ir_gazebo/script/structure/ur5e_onrobot.urdf
        # UR5e  
        self.file_name = _file_name 
        # parse xml file
        self.doc = ET.parse(self.file_name)
        # Get root node from URDF
        self.root = self.doc.getroot()
        # NAME & RPY & XYZ & AXIS & PARENT & CHILD 
        self.joint = []
        self.joint_dict = {}
        # MESH FILE, SCALE 
        self.link= []
        self.link_dict  = {} 
        # INITAILIZE TREE 
        self.get_kinematics_tree()

    """ CONNECT JOINT TO LINK """
    def get_kinematics_tree(self):
        # Get link information
        for link in self.root.iter("link"):
            try:
                self.link_dict['name'] = link.attrib['name']
                self.link_dict['filepath'] = link.find("visual").find("geometry").find("mesh").attrib['filename']
                self.link_dict["collision_type"] = 'mesh'
                if link.find('visual') != None: 
                    if link.find('visual').find("origin") != None:
                        self.link_dict["P_offset"] = link.find('visual').find("origin").attrib["xyz"]
                        self.link_dict["R_offset"] = link.find('visual').find("origin").attrib["rpy"]
                    else:
                        self.link_dict["P_offset"] = "0 0 0"
                        self.link_dict["R_offset"] = "0 0 0"
                    # In case, when 'color' is not in urdf 
                    if  link.find("visual").find("material") != None:
                        self.link_dict['color'] = link.find("visual").find("material").find("color").attrib['rgba']
                    else:
                        self.link_dict['color'] ='0.2 0.2 0.2 1'
                    # In case, when 'scale' is not in urdf 
                    if 'scale' in link.find("visual").find("geometry").find("mesh").attrib:
                        self.link_dict['scale'] = link.find("visual").find("geometry").find("mesh").attrib['scale']
                    else: 
                        self.link_dict['scale'] = '1 1 1'
                else: 
                    self.link_dict["P_offset"] = "0 0 0"
                    self.link_dict["R_offset"] = "0 0 0"

            except AttributeError as a:
                self.link_dict['name'] = link.attrib['name']
                self.link_dict['filepath'] = None 
                if self.link_dict['filepath'] == None: 
                    self.link_dict["collision_type"] = None
            # Append to link list 
            link_dict_copy = self.link_dict.copy()
            self.link.append(link_dict_copy)

        # Get joint information
        for idx, joint in enumerate(self.root.iter("joint")):
            try: 
                self.joint_dict['name'] = joint.attrib['name']
                self.joint_dict["parent"] = joint.find('parent').attrib["link"]
                self.joint_dict["child"]  = joint.find('child').attrib["link"]
                self.joint_dict["id"]     = idx+2 # Base joint ID: 1 and init idx start from 0 that's why idx is added +2
                self.joint_dict['type'] = joint.attrib['type']
                self.joint_dict['rpy']  = joint.find('origin').attrib['rpy']
                self.joint_dict['xyz']  = joint.find('origin').attrib['xyz']
                self.joint_dict['p_offset']   = None
                if joint.find('axis') == None: #In case that "axis" is not in URDF
                    self.joint_dict["axis"] = "1 1 1"
                else:
                    self.joint_dict["axis"] = joint.find('axis').attrib['xyz']

                joint_dict_copy = self.joint_dict.copy()
                self.joint.append(joint_dict_copy)
            except (AttributeError, KeyError) as e:
                continue 

        # Base Joint 
        self.joint_dict["name"] = "base_joint"
        self.joint_dict["parent"] = "world_link" 
        self.joint_dict["child"] = self.joint[0]["parent"]
        self.joint_dict["id"] = 1 
        self.joint_dict["type"] = "fixed"
        self.joint_dict["rpy"] = '0 0 0'
        self.joint_dict["xyz"] = '0 0 0'
        self.joint_dict['axis'] = '0 0 1' #Glboal Configuration: Z axis default
        self.joint_dict["p_offset"] = column_v(0.18, 0, 0.25)
        base_joint_dict = self.joint_dict.copy()
        self.joint.insert(0, base_joint_dict) 
        
    def get_mother(self, mother):
        for joint in self.joint: 
            if joint["child"] == mother:
                return joint["id"]
        return 0  # The Base joint has no parent link


    def get_child_joint_tree(self, child_link):
        child_joint_lst = []
        for joi in self.joint:
            if joi["parent"] == child_link:
                child_joint_lst.append(joi["id"])
        if child_joint_lst == []:
            return [0]
        return child_joint_lst

    def get_link_joint_name(self, link_name):
        for joi in self.joint: 
            if joi['child'] == link_name:  
                link_joint_name = joi['name']
                return link_joint_name
        return "None"

    def get_link_joint_id(self, link_name):
        for joi in self.joint: 
            if joi['child'] == link_name:  
                link_joint_id = joi['id']
                return link_joint_id
        return "None"

