#! /usr/bin/env python
import rospy
from tf import transformations
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
import math 
import numpy as np

def publish_viz_robot(boxes):       
        markers = []
        for i, box in enumerate(boxes):
            if box[6] == None:
                pass
            else:
                marker = Marker(type=Marker.MESH_RESOURCE, mesh_use_embedded_materials = True, action = Marker.ADD,\
                                lifetime = rospy.Duration(secs=1/20), color = ColorRGBA(0.2, 0.2, 0.2, 1.0))
                marker.id = i
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = 'base_link' 
                quaternion = transformations.quaternion_from_euler(box[3], box[4], box[5])
                marker.pose.position.x = box[0]
                marker.pose.position.y = box[1]
                marker.pose.position.z = box[2]
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]  
                marker.scale.x = float(box[7])
                marker.scale.y = float(box[8])
                marker.scale.z = float(box[9])
                marker.color.r = box[10]
                marker.color.g = box[11]
                marker.color.b = box[12]
                marker.color.a = box[13]
                marker.mesh_resource = str(box[6])

                markers.append(marker)
        return markers

def euclidean_dist(point1, point2):
    return math.sqrt(sum([math.pow(point1[i] - point2[i], 2)
                          for i in range(len(point1))]))

def publish_viz_line(ee_position_lst, color_r, color_g, color_b):
        line_strip = Marker(
                header=Header(frame_id='base_link', stamp=rospy.Time.now()),
                ns="points_and_lines",
                action=Marker.ADD,
                id=1,
                type=Marker.LINE_STRIP,
                # LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
                color=ColorRGBA(color_r, color_g, color_b, 1.0)
        )

        for ee_position in ee_position_lst:
                for x, y, z in zip(ee_position[0], ee_position[1], ee_position[2]):
                        p = Point(x, y, z)
                        line_strip.points.append(p)
                        line_strip.scale.x = 0.01
                        line_strip.pose.orientation.w = 1.0
        return line_strip 

def publish_viz_markers(boxes):
        markers = []
        for i, box in enumerate(boxes):
                marker = Marker()
                marker.header.frame_id = "base_link" #odom, #map
                marker.header.stamp = rospy.Time.now()
                if box["type"] == "sphere":
                        marker.type = Marker.SPHERE
                elif box["type"] == 'cylinder':
                        marker.type = Marker.CYLINDER
                elif box["type"] == "cube":
                        marker.type = Marker.CUBE
                elif box["type"] == "arrow":
                        marker.type = Marker.ARROW
                else: raise TypeError(f"No valid marker type called: {box['type']}")
                marker.action = Marker.ADD
                marker.id = i
                quaternion = transformations.quaternion_from_euler(box["info"][3], box["info"][4], box["info"][5])
                marker.pose.position.x = box["info"][0]
                marker.pose.position.y = box["info"][1]
                marker.pose.position.z = box["info"][2]
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                marker.scale.x = box["info"][-3]
                marker.scale.y = box["info"][-2]
                marker.scale.z = box["info"][-1]
                marker.color.r = box["color"][0]
                marker.color.g = box["color"][1]
                marker.color.b = box["color"][2]
                marker.color.a = box["color"][3]
                marker.lifetime = rospy.Duration()
                markers.append(marker)
        return markers


def publish_box_mesh(boxes):
        markers = []
        for i, box in enumerate(boxes):
                marker = Marker()
                marker.header.frame_id = "base_link" 
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.id = i
                quaternion = transformations.quaternion_from_euler(box[3], box[4], box[5])
                marker.pose.position.x = box[0]
                marker.pose.position.y = box[1]
                marker.pose.position.z = box[2]
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                marker.scale.x = box[-3]
                marker.scale.y = box[-2]
                marker.scale.z = box[-1]
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.5
                marker.lifetime = rospy.Duration()
                
                markers.append(marker)
        return markers


def publish_sphere_mesh(boxes):
        markers = []
        for i, box in enumerate(boxes):
                marker = Marker()
                marker.header.frame_id = "base_link" #odom, #map
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = i
                quaternion = transformations.quaternion_from_euler(box[3], box[4], box[5])
                marker.pose.position.x = box[0]
                marker.pose.position.y = box[1]
                marker.pose.position.z = box[2]
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                marker.scale.x = box[6]*2
                marker.scale.y = box[7]*2
                marker.scale.z = box[8]*2
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.5
                marker.lifetime = rospy.Duration()
                markers.append(marker)
        return markers

def publish_cylinder(boxes):
        markers = []
        for i, box in enumerate(boxes):
                marker = Marker()
                marker.header.frame_id = "base_link" #odom, #map
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.id = i
                quaternion = transformations.quaternion_from_euler(box[3], box[4], box[5])
                marker.pose.position.x = box[0]
                marker.pose.position.y = box[1]
                marker.pose.position.z = box[2]
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                marker.scale.x = box[6]*2
                marker.scale.y = box[7]*2
                marker.scale.z = box[8]
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.0
                marker.lifetime = rospy.Duration()
                
                markers.append(marker)
        return markers

def publish_marker_cube(box ,r=0, g=1, b=0):
        marker = Marker()
        marker.header.frame_id = "base_link" #"odom" #map
        marker.header.stamp = rospy.Time.now()
        marker.ns = "basic_shapes"
        marker.id = 0

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        quaternion = transformations.quaternion_from_euler(box[3], box[4], box[5])
        marker.pose.position.x = box[0]
        marker.pose.position.y = box[1]
        marker.pose.position.z = box[2]
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = box[-3]
        marker.scale.y = box[-2]
        marker.scale.z = box[-1]
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration()
        return marker

def publish_marker_sphere(box, r=1.0, g=0.423529411765 , b= 0.0392156862745):
        marker = Marker()
        marker.header.frame_id = "base_link" #"odom" #map
        marker.header.stamp = rospy.Time.now()
        marker.ns = "basic_shapes"
        marker.id = 0

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        quaternion = transformations.quaternion_from_euler(box[3], box[4], box[5])
        marker.pose.position.x = box[0]
        marker.pose.position.y = box[1]
        marker.pose.position.z = box[2]
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = box[-3]
        marker.scale.y = box[-2]
        marker.scale.z = box[-1]
        marker.color.r = r
        marker.color.g = g  
        marker.color.b = b
        marker.color.a = 1

        marker.lifetime = rospy.Duration()
        return marker