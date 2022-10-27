from json import load
from operator import truediv
from re import A
from turtle import width
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import myutils.point_cloud as pc
import myutils.parse as parse
# from datasets.scannet_dataset import ScanNetDataset
import matplotlib.pyplot as plt
import cv2
from feature_extractor.DINO_utils import get_feature_dictonary, intra_distance
import open3d as o3d
from clustering.clustering_algorithms import test_with_Kmean
from myutils.parse import parse_args, get_cfg_node


# rotation_from_original_to_point = np.matrix( '-0.12086600065231323  0.04516300931572914  -0.1749730259180069 0')
# camera_original_rotation = np.matrix('1.588232159614563 -0.003562330733984709 2.7595205307006836 0')
# print(rotation_from_original_to_point.T@camera_original_rotation)
# a = 3
# b= 4
# j = a
# a = 1000

# K = np.array([
# [863.9415283203125, 0.0, 540.0],
# [0.0, 863.9415283203125, 540.0],
# [0.0, 0.0, 1.0]])
# cam = np.array([-16.956472, 40.124409, 1.465389])#world
# RT = np.array([
# [-0.8501240015029907, 0.5265823006629944, 0.0005163096939213574, -35.54466247558594], 
# [-0.07724553346633911, -0.12373658269643784, -0.9893040060997009, 5.104760646820068], 
# [-0.5208860635757446, -0.8410709500312805, 0.14586757123470306, 24.701332092285156]
# ,[0,0,0,1]]) #go world->cam
# R = RT[0:3, :3]
# T =(-1)* (R@cam)



# P = np.array([-15.609, 39.505, 2.214])#world
# # print(P.size)

# P_cam = (R@P)+T
# P_pixel = K@P_cam
# z = P_pixel[2]
# P_pixel = P_pixel/z
# u = int(P_pixel[0])

# v = int(P_pixel[1])

# print(u,v)