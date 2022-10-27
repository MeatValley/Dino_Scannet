import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import cv2

from geometry.camera import Camera
import myutils.parse as parse
import myutils.point_cloud as pc
from myutils.parse import parse_args

def test_camera(file):
    config = parse.get_cfg_node(file)
    #random Matrix
    Tcw = np.array([
        [-0.955421, 0.119616, -0.269932, 2.65583], 
        [0.295248, 0.388339, -0.872939, 2.9816], 
        [0.000407581, -0.91372, -0.406343, 1.36865], 
        [0, 0, 0, 1]
        ]
    )
    
    Tcw2 = np.array([
        [-0.955421, 0.119616, -0.269932, 3.65583], 
        [0.295248, 0.388339, -0.872939, 0.9816], 
        [0.000407581, -0.91372, -0.406343, 0.36865], 
        [0, 0, 0, 1]
        ]
    )
    
    K = np.array([
        [1169.62, 0, 646.295, 0],
        [0, 1167.11, 489.927, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]
    )

    H=224
    W=224
    cam = Camera(K=K, Tcw=np.linalg.inv(Tcw), dimensions = (H,W)) #Tcw-1 is Twc
    # cam = cam.scaled(256./968, 256./1296)
    # print(cam.H)
    # cam = cam.crop((32,32,32,32))

    loaded_pc = pc.load_point_cloud(config.data.point_cloud.path)
    pc_np = np.asarray(loaded_pc.points) #Coordinates 
    color_np = np.asarray(loaded_pc.colors) # (N,3) array for RGB colors
    proj = cam.project(pc_np) 


    image = np.zeros((H, W, 4))
    c = 0
    print(proj)

    for i in range(len(pc_np)):
        if proj[i,0] >=0 and proj[i,1] >= 0:
            Z = proj[i, 2]
            if Z<0:
                continue
            if image[int((proj[i,1]) * H), int((proj[i,0]) * W), 3] == 0 or Z < image[int((1-proj[i,0]) * H), int(proj[i,1] * W), 3]:
                image[int((proj[i,1]) * H), int((proj[i,0]) * W), 0:3] = color_np[i]
                image[int((proj[i,1]) * H), int((proj[i,0]) * W), 3] = Z
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0:3])
    plt.subplot(1,2,2)
    plt.imshow(cv2.dilate(image[:,:,0:3], np.ones((2, 2), 'uint8'), iterations=1))
    plt.show()
    print(c)

if __name__ == "__main__":
    args = parse_args()
    test_camera(args.file)