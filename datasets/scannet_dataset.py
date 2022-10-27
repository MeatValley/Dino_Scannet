import numpy as np
import os

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from myutils.image import load_image
from myutils.point_cloud import load_point_cloud
from geometry.camera import Camera
import feature_extractor.DINO as DINO
import matplotlib.pyplot as plt
import open3d as o3d

########################################################################################################################
#### Scannet Dataset
########################################################################################################################

class ScanNetDataset(Dataset):
    print('[creating a dataset...]')
    def __init__(self, root_dir, point_cloud_name, file_list = None, add_patch = None, scale = 0.25, features = "dino", device ="cpu"):
        """DATASET
            images from folder are 1296x972

            original image have 640x480
            then we reshape for 336x252
            then we crop top 5 224x224

            root_dir: path to data
            file_list: 
        """
        self.scale = scale
        self.features = features
        self.add_patch = add_patch
        self.data_transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) ) ] )
        point_cloud_path = os.path.join(root_dir, point_cloud_name)

        if file_list is not None:
            print("[file_list is not none]")
            self.split = file_list.split('/')[-1].split('.')[0]
            with open(file_list, "r") as f:
                self.color_split = f.readlines()         
        
        else:
            #get all images in the folder stream from "color" and with "depth"

            #list of color images
            image_dir = os.path.join(root_dir, 'stream', 'color') #join the str in a path
            self.color_split = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                          if os.path.isfile(os.path.join(image_dir, f)) and 
                          (f.endswith('.png') or f.endswith('.jpg') 
                           or f.endswith('.jpeg'))]
            
            #list of detph images
            depth_dir = os.path.join(root_dir, 'stream', 'depth')
            self.depth_split = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir)]

            #list of poses matrix
            pose_dir = os.path.join(root_dir, 'stream', 'pose')
            self.pose_split = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir) 
                          if os.path.isfile(os.path.join(pose_dir, f)) and 
                          f.endswith('.txt')]
            # print(self.pose_split)

            
            self.root_dir = root_dir

            #K for color img
            intrinsic_path = os.path.join(root_dir, 'stream', 'intrinsics_color.txt')
            self.intrinsic = self._get_intrinsics(intrinsic_path)
            # print(self.intrinsic)

            #K for depth img
            intrinsic_depth_path = os.path.join(root_dir, 'stream', 'intrinsics_depth.txt')
            self.intrinsic_depth = self._get_intrinsics(intrinsic_depth_path)


            #translation correction between the 2 cameras

            colortodepth_path = os.path.join(root_dir, os.path.basename(os.path.basename(root_dir))+'.txt')

            self.Tcolortodepth = self._get_colortodepth(colortodepth_path)
            
            #point_cloud in the data
            self.point_cloud = load_point_cloud(point_cloud_path)
            self.point_cloud_points = np.asarray(self.point_cloud.points)
            self.point_cloud_colors = np.asarray(self.point_cloud.colors)
            
            self.device = torch.device(device)
            self.dino_model, self.patch_size = DINO.get_model('vits8')
            self.dino_model = self.dino_model.to(self.device)

            self.preprocess = transforms.Compose([
                                transforms.Resize(252, interpolation= transforms.InterpolationMode.BICUBIC),
                                transforms.FiveCrop(224),
                            ])
            self.preprocess2 = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

########################################################################################################################

    @staticmethod
    def _get_pose(pose_file):
        return np.loadtxt(pose_file)

# ########################################################################################################################                            
    def __len__(self):
        """Dataset length"""
        return len(self.color_split)

########################################################################################################################
    @staticmethod
    def _get_intrinsics(intrinsic_file):
        """Get intrinsics from the calib_data dictionary."""
        return np.loadtxt(intrinsic_file)

    @staticmethod
    def _get_colortodepth(colortodepth_file):
        """Get color to depth from the calib_data dictionary."""
        # print("colortodepth_path: ", colortodepth_file)
        with open(colortodepth_file, "r") as f:
            colortodepth = f.readlines()
        colortodepth = colortodepth[2].split(" ")[2::]
        colortodepth = np.array([[float(colortodepth[4*i+j]) for j in range(4)] for i in range(4)], dtype=np.float32)
        return colortodepth
########################################################################################################################

    def __getitem__(self, index):
        print(f'[getting an sample for the dataset with {index} index... ]')
        """Get dataset sample given an index."""
        image = load_image(self.color_split[index]) # at this point size=1296x968 
        depth = load_image(self.depth_split[index]) 
        #dimensions of the original_image
        dims_image = tuple(np.asarray(image).shape[0:2][::-1])
        dims_depth = tuple(np.asarray(depth).shape[0:2][::-1])
        # print(dims_image)
        # print(dims_depth) # depth has size 640x480


        image = image.crop((20,20,1292-20,968-20))
        # print('img shape:       ', image) #size=1252x928


        original_image = image
        image_5crops = self.preprocess(image)
        image_res = image.resize((int(1292 * (252/968)), 252))
        width, height = image_res.size   # Get dimensions
        

        depth = np.asarray(depth)/1000 #[0,0, ..., 0]
        pose = self._get_pose(self.pose_split[index]) #get the pose of the current image

        H, W = 224, 224 #size for dino
        new_width = W
        new_height = H

        crops = [(0, 0, new_width, new_height),
                 (width - new_width, 0, width, new_height),
                 (0, height - new_height, new_width, height),
                 (width - new_width , height - new_height, width, height),
                 ((width - new_width)/2, (height - new_height)/2, (width + new_width)/2, (height + new_height)/2)
                ]

        image_crop_tot = []
        projection3dto2d_tot = []
        pose_tot = []
        pc_im_correspondances_tot = []
        image_DINO_features_tot = []
        depth_tot = []
        projection3dto2d_depth_tot = []
        features_interpolation_tot = []

        for n, image_ts in enumerate(list(image_5crops)):
            print(f'[doing the process to the {n} crop..]')
            image_ts = self.preprocess2(image_ts)

            (left, top, right, bottom) = crops[n]
 
            image_crop = image_res.crop((left, top, right, bottom)) #image with 224, 224
            image_DINO_features_ts = DINO.get_DINO_features(self.dino_model.to('cpu'), 
                                                    image_ts) 
            #each img is 224x224, we have patchs of size 8, so we have 28x28 pathces
            #we get 784 vector (28*28) with 384 values of features (ts)

            #we want to have to reshape it to become a 28,28 for 384, each patch has a 384 vector
            image_DINO_features = image_DINO_features_ts.detach().cpu().numpy().reshape(W//self.patch_size, H//self.patch_size, -1)
            #img_dino_features is a (28,28,384) which is a feature vector 384 for each 28x28 patch of the img ((224x224)/8) 

            cam = Camera(K=self.intrinsic, dimensions=dims_image, Tcw=np.linalg.inv(pose)).scaled(252./968.).crop((left, top, right, bottom))
            #TCW is just the pose
            #cam need to be reshaped to the crop imgs

            cam_depth = Camera(K=self.intrinsic_depth, dimensions=dims_depth, Tcw=self.Tcolortodepth@np.linalg.inv(pose))
            #TCW is the translation of color/depth and the pose

            coord_depth_000000 = np.asarray([[i, j] for i in range(depth.shape[0]) for j in range(depth.shape[1])])
            #the coord of each pixel
            value_depth_000000 = np.asarray([[depth[i,j]] for i in range(depth.shape[0]) for j in range(depth.shape[1])])
            #its values
            point_cloud_000000_np = cam_depth.get_point_cloud(coord_depth_000000, value_depth_000000)  #(x,y,z, 640*480)

            projection3dto2d_depth = cam.project_on_image(point_cloud_000000_np, value_depth_000000) #(224,224,1) for each pixel of the img that goes dino, its depth

            projection3dto2d, pc_im_correspondances = cam.project_on_image(self.point_cloud_points, self.point_cloud_colors, 
                                                                        projection3dto2d_depth, corres=True, eps=0.04)
             
            features_interpolation = None


            pose_tot.append(pose)
            image_crop_tot.append(np.asarray(image_crop)/255.0)
            depth_tot.append(depth) #depth is the Z coordinate (1 value for each 640x480 original pixel)
            image_DINO_features_tot.append(image_DINO_features) 
            features_interpolation_tot.append(features_interpolation) #use more than 1 feature, in this case not yet
            projection3dto2d_depth_tot.append(projection3dto2d_depth)
            projection3dto2d_tot.append(projection3dto2d)
            pc_im_correspondances_tot.append(pc_im_correspondances)

            
        
        sample = {
            "original_image": original_image,
            'pose': pose_tot, #all TCW
            'image': image_crop_tot, #all img with 224x224
            "image_DINO_features": image_DINO_features_tot, #all (28,28,384) tuples of patch & feature
            "depth_map" : depth_tot, #each z coordinate
            'proj3dto2d_depth': projection3dto2d_depth_tot,
            "feature_interpolation": features_interpolation_tot,
            'proj3dto2d': projection3dto2d_tot,
            'correspondances' : pc_im_correspondances_tot,


        }

        return sample