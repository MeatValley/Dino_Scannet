from json import load
from turtle import width
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import myutils.point_cloud as pc
import myutils.parse as parse
from datasets.scannet_dataset import ScanNetDataset
import matplotlib.pyplot as plt
import cv2
from feature_extractor.DINO_utils import get_feature_dictonary, intra_distance
import open3d as o3d
from clustering.clustering_algorithms import test_with_Kmean
from features.get_feature_from_file import get_feature_from_file

def test_load_point_cloud_from_scannet(file):
    """ Save a point cloud to a file.
    
        Args:
            point_cloud: o3d.geometry.PointCloud
        
        Returns:
            None
    """    
    config = parse.get_cfg_node(file)
    loaded_pc = pc.load_point_cloud(config.data.point_cloud.path)
    pc.show_point_cloud(loaded_pc)

def plot_sample_from_dataset(sample, j):
    print('[ploting...]')
    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(sample["image"][j]) #img with 224x224
    plt.subplot(1, 5, 2)
    plt.imshow(sample["proj3dto2d"][j]) #the points that came from the point cloud
    plt.subplot(1, 5, 3)
    plt.imshow(sample["image"][j]-sample["proj3dto2d"][j])
    plt.subplot(1, 5, 4)
    plt.imshow(sample["proj3dto2d_depth"][j]) # depth 
    plt.subplot(1, 5, 5)
    plt.imshow(sample["depth_map"][j])
    plt.show()

def plot_proximity_betweeen_feature(dictionary, dataset):
    #graphics and dot products
        dist_array_nb_patches = [len(dictionary[j]) for j in dictionary.keys()] #knows how many features each point have
        dist_array_nb_patches = np.asarray(dist_array_nb_patches).astype(np.int16)
        mean_nb_patches = dist_array_nb_patches.mean() #make the mean of the values (dictionary with means)
        std_nb_patches = dist_array_nb_patches.std()        


        # print("intra distances")
        dist = intra_distance(dictionary)
        dist_array = np.asarray(list(filter(lambda x: x!=1.0, list(dist.values())))).astype(np.float32)
        mean = dist_array.mean()
        std = dist_array.std()

        # print('the mean is: ', mean)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.hist(dist_array, bins=50)
        plt.gca().set(title='mean : ' + str(mean) + " | std : " + str(std), ylabel='Average normalized scalar product of features form several images');
        plt.subplot(1, 3, 3)
        plt.hist(dist_array_nb_patches, bins=int(dist_array_nb_patches.max()))
        plt.gca().set(title='mean : ' + str(mean_nb_patches) + " | std : " + str(std_nb_patches), ylabel='number of patches per 3d point');
        plt.show()
        dataset.point_cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # print("normals: ", np.asarray(dataset.point_cloud.normals))

with torch.no_grad():
    def test_dataset(file, number_images = 3, save_pc = False, save_features = False, run_DINO = True, K=5):
        print('[test dataset starting...]')
        
        config = parse.get_cfg_node(file)

        if run_DINO:
            print('[running DINO to get features...]')
            dataset = ScanNetDataset(config.data.path, config.data.point_cloud_name)

            dictionary = {}
            ind = number_images
            
            print("[getting a sample: ]")
            for i, sample in enumerate(iter(dataset)): #enumerate is just for i to be a counter
                if i == ind+1: break
                for j in range(len(sample["correspondances"])): #for all five crops
                #     plt.figure()
                #     plt.subplot(1, 5, 1)
                #     plt.imshow(sample["image"][j])
                #     plt.subplot(1, 5, 2)
                #     # plt.imshow(cv2.dilate(sample["proj3dto2d"], np.ones((2, 2), 'uint8'), iterations=4))
                #     plt.imshow(sample["proj3dto2d_depth"][j])
                #     plt.subplot(1, 5, 3)
                #     # plt.imshow(np.where(sample["proj3dto2d"] != 0, sample["proj3dto2d"], sample["image"]))
                #     plt.imshow(sample["image"][j]-sample["proj3dto2d"][j])
                #     # plt.imshow(sample["image"]-sample["proj3dto2d"])
                #     plt.subplot(1, 5, 4)
                #     plt.imshow(sample["proj3dto2d"][j])
                        
                #     plt.subplot(1, 5, 5)
                #     plt.imshow(sample["depth_map"][j])
                #     # plt.imshow(np.where(sample["proj3dto2d"] != 0, sample["proj3dto2d"], np.expand_dims(sample["depth_map"], axis=-1)))
                #     plt.show()
                    dictionary, patches = get_feature_dictonary(dictionary, sample["correspondances"][j], 
                                            sample["image_DINO_features"][j], dataset)    
                    # print('dictionary is: ', dictionary) 
                    # is a big amount of values      
                    # for each point in the image (key), a list (dictionary) of its features
                    # dictionary[key] is the list of features of the point key


            #now calculating the means/robust_mean
        
            dataset.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # print("normals : ", np.asarray(dataset.point_cloud.normals))
            keys_sel = {}
            raw_features = []
            finale_features = []
            points = []
            colors = []
            normals = []


            for j, point in enumerate(dataset.point_cloud_points): # for each point
                if j in dictionary.keys(): 

                    mean = sum(dictionary[j])/len(dictionary[j])
                    raw_features.append(mean)
                    #for each point we can have more than 1 feature stacked in dictionary, so make a mean

                    feat_temp = dictionary[j]
                    feat_temp_dist=[np.linalg.norm(robust_feature - mean) for robust_feature in dictionary[j]]
                    #distance of each feature to the mean

                    n_new_feat = max([int(0.6*len(feat_temp)), 1])
                    # we catch just the ones near
                    feat_temp_filtr = sorted(zip(feat_temp, feat_temp_dist), key= lambda x: x[1])[0:n_new_feat]
                    feat_temp_filtr = [x[0] for x in feat_temp_filtr]                    # just add the ones near

                    robust_mean = sum(feat_temp_filtr)/len(feat_temp_filtr)

                    #new mean (robust mean)
                    keys_sel[len(points)]=j
                    point = np.asarray(point)
                    normal = np.asarray(dataset.point_cloud.normals[j])
                    color = dataset.point_cloud_colors[j]
                        
                    # dino = mean
                    dino = robust_mean
                    points.append(point[None, :]) # [None, :] treats point as a single element
                    # x -> [x]
                    colors.append(color[None, :])
                    normals.append(normal[None, :])
                    #creates the finale feature

                    feature  = dino[None, :]
                    # feature  = color[None, :]
                    finale_features.append(feature)
                    if save_features: path = save_features_in_a_file(config, feature)
                    
            

            finale_features = np.concatenate(finale_features, axis=0)
            points = np.concatenate(points, axis=0) #axis = 0 treats each independent
            colors = np.concatenate(colors, axis=0)
            # finale_features = np.concatenate([np.asarray(dataset.point_cloud_points), np.asarray(dataset.point_cloud_colors)], axis=1)
            # finale_features_save = finale_features.copy()
            finale_features = (finale_features - finale_features.mean(axis=0))/finale_features.std(axis=0)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            # point_cloud.estimate_normals(
            #             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # print("normals : ", np.asarray(point_cloud.normals))
            
            print('[vizualiation of pc...]')
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(width = 1300, height = 700)
            vis.add_geometry(point_cloud)
            # pc.save_point_cloud(point_cloud,'configs\logs\point_clouds\segmentation_000img3.ply' )
            vis.run()  # user picks points
            vis.destroy_window()

            # test_with_Kmean(raw_features, points, config, range_for_K=4)
            test_with_Kmean(finale_features, points, config, range_for_K=K, save_pc= save_pc, index = number_images)

        else:
            print('[reading a previous saved file to get features...]')
            file_name = 'features_from_'+config.pipeline.feature_extractor.network+'_of_'+config.experiment.scene
            file_name = os.path.join('features', file_name)
            file_name+='.txt'
            get_feature_from_file(file_name)



def save_features_in_a_file(config, feature):
    print('[saving the features in a txt...]')
    file_name = 'features_from_'+config.pipeline.feature_extractor.network+'_of_'+config.experiment.scene
    file_name = os.path.join('features', file_name)
    file_name+='.txt'
    print(file_name)
    f = open(file_name, 'a')
    for f_384 in feature:
        f.write('[')
        for coord in f_384:
            f.write(str(coord)+'f\n')
        f.write(']')

    f.close()
    return file_name
