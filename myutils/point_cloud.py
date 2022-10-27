from json import load
from turtle import width
import numpy as np
import open3d as o3d
import os

FIX_COLORS = np.array([
    [0,1,0], 
    [0,0,1], 
    [1,1,0], 
    [1,0,1], 
    [1,1,0], 
    [0,1,1],
    [0.5,1,0],
    [0,0.5,1],
    [1,0,0.5],
    [1,1,0.5],
    [0.5,1,1]
    ])

#################################################################################### - basic pc
def generate_random_point_cloud (num_points, shift =0):
    """ Generate a random point cloud.
    
        Args:
            num_points: int
            shift: numpy array of shape (1, d)
        
        Returns:
            o3d.geometry.PointCloud
    """
    points = np.random.normal(loc = shift, scale =1., size = (num_points, 3))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def merge_point_clouds(list_point_clouds):
    """ Merge a list of point clouds.
    
        Args:
            point_clouds: list of o3d.geometry.PointCloud
        
        Returns:
            o3d.geometry.PointCloud
    """
    merged_point_cloud = list_point_clouds[0]
    for point_cloud in list_point_clouds[1:]:
        merged_point_cloud += point_cloud
    return merged_point_cloud

def generate_random_point_cloud_with_clusters(cluster_count, num_points=100):
    """ Generate a random point cloud with clusters.
    
        Args:
            cluster_count: int
            num_points: int
        
        Returns:
            o3d.geometry.PointCloud
    """
    current_centroid = np.array([0,0,0])
    point_cloud = generate_random_point_cloud(num_points, shift=current_centroid)
    
    for i in range(cluster_count-1):
        new_direction = np.random.normal(loc=0., scale=1., size=(1, 3))
        new_direction = new_direction/np.linalg.norm(new_direction)
        current_centroid = 15*(i+1)*new_direction + current_centroid
        point_cloud = point_cloud + generate_random_point_cloud(num_points, 
                                                    shift=current_centroid)
        
    return point_cloud

#################################################################################### - saving pc
def load_point_cloud(path):
    """ load point cloud from a file.
    
        Args:
            point_cloud: o3d.geometry.PointCloud
        
        Returns:
            o3d.geometry.PointCloud
    """
    point_cloud_loaded = o3d.io.read_point_cloud(path)
    return point_cloud_loaded

def save_point_cloud_yaml(point_cloud, config):
    """ Save a point cloud to a file.
    
        Args:
            point_cloud: o3d.geometry.PointCloud
            config: .yaml file with save/folder/
        
        Returns:
            None
    """
    path = os.path.join(config.save.folder, "point_clouds", config.experiment.name + "_" + str(config.experiment.time)+"_point_cloud.ply")
    print(path)
    o3d.io.write_point_cloud(path, point_cloud)

def save_point_cloud(point_cloud, path):
    """ Save a point cloud to the specified path.
    
    Args:
        point_cloud: o3d.geometry.PointCloud
        path: str
        
    Returns:
        None
    """
    o3d.io.write_point_cloud(path, point_cloud)

def save_point_cloud_with_labels(point_clouds_np, labels, config,  random_colors = False, n_img = 0, K=0):
    print('[saving pc with labels...]')
    """ Show a list of point clouds with their labels.
    
        Args:
            point_clouds_np: list of numpy arrays of shape (n, d)
            labels: list of numpy arrays of shape (n,)
            config: .yaml file with save/folder/
        
        Returns:
            None
    """
    colors = np.random.rand(len(labels),3)
    # print(colors)

    if random_colors:
        point_colors = [colors[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
    else:
        point_colors = [FIX_COLORS[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds_np)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    
    path = os.path.join(config.save.folder, "point_clouds", config.experiment.scene + str(K) + 'mean' + str(n_img) + "_" + str(config.experiment.time)+"_point_cloud_colors.ply")
    o3d.io.write_point_cloud(path, point_cloud)

#################################################################################### - show pc
def show_point_clouds(point_clouds):
    """ Show a list of point clouds.
    
        Args:
            point_clouds: list of o3d.geometry.PointCloud
        
        Returns:
            None
    """
    o3d.visualization.draw_geometries(point_clouds, width = 1500, height = 800)
    
def show_point_cloud(point_cloud,  window_name="Point Cloud vizualization", width = 1500, height = 800):
    """ Show a list of point clouds.
    
        Args:
            point_clouds: o3d.geometry.PointCloud
        
        Returns:
            None
    """

    point_clouds = [point_cloud]
    
    o3d.visualization.draw_geometries(point_clouds, window_name, width , height)

def show_point_clouds_with_labels(point_clouds_np, labels, random_colors = False):
    print('[showing point cloud with labels...]')
    """ Show a list of point clouds with their labels.
    
        Args:
            point_clouds_np: list of numpy arrays of shape (n, d)
            labels: list of numpy arrays of shape (n,), same of points in the same cluster
        
        Returns:
            o3d.geometry.PointCloud
    """

    colors = np.random.rand(len(labels), 3) #for each pixel
    # [0.15 0.85 0.32] normalzied colors in rgb
    


    if random_colors:
        point_colors = [colors[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
    else:
        point_colors = [FIX_COLORS[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
   
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds_np)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    
    show_point_cloud(point_cloud)
    return point_cloud

def show_point_cloud_separeted_by_color(point_cloud):
    """receive a point cloud and show all the clusterings"""
    colors = get_point_cloud_colors(point_cloud)

    for color in colors:
        pc = get_point_cloud_by_color(point_cloud, color)
        show_point_cloud(pc)

#################################################################################### - filtering pc
def get_point_cloud_by_color(point_cloud, color):
    """receive a pc and return a point cloud with the points with that color
    
    input: o3d.geometry.PointCloud, color
    output: o3d.geometry.PointCloud with that color
    
    """
    print(f'[filtering for color {color} ...]')
    sub_point_cloud_points = o3d.utility.Vector3dVector()
    sub_point_cloud_color = o3d.utility.Vector3dVector()

    for key, point in enumerate(point_cloud.points):
        if (point_cloud.colors[key] == color).all(): 
            sub_point_cloud_color.append(color)
            sub_point_cloud_points.append(point)


    sub_point_cloud = o3d.geometry.PointCloud()
    sub_point_cloud.points = sub_point_cloud_points
    sub_point_cloud.colors = sub_point_cloud_color
    return sub_point_cloud

def get_point_cloud_colors(point_cloud):
    """return a vector with the rgb of the colors presents in this point cloud"""
    colors = []
    put = True
    for key, point in enumerate(point_cloud.points):

        if key == 0: 
            colors.append(point_cloud.colors[key])

        else:
            for color in colors:
                if (color == point_cloud.colors[key]).all():
                    put = False

            if put: colors.append(point_cloud.colors[key])

            put=True
    
    return colors

def get_point_clouds_separeted_by_color(point_cloud):
    """receive a point cloud and return all the clusterings in a list of o3d.geometry.PointCloud """
    colors = get_point_cloud_colors(point_cloud)
    monocolor_point_clouds = []

    for color in colors:
        pc = get_point_cloud_by_color(point_cloud, color)
        monocolor_point_clouds.append(pc)
    
    return monocolor_point_clouds

if __name__ == "__main__":
    pc = generate_random_point_cloud_with_clusters(3)
    path = "./test_point_cloud_with_clusters.ply"
    save_point_cloud(pc, path)
    # save_point_cloud_with_labels()
    pc = load_point_cloud(path)
    pct = load_point_cloud('test_point_cloud_with_clusterskkkkkkkkkkk.ply')
    show_point_cloud(pct)
    show_point_cloud(pc)