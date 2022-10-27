import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_cfg_node
import tests.test as test
import open3d as o3d



def get_bounding_box(point_cloud, color = [1,0,0]):
    bb = point_cloud.get_oriented_bounding_box()
    bb.color = color
    return bb

def show_bounding_box(point_cloud, color = [1,0,0]):
    bb = point_cloud.get_oriented_bounding_box()
    bb.color = color
    o3d.visualization.draw_geometries([point_cloud, bb], width =1400, height= 1000)

def reunion1():
    """isolate caussin"""
    default_pc = pc.load_point_cloud('configs\logs\point_clouds\segmentation_img3.ply')
    pc_list = pc.get_point_clouds_separeted_by_color(default_pc)
    colors = pc.get_point_cloud_colors(default_pc)
    color =  colors[3]
    almofada = pc.get_point_cloud_by_color(default_pc, color)
    bb=get_bounding_box(almofada)
    pc_list.append(bb)
    pc.show_point_clouds(pc_list)
    bounding_box_all_point_cloud(default_pc)

def reunion2():
    """see dino features"""
    for i in range(2,6):
        print(i)
        file = 'configs\logs\point_clouds\s0279_00'+str(i)+'mean0_2022-10-13-15-47_point_cloud_colors.ply'
    
        test_pc = pc.load_point_cloud(file)

        bounding_box_all_point_cloud(test_pc)

def bounding_box_all_point_cloud(point_cloud):
    
    pc_list = pc.get_point_clouds_separeted_by_color(point_cloud)
    colors = pc.get_point_cloud_colors(point_cloud)

    for color in colors:
        mono_pc = pc.get_point_cloud_by_color(point_cloud, color)
        bb = get_bounding_box(mono_pc)
        pc_list.append(bb)

    pc.show_point_clouds(pc_list)
    return pc_list

def reunion3(show_without_bb = False):
    default_pc = pc.load_point_cloud('configs\logs\point_clouds\segmentation_img3.ply')
    k9 = pc.load_point_cloud('configs\logs\point_clouds\s0279_0010mean0_2022-10-14-14-50_point_cloud_colors.ply')
    if show_bounding_box:
        pc.show_point_cloud(default_pc)
        pc.show_point_cloud(k9)
    bounding_box_all_point_cloud(k9)
    bounding_box_all_point_cloud(default_pc)

def get_one_bounding_box_by_color(point_cloud_complete, point_cloud_color):
    ply = pc.get_point_cloud_by_color(point_cloud_complete, point_cloud_color)
    bb = get_bounding_box(ply)
    return bb

def get_each_bounding_box(point_cloud):

    colors = pc.get_point_cloud_colors(point_cloud)
    for color in colors:
        bb = get_one_bounding_box_by_color(point_cloud, color)
        pc.show_point_clouds([point_cloud, bb])

def reunion4():
    default_pc = pc.load_point_cloud('configs\logs\point_clouds\segmentation_img3.ply')
    k9 = pc.load_point_cloud('configs\logs\point_clouds\s0279_0010mean0_2022-10-14-14-50_point_cloud_colors.ply')
    get_each_bounding_box(default_pc)
    get_each_bounding_box(k9)

def reunion5():
    for i in range(2,8):
        path = 'configs\logs\point_clouds\s0321_00'+str(i)+'mean0_2022-10-14-16-31_point_cloud_colors.ply'
        pc3 = pc.load_point_cloud(path)
        get_each_bounding_box(pc3)
        # pc.show_point_cloud(pc3)


if __name__ == "__main__":
    loaded_pc = pc.load_point_cloud('configs\logs\point_clouds\Default_2022-10-11-15-57_point_cloud_colors.ply')

    pc3 = pc.load_point_cloud('configs\logs\point_clouds\s0321_002mean0_2022-10-14-16-31_point_cloud_colors.ply')
    default_pc0279 = pc.load_point_cloud('configs\logs\point_clouds\segmentation_img3.ply')
    default_pc0321 = pc.load_point_cloud('configs\logs\point_clouds\segmentation_321img3.ply')
    default_pc0000 = pc.load_point_cloud('configs\logs\point_clouds\segmentation_000img3.ply')
    k9 = pc.load_point_cloud('configs\logs\point_clouds\s0279_0010mean0_2022-10-14-14-50_point_cloud_colors.ply')

    # reunion1()
    # reunion2()
    # reunion3(show_without_bb=False)
    # reunion4()
    # reunion5()
    # pc.show_point_cloud(default_pc0321)
    # pc.show_point_cloud(pc3)
    
    


