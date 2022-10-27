import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_cfg_node
import tests.test as test



def run(file, number_images = 3, save_pc = False, save_features = False, run_DINO = True, K=5):
    test.test_dataset(file, number_images=number_images, save_pc = save_pc, save_features=save_features, run_DINO=run_DINO, K=K)