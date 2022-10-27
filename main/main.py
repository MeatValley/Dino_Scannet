import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_cfg_node
import tests.test as test
from myutils.run import run


if __name__ == "__main__":

    print('[main starting ...]')
    args = parse_args()

    run(args.file, number_images = 0, save_features=False, save_pc=False, run_DINO=True, K=12)
