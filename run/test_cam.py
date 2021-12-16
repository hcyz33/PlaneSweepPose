import os
import logging
import argparse
import pprint
from pathlib import Path

import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import copy

import _init_paths
import scipy.io as scio
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from utils.transforms import torch_back_project_pose, torch_project_pose
from core.config import config, update_config, get_model_name
from utils.utils import load_checkpoint
from utils.transforms import project_pose, back_project_pose

import dataset
import models

import cv2
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("-t", "--tag", help="time tag of checkpoint", required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args




CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def project(k3d, cam):
    cam['R'] = np.expand_dims(cam['R'],0)
    cam['T'] = np.expand_dims(cam['T'],0)
    cam['fx'] = np.expand_dims(cam['fx'],0)
    cam['fy'] = np.expand_dims(cam['fy'],0)
    cam['cx'] = np.expand_dims(cam['cx'],0)
    cam['cy'] = np.expand_dims(cam['cy'],0)
    cam['k'] = np.expand_dims(cam['k'],0)
    cam['p'] = np.expand_dims(cam['p'],0)
    poses_2d_target = torch_project_pose(k3d, cam)
    return poses_2d_target

def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]

    print("=> Loading data..")
    test_dataset = eval("dataset." + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, 9, dataset.Mode.Demo)

    cam_id = 5

    k3d = torch.unsqueeze(torch.as_tensor(test_dataset.k3d, dtype=torch.float), 0)
    cam = test_dataset.cameras
    k2d = torch.unsqueeze(torch.as_tensor(test_dataset.k2d, dtype=torch.float)[cam_id], 0) 

    # dataset.plt_3d(k3d[0,0]*10)
    # 
    poses_2d_target_0 = project(k3d, cam[cam_id])

    # poses_2d_target_1 = project(k3d, cam[1])

    # poses_2d_target_0[:,:,:,0] += 1200
    # poses_2d_target /= 2
    # poses_2d_target_1[:,:,:,0] -= 1000
    # poses_2d_target_1[:,:,:,1] -= 3500
    dataset.show_contrast_img(k2d[0], poses_2d_target_0[0])
    
    pass

if __name__ == "__main__":
    main()
