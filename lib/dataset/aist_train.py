



from enum import Enum
from aist_plusplus.loader import AISTDataset
import os
import json
from cv2 import data
import numpy as np
from aniposelib import cameras
from collections import OrderedDict
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

import scipy.io as scio
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import _TEST_METRICS, squareform
from core.config import config, update_config, get_model_name
from utils.utils import load_checkpoint
from utils.transforms import project_pose, back_project_pose

import scipy.linalg as linalg
import dataset
import models
import cv2 as cv
class Mode(Enum):
    Train = 1
    Test = 2
    Demo = 3


logger = logging.getLogger(__name__)

class PreProcessAist:
    def __init__(self,data_path="/data/hcy/aist_plusplus_final/",seqence_name="gBR_sBM_c02_d04_mBR0_ch01") -> None:
        self.aist_dataset = AISTDataset(data_path)
        self.seq_name, view = AISTDataset.get_seq_name(seqence_name)
        # self.view_idx = AISTDataset.VIEWS.index(view)
        
    def get_camera(self):
        env_name = self.aist_dataset.mapping_seq2env[self.seq_name]
        file_path = os.path.join(self.aist_dataset.camera_dir, f'{env_name}.json')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'r') as f:
            params = json.load(f)
        cameras = []
        for param_dict in params:
            camera = {}
            # camera['R'] = cv.Rodrigues(np.array(param_dict['rotation']).ravel()[[2,0,1]])[0]
            # camera['T'] = np.array(param_dict['translation'])[[2,0,1]].reshape([3,1])
            camera['R'] = cv.Rodrigues(np.array(param_dict['rotation']).ravel())[0]
            # camera['R'] = cv.Rodrigues(np.array([3.0,0.0,0.0]).ravel())[0]
            camera['T'] = np.array(param_dict['translation']).reshape([3,1])*10
            camera['fx'] = np.array(param_dict['matrix'][0][0])
            camera['fy'] = np.array(param_dict['matrix'][1][1])
            camera['cx'] = np.array(param_dict['matrix'][0][2])
            camera['cy'] = np.array(param_dict['matrix'][1][2])
            camera['k'] = np.zeros([3,1])
            camera['p'] = np.zeros([2,1])
            cameras.append(camera)
        return cameras
    
    def get_keypoints_2d(self):
        keypoints2d, _, _ = AISTDataset.load_keypoint2d(
            self.aist_dataset.keypoint2d_dir, self.seq_name)
        return keypoints2d

    def get_keypoints_3d(self):
        keypoints3d = AISTDataset.load_keypoint3d(
            self.aist_dataset.keypoint3d_dir, self.seq_name)
        return keypoints3d



class AistTrain(torch.utils.data.Dataset):
    def __init__(self, cfg, image_set, is_train) -> None:
        
        self.image_set = image_set
        self.data_root = "/data/hcy/aist_plusplus_final/"

        # self.mode = mode
        this_dir = os.path.dirname(__file__)
        
        self.num_views = 3
        self.cam_list = list(range(self.num_views))

        self.image_height = 1080
        self.image_width = 1920
        # self.cameras = self._get_cam()
        # self.db = self._get_db()

        self.use_pred_confidence = cfg.TEST.USE_PRED_CONFIDENCE
        self.nms_threshold = cfg.TEST.NMS_THRESHOLD


        # get the whole 2d data
        self.k2d, self.cameras, self.k3d = self.get_aist()

        self.max_num_persons = 2
        self.num_joints_coco = 17
        self.num_joints = 17

        self.db = self._get_db()

        self.frame_range = range(len(self.k3d)//self.num_views)
    
    def is_nan(self, data):
        return np.isnan(data).any()


    def get_aist(self):
        # f = open(self.data_root + '/splits/pose_train.txt', "r",encoding='utf-8')
        if self.image_set == "validation":
            f = open(self.data_root + '/splits/pose_val.txt', "r",encoding='utf-8')
        elif self.image_set == "train":
            f = open(self.data_root + '/splits/pose_val.txt', "r",encoding='utf-8')

        movie_list = f.readlines()
        
        k2ds = []
        k3ds = []
        cams = []

        for movie_name in movie_list:
            process_aist = PreProcessAist(data_path=self.data_root,seqence_name=movie_name.strip())
            cam = process_aist.get_camera()
            k2d = process_aist.get_keypoints_2d()
            k3d = process_aist.get_keypoints_3d()
        # there are 9 cameras for one subjucet, but we only need three of them to tran
            
            for frame_id in range(len(k2d[0])):
            # for frame_id in range(100):
                if self.is_nan(k2d[:self.num_views,frame_id]) or self.is_nan(k3d[frame_id]):
                    continue
                for cam_id in range(self.num_views):
                    k2ds.append(k2d[cam_id][frame_id])
                    k3ds.append(k3d[frame_id])
                    cams.append(cam[cam_id])


        # return np.array(k2ds[:300]), np.array(cams[:300]), np.array(k3ds[:300]) * 10
        return np.array(k2ds), np.array(cams), np.array(k3ds) * 10


    def __len__(self):
        # TODO: the count of data is equals frame count when demo
        #       when for test or train the count need to be changed to self.k2d.shape[0] * self.k2d.shape[1]
        # if self.mode == Mode.Demo:
        #     return self.k2d.shape[1]
        # else:
        #     return self.k2d.shape[0]*self.k2d.shape[1]
        return self.k2d.shape[0]

    def __getitem__(self, idx):
        # if self.mode == Mode.Demo:
        #     # if mode is demo ,the idx is frame id
        #     frame_id = idx
        #     idx = frame_id * self.num_views
        # else:
        #     # if mode is not demo ,the idx is frame_id * num_views + view_id
        frame_id = idx // self.num_views
        # === obtain all camera views corresponding to the frame
        views = list(range(self.num_views))

        # === move the current view to the first as target view
        views[0],views[idx - frame_id * self.num_views] = views[idx - frame_id * self.num_views],views[0]

        kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta = [], [], [], [], [], []
        for view_id in views:
            k, pv, jv, pd, jd, m = self._get_single_view_item(frame_id * self.num_views + view_id)
            kpts.append(k)  # [Np, Nj, 2]
            pose_vis.append(pv)  # [Np]
            joint_vis.append(jv)  # [Np, Nj]
            pose_depths.append(pd)  # [Np]
            joint_depths.append(jd)  # [Np, Nj]
            meta.append(m)

        return kpts, pose_vis, joint_vis, pose_depths[0], joint_depths[0], meta

    def _get_single_view_item(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        kpts = np.zeros([self.max_num_persons, self.num_joints_coco, 2])       # [Np, Nj, 2]
        pose_vis = np.zeros([self.max_num_persons])                            # [Np]
        joint_vis = np.zeros([self.max_num_persons, self.num_joints_coco])     # [Np, Nj]
        pose_depths = np.zeros([self.max_num_persons])                         # [Np]
        joint_depths = np.zeros([self.max_num_persons, self.num_joints_coco])  # [Np, Nj]

        pred_pose2d = db_rec["pred_pose2d"]  # [Np_hrnet, Nj_coco, 2+1]
        nposes = pred_pose2d.shape[0]

        pose_3d = db_rec["joints_3d"]
        cam =  db_rec["camera"]

        for n in range(nposes):
            kpts[n] = pred_pose2d[n, :, :2]
            pose_vis[n] = 1
            if self.use_pred_confidence:
                joint_vis[n] = pred_pose2d[n, :, 2]
            else:
                joint_vis[n] = 1.0

            # get pose depth and joints depth:
            _, depth = project_pose(pose_3d[n],cam )
            pose_depths[n] = (depth[11] + depth[12]) / 2.0
            joint_depths[n] = depth - pose_depths[n]

        
        meta = {
            "image":"home",
            "image_height": self.image_height,
            "image_width": self.image_width,
            "num_persons": nposes,
            "joints_2d": kpts,
            "joints_2d_vis": joint_vis,
            "pose_depths": pose_depths,
            "joint_depths": joint_depths,
            "pose_vis": pose_vis,
            "camera": cam,
        }
        kpts = torch.as_tensor(kpts, dtype=torch.float)
        pose_vis = torch.as_tensor(pose_vis, dtype=torch.float)
        joint_vis = torch.as_tensor(joint_vis, dtype=torch.float)
        pose_depths = torch.as_tensor(pose_depths, dtype=torch.float)
        joint_depths = torch.as_tensor(joint_depths, dtype=torch.float)

        return kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta

    def _get_db(self):
        db = []

        # get aist data k2d k3d cam
        k2d, cam, k3d = self.get_aist()

        num_frames = len(k3d)

        all_depths = []

        for frame_id in range(num_frames):

            pose2d, depths = project_pose(k3d[frame_id], cam[frame_id])  # [Nj, 2], [Nj]
            all_depths.extend(depths.tolist())
            all_poses_2d_vis = [np.ones([self.num_joints])]
            all_poses_3d_vis = [np.ones([self.num_joints])]
            
            all_poses_2d = [pose2d]
            all_poses_3d = [k3d[frame_id]]
            preds = np.array([k2d[frame_id]])   # [Np, Nj_coco, 2+1]

            db.append({
                "image_path": "",
                "joints_3d": np.array(all_poses_3d),  # [Np, Nj, 3]
                "joints_3d_vis": np.array(all_poses_3d_vis),  # [Np, Nj] all one
                "joints_2d": np.array(all_poses_2d),  # [Np, Nj, 2]
                "joints_2d_vis": np.array(all_poses_2d_vis),  # [Np, Nj]
                "camera": cam[frame_id],
                "pred_pose2d": preds,  # [Np_hrnet, Nj_coco, 2+1]
            })

        return db

    def get_images(self, video_path):
        """
        1. get images from a video.
        2. get 2d points with hrnet 
        """
        pass

    def recontruct_3d(self, preds_depth, confs):
        """
        Args
            preds_depth: [N, Np, Nj]
        """
        # datafile = os.path.join(self.dataset_root, "actorsGT.mat")
        # data = scio.loadmat(datafile)
        # actor_3d = np.array(np.array(data["actor3D"].tolist()).tolist()).squeeze()  # [Np, Nf]

        # num_persons, _ = actor_3d.shape

        #TODO: get frame number
        frame_number = self.k3d.shape[0]
        
        pred_rescontructed = []
        
        for frame_id, frame_no in enumerate(range(frame_number)):
            pose3d_pool = []
            angle_pool = []
            for cam_id in range(self.num_views):
                view_id = frame_id * self.num_views + cam_id

                pred_pose2d = self.k2d[cam_id][frame_id].reshape([1,self.k2d[cam_id][frame_id].shape[0],-1])
                # pred_pose2d = self.db[view_id]["pred_pose2d"]  # [Np, Nj, 2+1]
                pred_depth = preds_depth[view_id].copy()  # [Np_max, Nj]
                pred_depth = pred_depth[:pred_pose2d.shape[0]]  # [Np, Nj]

                conf_depth = confs[view_id].copy()  # [Np_max, Nj]
                conf_depth = conf_depth[:pred_pose2d.shape[0]]  # [Np, Nj]

                conf_pose2d = pred_pose2d[:, :, 2]  # [Np, Nj]
                conf_pose2d = conf_pose2d * conf_depth

                # === back project [2D + depth estimation] to 3D pose
                pred_pose2d = pred_pose2d[:, :, :2].reshape(-1, 2)  # [Np * Nj, 2]
                pred_depth = pred_depth.reshape(-1)  # [Np * Nj]
                pred_coco = back_project_pose(pred_pose2d, pred_depth, self.cameras[cam_id])  # [Np * Nj, 3]
                pred_coco = pred_coco.reshape(-1, self.num_joints_coco, 3)  # [Np, Nj, 3]
                pred_coco = np.concatenate([pred_coco, conf_pose2d[:, :, np.newaxis]], axis=-1)  # [Np, Nj, 4]

                # === use the angle between the facing direction of each pose and the camera ray pointing towards the pose as the weight for fusion
                for pose in pred_coco:
                    pose3d_pool.append(pose)

                    lsh = pose[5, :3]
                    rsh = pose[6, :3]
                    lhip = pose[11, :3]
                    rhip = pose[12, :3]

                    msh = (lsh + rsh) / 2.0
                    mhip = (lhip + rhip) / 2.0

                    sh = rsh - lsh
                    spine = mhip - msh
                    person_dir = np.cross(sh, spine)

                    cam_loc = self.cameras[cam_id]["T"].flatten()
                    person_cam = msh - cam_loc

                    v1 = person_dir / np.linalg.norm(person_dir)
                    v2 = person_cam / np.linalg.norm(person_cam)

                    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi * 180.0
                    if angle > 90:
                        angle = 180 - angle
                    angle_pool.append(angle)

            # === fuse multiple views
            pose3d_pool = np.stack(pose3d_pool, axis=0)  # [N, Nj, 4]

            import dataset
            for i in range(len(self.cam_list)):
                dataset.plt_3d(pose3d_pool[i])


            angle_pool = np.array(angle_pool)  # [N]
            dist_matrix = np.expand_dims(pose3d_pool[:, :, :3], axis=1) - np.expand_dims(pose3d_pool[:, :, :3], axis=0)  # [N, N, Nj, 3]
            dist_matrix = np.sqrt(np.sum(dist_matrix ** 2, axis=-1))  # [N, N, Nj]
            dist_matrix = np.mean(dist_matrix, axis=-1)  # [N, N]

            dist_vector = squareform(dist_matrix)
            Z = linkage(dist_vector, 'single')
            labels = fcluster(Z, t=self.nms_threshold, criterion='distance')

            clusters = [[] for _ in range(labels.max())]
            for pid, label in enumerate(labels):
                clusters[label - 1].append(pid)

            final_pose3d_pool = []

            for cluster in clusters:
                if len(cluster) == 1:
                    final_pose3d_pool.append(pose3d_pool[cluster[0]])
                else:
                    all_pose3d = pose3d_pool[np.array(cluster)]  # [Nc, Nj, 4]
                    all_angle = angle_pool[np.array(cluster)]  # [Nc]

                    weights = 90 - all_angle
                    mean_pose3d = np.sum(all_pose3d[:, :, :3] * weights.reshape(-1, 1, 1), axis=0) / (np.sum(weights) + 1e-8)
                    final_pose3d_pool.append(mean_pose3d)
            
            pred_rescontructed.append(final_pose3d_pool)

        return pred_rescontructed

    def evaluate(self, preds, confs, recall_threshold=500):
        """_get_single_view_item
        Args
            preds: [N, Np, Nj]
        """
        # datafile = os.path.join(self.dataset_root, "actorsGT.mat")
        # data = scio.loadmat(datafile)
        # actor_3d = np.array(np.array(data["actor3D"].tolist()).tolist()).squeeze()  # [Np, Nf]

        num_persons = self.max_num_persons

        alpha = 0.5
        limbs = [ [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]

        total_gt = 0
        match_gt = 0
        correct_parts = np.zeros([num_persons])
        total_parts = np.zeros([num_persons])
        bone_correct_parts = np.zeros([num_persons, len(limbs)])

        for frame_id, frame_no in enumerate(self.frame_range):
            pose3d_pool = []
            angle_pool = []
            for cam_id in range(self.num_views):
                view_id = frame_id * self.num_views + cam_id

                pred_pose2d = self.db[view_id]["pred_pose2d"]  # [Np, Nj, 2+1]
                pred_depth = preds[view_id].copy()  # [Np_max, Nj]
                pred_depth = pred_depth[:pred_pose2d.shape[0]]  # [Np, Nj]

                conf_depth = confs[view_id].copy()  # [Np_max, Nj]
                conf_depth = conf_depth[:pred_pose2d.shape[0]]  # [Np, Nj]

                conf_pose2d = pred_pose2d[:, :, 2]  # [Np, Nj]
                conf_pose2d = conf_pose2d * conf_depth

                # === back project [2D + depth estimation] to 3D pose
                pred_pose2d = pred_pose2d[:, :, :2].reshape(-1, 2)  # [Np * Nj, 2]
                pred_depth = pred_depth.reshape(-1)  # [Np * Nj]
                pred_coco = back_project_pose(pred_pose2d, pred_depth, self.db[view_id]["camera"])  # [Np * Nj, 3]
                pred_coco = pred_coco.reshape(-1, self.num_joints_coco, 3)  # [Np, Nj, 3]
                pred_coco = np.concatenate([pred_coco, conf_pose2d[:, :, np.newaxis]], axis=-1)  # [Np, Nj, 4]

                # === use the angle between the facing direction of each pose and the camera ray pointing towards the pose as the weight for fusion
                for pose in pred_coco:
                    pose3d_pool.append(pose)

                    lsh = pose[5, :3]
                    rsh = pose[6, :3]
                    lhip = pose[11, :3]
                    rhip = pose[12, :3]

                    msh = (lsh + rsh) / 2.0
                    mhip = (lhip + rhip) / 2.0

                    sh = rsh - lsh
                    spine = mhip - msh
                    person_dir = np.cross(sh, spine)

                    cam_loc = self.db[view_id]["camera"]["T"].flatten()
                    person_cam = msh - cam_loc

                    v1 = person_dir / np.linalg.norm(person_dir)
                    v2 = person_cam / np.linalg.norm(person_cam)

                    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi * 180.0
                    if angle > 90:
                        angle = 180 - angle
                    angle_pool.append(angle)

            # === fuse multiple views
            pose3d_pool = np.stack(pose3d_pool, axis=0)  # [N, Nj, 4]
            angle_pool = np.array(angle_pool)  # [N]
            dist_matrix = np.expand_dims(pose3d_pool[:, :, :3], axis=1) - np.expand_dims(pose3d_pool[:, :, :3], axis=0)  # [N, N, Nj, 3]
            dist_matrix = np.sqrt(np.sum(dist_matrix ** 2, axis=-1))  # [N, N, Nj]
            dist_matrix = np.mean(dist_matrix, axis=-1)  # [N, N]

            dist_vector = squareform(dist_matrix)
            Z = linkage(dist_vector, 'single')
            labels = fcluster(Z, t=self.nms_threshold, criterion='distance')

            clusters = [[] for _ in range(labels.max())]
            for pid, label in enumerate(labels):
                clusters[label - 1].append(pid)

            final_pose3d_pool = []

            for cluster in clusters:
                if len(cluster) == 1:
                    final_pose3d_pool.append(pose3d_pool[cluster[0]][:, :3])
                else:
                    all_pose3d = pose3d_pool[np.array(cluster)]  # [Nc, Nj, 4]

                    import dataset
                    for i in range(len(cluster)):
                        print(i)
                        dataset.plt_3d(all_pose3d[i])
                          

                    all_angle = angle_pool[np.array(cluster)]  # [Nc]

                    weights = 90 - all_angle
                    mean_pose3d = np.sum(all_pose3d[:, :, :3] * weights.reshape(-1, 1, 1), axis=0) / (np.sum(weights) + 1e-8)
                    final_pose3d_pool.append(mean_pose3d)
            
            pred = np.array(final_pose3d_pool)[:,:,:3]
            
            # pred = np.stack([self.coco2campus3D(p[:, :3]) for p in final_pose3d_pool])  # [Np, Nj, 3]

            for person in range(num_persons):
                # gt = actor_3d[person][frame_no] * 1000.0  # [Nj, 3]
                gt = self.k3d[frame_id * self.num_views]
                if person > 0:
                # if gt.size == 0:
                    continue

                mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)  # [Np]
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0]] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1]] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1
                        bone_correct_parts[person, j] += 1
                pred_hip = (pred[min_n, 2] + pred[min_n, 3]) / 2.0
                gt_hip = (gt[2] + gt[3]) / 2.0
                total_parts[person] += 1
                error_s = np.linalg.norm(pred_hip - gt_hip)
                error_e = np.linalg.norm(pred[min_n, 12] - gt[12])
                limb_length = np.linalg.norm(gt_hip - gt[12])
                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person] += 1
                    bone_correct_parts[person, 9] += 1

        bone_group = OrderedDict(
            [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
             ('Lower arms', [4, 7]), ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        bone_person_pcp = OrderedDict()
        for k, v in bone_group.items():
            bone_person_pcp[k] = np.sum(bone_correct_parts[:, v], axis=-1) / (total_parts / 10 * len(v) + 1e-8)

        logger.info("==============================================\n"
                    "     | Actor 1 | Actor 2 | Actor 3 | Average |\n"
                    " PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500m: {recall:.4f}".format(
                        pcp_1=actor_pcp[0] * 100, pcp_2=actor_pcp[1] * 100, pcp_3=actor_pcp[1] * 100, pcp_avg=avg_pcp * 100, recall=match_gt / (total_gt + 1e-8)))
        for k, v in bone_person_pcp.items():
            logger.info("{:10s}: {:.2f}".format(k, np.mean(v)))

        return avg_pcp
