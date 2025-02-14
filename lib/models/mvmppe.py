import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


from utils.transforms import torch_back_project_pose, torch_project_pose, torch_back_project_pose_diff_depth_per_joint
from models.softargmax import SoftArgMax
from models.cnns import PoseCNN, JointCNN
from models.pose_regress_net import PoseRegressNet
from dataset.coco_bone import cal_angle, bone_pairs, cal_bone_length, coco_bones_def


class MultiViewMultiPersonPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dataset = cfg.DATASET.TEST_DATASET

        self.pose_min_depth = cfg.MULTI_PERSON.POSE_MIN_DEPTH
        self.pose_max_depth = cfg.MULTI_PERSON.POSE_MAX_DEPTH
        self.pose_num_depth_layers = cfg.MULTI_PERSON.POSE_NUM_DEPTH_LAYERS
        self.joint_min_depth = cfg.MULTI_PERSON.JOINT_MIN_DEPTH
        self.joint_max_depth = cfg.MULTI_PERSON.JOINT_MAX_DEPTH
        self.joint_num_depth_layers = cfg.MULTI_PERSON.JOINT_NUM_DEPTH_LAYERS
        self.pose_sigma = cfg.MULTI_PERSON.POSE_SIGMA
        self.joint_sigma = cfg.MULTI_PERSON.JOINT_SIGMA

        self.plabels = np.arange(self.pose_num_depth_layers) / (self.pose_num_depth_layers - 1) * (self.pose_max_depth - self.pose_min_depth) + self.pose_min_depth  # [D]
        self.jlabels = np.arange(self.joint_num_depth_layers) / (self.joint_num_depth_layers - 1) * (self.joint_max_depth - self.joint_min_depth) + self.joint_min_depth  # [RD]
        self.register_buffer("pose_depth_labels", torch.as_tensor(self.plabels, dtype=torch.float))
        self.register_buffer("joint_relative_depth_labels", torch.as_tensor(self.jlabels, dtype=torch.float))

        num_joints=cfg.NETWORK.NUM_JOINTS
        self.rough_3d_pose_cnn = PoseRegressNet(num_joints=num_joints*2, hidden_size=cfg.NETWORK.HIDDEN_SIZE_ROUGH, output_size=num_joints)
        self.pose_cnn = PoseCNN(num_joints=num_joints, num_bones=len(coco_bones_def), hidden_size=cfg.NETWORK.HIDDEN_SIZE, output_size=1)
        self.joint_cnn = JointCNN(num_joints=num_joints, num_bones=len(coco_bones_def), hidden_size=cfg.NETWORK.HIDDEN_SIZE, output_size=num_joints)
        self.softargmax_kernel_size = cfg.NETWORK.SOFTARGMAX_KERNEL_SIZE
        self.softargmax_net = SoftArgMax()

    


    def feature_extraction(self, poses_3d, poses_2d_ref, vis_target, vis_ref, meta_target, meta_ref, sigma, show=False):
        """
        Args
            poses_3d: [B, Npt, Nj, Nd, 3]
            poses_2d_ref: [B, Npr, Nj, 2]
            vis_target: [B, Npt, Nj]
            vis_ref: [B, Npr, Nj]
        Steps:
            1. project poses_3d to reference view
            2. search for the nearest pose from poses_2d_ref
            3. compute score
            4. compute visibility in the reference view (bounding)
            5. return per joint score and bounding (used for fusing multiple views)
        Returns
            score: [B, Npt, Nj, Nd]
            bounding: [B, Npt, Nj, Nd]
        """
        batch_size, num_persons, num_joints, num_depth_levels, _ = poses_3d.size()
        device = poses_3d.device

        cam_ref = meta_ref["camera"]

        # === project 3d pose to reference view
        poses_2d_target = torch_project_pose(poses_3d.reshape(batch_size, num_persons, num_joints * num_depth_levels, 3), cam_ref)  # [B, Npt, Nj * Nd, 2]
        poses_2d_target = poses_2d_target.reshape(batch_size, num_persons, num_joints, num_depth_levels, 2)  # [B, Npt, Nj, Nd, 2]


        # === form pose distance matrix between target and reference view
        pt = poses_2d_target.reshape(batch_size, num_persons, 1, num_joints, num_depth_levels, 2)  # [B, Npt, 1, Nj, Nd, 2]
        pr = poses_2d_ref.reshape(batch_size, 1, num_persons, num_joints, 1, 2)  # [B, 1, Npr, Nj, 1, 2]
        

        # todo: show poses 
        if show:
            import dataset
            for i in range(batch_size):
                # dataset.plt_2d(pr[i, 0, 0, :, 0, :].cpu().numpy(),time=4000)\
                dataset.plt_3d(poses_3d[i, 0, :, 5, :].cpu().numpy())
                for j in range(num_depth_levels):
                    pass
                    # dataset.plt_3d(poses_3d[i, 0, :, j, :].cpu().numpy())
                    # dataset.plt_2d(poses_2d_target[i, 0, :, j, :].cpu().numpy(),time=100)
                
        
        
        poses_dist = torch.sum((pt - pr) ** 2, dim=-1)  # [B, Npt, Npf, Nj, Nd]
        # === distance is weighted by joint vis of reference poses
        poses_dist = torch.sum(poses_dist * vis_ref.reshape(batch_size, 1, num_persons, num_joints, 1), dim=-2) / (torch.sum(vis_ref.reshape(batch_size, 1, num_persons, num_joints, 1), dim=-2) + 1e-8)  # [B, Npt, Npf, Nd]

        # === set distance to padding poses to a large value
        for b in range(batch_size):
            poses_dist[b, :, meta_ref["num_persons"][b]:, :] = 1e5

        # === obtain the nearest pose
        min_dist, min_matching = poses_dist.min(dim=-2)  # [B, Npt, Nd], [B, Npt, Nd]
        matched_poses_2d_ref = torch.gather(poses_2d_ref.unsqueeze(3).repeat(1, 1, 1, num_depth_levels, 1), dim=1, index=min_matching.reshape(batch_size, num_persons, 1, num_depth_levels, 1).repeat(1, 1, num_joints, 1, 2))  # [B, Npt, Nj, Nd, 2]
        matched_vis_ref = torch.gather(vis_ref.reshape(batch_size, num_persons, num_joints, 1, 1).repeat(1, 1, 1, num_depth_levels, 1),
                                       dim=1,
                                       index=min_matching.reshape(batch_size, num_persons, 1, num_depth_levels, 1).repeat(1, 1, num_joints, 1, 1))  # [B, Npt, Nj, Nd, 1]
        
        # === compute score for each target pose based on the distance to its respective matched reference pose
        matching_dist = torch.sum((poses_2d_target - matched_poses_2d_ref) ** 2, dim=-1)  # [B, Npt, Nj, Nd]
        vr = matched_vis_ref.reshape(batch_size, num_persons, num_joints, num_depth_levels)  # [B, Npt, Nj, Nd]
        if "panoptic" in self.dataset:
            score = torch.exp(-torch.sqrt(matching_dist) / sigma)  # [B, Npt, Nj, Nd]
        else:
            score = torch.exp(-matching_dist / (sigma ** 2))  # [B, Npt, Nj, Nd]
        
        # angle_ref = torch.zeros(batch_size, num_persons, len(bone_pairs)).to(device)
        # angle_target = torch.zeros(batch_size, num_persons, len(bone_pairs), num_depth_levels).to(device)

        # angle_ref = cal_angle(poses_2d_ref,angle_ref)
        # angle_target = cal_angle(poses_2d_target,angle_target)

        # score_angle = torch.exp(-torch.abs(angle_ref.reshape(*angle_ref.shape,1) - angle_target))

        # score = torch.cat([score,score_angle],dim=2)

        bone_length_ref = torch.zeros(batch_size, num_persons, len(coco_bones_def), num_depth_levels).to(device)
        bone_length_target = torch.zeros(batch_size, num_persons, len(coco_bones_def), num_depth_levels).to(device)

        bone_length_ref = cal_bone_length(matched_poses_2d_ref,bone_length_ref)
        bone_length_target = cal_bone_length(poses_2d_target,bone_length_target)

        score_bone_length = torch.exp(-torch.abs(bone_length_ref - bone_length_target)/5)

        temp = torch.cat([score,score_bone_length],dim=2)
        


        # === compute the visibility of each target joint in the reference view
        bounding = torch.zeros(batch_size, num_persons, num_joints, num_depth_levels)  # [B, Npt, Nj, Nd]
        bounding = bounding.to(device)
        for b in range(batch_size):
            image_width = meta_ref["image_width"][b]
            image_height = meta_ref["image_height"][b]
            bounding[b, :, :num_joints, :] = (poses_2d_target[b, :, :, :, 0] >= 0) & (poses_2d_target[b, :, :, :, 1] >= 0) & (poses_2d_target[b, :, :, :, 0] <= image_width - 1) & (poses_2d_target[b, :, :, :, 1] <= image_height - 1)
            # bounding[b, :, num_joints:, :] = 1
        # === incorporate reference joint visibility into the aggregation of scores
        bounding = bounding * vr
        # bounding[:, :, :num_joints, :]  = bounding[:, :, :num_joints, :]  * vr

        bounding2 = torch.ones(batch_size, num_persons, len(coco_bones_def), num_depth_levels).to(device) 
        bounding2 = bounding2 * vr[:,:,:1,:].repeat(1, 1, len(coco_bones_def), 1)
        return score, score_bone_length, bounding, bounding2

    def forward(self, kpts, pose_vis, joint_vis, gt_pose_depths, gt_joint_depths, meta):
        """
        kpts:            2D poses per view
                         list (view) of [B, Np, Nj, 2]
        pose_vis:        pose visibility per view
                         list (view) of [B, Np]
        joint_vis:       joint visibility per view
                         list (view) of [B, Np, Nj]
        gt_pose_depths:  pose depth in target view
                         [B, Np]
        gt_joint_depths: joint depth in target view
                         [B, Np, Nj]
        meta:            list (view) of dict
        """

        output = dict()

        num_views = len(kpts)
        batch_size, num_persons, num_joints, num_coordinate = kpts[0].size()

        cam_target = meta[0]["camera"]
        kpts_2d_target = kpts[0]  # [B, Np, Nj, 2]
        device = kpts_2d_target.device
        
        # === stage 0
        # get rough 3dpose of target 2d
        image_height = meta[0]['image_height'].reshape([batch_size,1,1,1]).to(device)
        image_width = meta[0]['image_width'].reshape([batch_size,1,1,1]).to(device)
        kpts_2d_target_cp = copy.deepcopy(kpts_2d_target)
        kpts_2d_target_cp[:,:,:,0] = torch.div(kpts_2d_target_cp[:,:,:,0],image_width[0])
        kpts_2d_target_cp[:,:,:,1] = torch.div(kpts_2d_target_cp[:,:,:,1],image_height[0])
        kpts_2d_target_cp = kpts_2d_target_cp.reshape([batch_size * num_persons, num_joints*num_coordinate, 1])
        rough_reletive_depth = self.rough_3d_pose_cnn(kpts_2d_target_cp) * 1000
        rough_reletive_depth = rough_reletive_depth.reshape([batch_size, num_persons, num_joints])
        # rough_reletive_depth = torch.zeros([batch_size, num_persons, num_joints]).to(device)


        # === stage 1
        kpts_3d_all_depth = []
        for depth_id, depth_label in enumerate(self.pose_depth_labels):
            # depth = depth_label.reshape(1).repeat(batch_size)  # [B]
            # depth = depth.to(device)
            depth = torch.ones([batch_size,num_persons,num_joints],device=device) * depth_label
            depth = depth.to(device)
            depth = depth + rough_reletive_depth.detach()
            
            # === back project target poses to 3D
            kpts_3d = torch_back_project_pose_diff_depth_per_joint(kpts_2d_target, depth, cam_target)  # [B, Np, Nj, 3]
            # kpts_3d = torch_back_project_pose(kpts_2d_target, depth, cam_target)  # [B, Np, Nj, 3]
            kpts_3d_all_depth.append(kpts_3d)

        kpts_3d_all_depth = torch.stack(kpts_3d_all_depth, dim=3)  # [B, Np, Nj, Nd, 3]

        # === extract scores from reference views
        scores = None
        boundings = None
        for rv in range(1, num_views):
            score, score2, bounding, bounding2 = self.feature_extraction(kpts_3d_all_depth, kpts[rv], joint_vis[0], joint_vis[rv], meta[0], meta[rv], self.pose_sigma, False)  # [B, Np, Nj, Nd], [B, Np, Nj, Nd]
            if scores is None:
                scores = score * bounding
                scores2 = score2 * bounding2
            else:
                scores += score * bounding
                scores2 += score2 * bounding2
            if boundings is None:
                boundings = bounding
                boundings2 = bounding2
            else:
                boundings += bounding
                boundings2 += bounding2

        pose_score_volume = scores / (boundings + 1e-8)  # [B, Np, Nj, Nd]

        output["pose_score_volume"] = pose_score_volume  # [B, Np, Nj, Nd]

        pose_score_volume = pose_score_volume.reshape(batch_size * num_persons, num_joints, len(self.pose_depth_labels))  # [B * Np, Nj, Nd]
        
        pose_score_volume2 = scores2 / (boundings2 + 1e-8)  # [B, Np, Nj, Nd]

        output["pose_score_volume2"] = pose_score_volume2  # [B, Np, Nj, Nd]

        pose_score_volume2 = pose_score_volume2.reshape(batch_size * num_persons, len(coco_bones_def), len(self.pose_depth_labels))  # [B * Np, Nj, Nd]
        
        
        pose_depth_volume = self.pose_cnn(pose_score_volume, pose_score_volume2)  # [B * Np, 1, Nd]
        pose_depth_volume = F.softmax(pose_depth_volume, dim=-1)  # [B * Np, 1, ~Nd]

        output["pose_depth_volume"] = pose_depth_volume.reshape(batch_size, num_persons, len(self.pose_depth_labels))  # [B, Np, Nd]

        pred_pose_indices = self.softargmax_net(pose_depth_volume, torch.as_tensor(np.arange(len(self.pose_depth_labels)), dtype=torch.float, device=device), kernel_size=self.softargmax_kernel_size)  # [B * Np, 1]
        pred_pose_indices = pred_pose_indices.reshape(batch_size, num_persons)  # [B, Np]
        pred_pose_depths = pred_pose_indices / (self.pose_num_depth_layers - 1) * (self.pose_max_depth - self.pose_min_depth) + self.pose_min_depth  # [B, Np]

        output["pred_pose_indices"] = pred_pose_indices  # [B, Np]
        output["pred_pose_depths"] = pred_pose_depths  # [B, Np]

        # === stage 2
        kpts_3d_all_depth = []
        for depth_id, depth_label in enumerate(self.joint_relative_depth_labels):
            if self.training:
                depth = depth_label.reshape(1, 1, 1).repeat(batch_size, num_persons, num_joints) + \
                    gt_pose_depths.reshape(batch_size, num_persons, 1).repeat(1, 1, num_joints)  + rough_reletive_depth.detach()  # [B, Np, Nj]
            else:
                depth = depth_label.reshape(1, 1, 1).repeat(batch_size, num_persons, num_joints) + \
                    pred_pose_depths.reshape(batch_size, num_persons, 1).repeat(1, 1, num_joints) + rough_reletive_depth.detach()  # [B, Np, Nj]

            # === back project target poses to 3D
            kpts_3d = torch_back_project_pose_diff_depth_per_joint(kpts_2d_target, depth, cam_target)  # [B, Np, Nj, 3]
            kpts_3d_all_depth.append(kpts_3d)

        kpts_3d_all_depth = torch.stack(kpts_3d_all_depth, dim=3)  # [B, Np, Nj, Nd, 3]

        # === extract scores from reference views
        scores = None
        boundings = None
        for rv in range(1, num_views):
            score, score2, bounding, bounding2 = self.feature_extraction(kpts_3d_all_depth, kpts[rv], joint_vis[0], joint_vis[rv], meta[0], meta[rv], self.joint_sigma,False)  # [B, Np, Nj, Nrd], [B, Np, Nj, Nrd]
            if scores is None:
                scores = score * bounding
                scores2 = score2 * bounding2
            else:
                scores += score * bounding
                scores2 += score2 * bounding2
            if boundings is None:
                boundings = bounding
                boundings2 = bounding2
            else:
                boundings += bounding
                boundings2 += bounding2

        joint_score_volume = scores / (boundings + 1e-8)  # [B, Np, Nj, Nrd]

        output["joint_score_volume"] = joint_score_volume  # [B, Np, Nj, Nrd]

        joint_score_volume = joint_score_volume.reshape(batch_size * num_persons, num_joints, len(self.joint_relative_depth_labels))  # [B * Np, Nj, Nrd]
        
        joint_score_volume2 = scores2 / (boundings2 + 1e-8)  # [B, Np, Nj, Nrd]

        output["joint_score_volume2"] = joint_score_volume2  # [B, Np, Nj, Nrd]

        joint_score_volume2 = joint_score_volume2.reshape(batch_size * num_persons, len(coco_bones_def), len(self.joint_relative_depth_labels))  # [B * Np, Nj, Nrd]

        joint_depth_volume = self.joint_cnn(joint_score_volume, joint_score_volume2)  # [B * Np, Nj, Nrd]
        joint_depth_volume = F.softmax(joint_depth_volume, dim=-1)  # [B * Np, Nj, ~Nrd]

        output["joint_depth_volume"] = joint_depth_volume.reshape(batch_size, num_persons, num_joints, len(self.joint_relative_depth_labels))  # [B, Np, Nj, Nrd]

        pred_joint_indices = self.softargmax_net(joint_depth_volume, torch.as_tensor(np.arange(len(self.joint_relative_depth_labels)), dtype=torch.float, device=device))  # [B * Np, Nj]
        pred_joint_indices = pred_joint_indices.reshape(batch_size, num_persons, num_joints)  # [B, Np, Nj]
        pred_joint_depths = pred_joint_indices / (self.joint_num_depth_layers - 1) * (self.joint_max_depth - self.joint_min_depth) + self.joint_min_depth  # [B, Np, Nj]
        pred_joint_depths = pred_joint_depths + rough_reletive_depth
        
        output["pred_joint_indices"] = pred_joint_indices  # [B, Np, Nj]
        output["pred_joint_depths"] = pred_joint_depths  # [B, Np, Nj]

        # === losses
        if self.training:
            loss_rough_pose = F.smooth_l1_loss(rough_reletive_depth * joint_vis[0], gt_joint_depths * joint_vis[0], reduction="sum") / (torch.sum(joint_vis[0]) + 1e-8)
            loss_pose = F.smooth_l1_loss(pred_pose_depths * pose_vis[0], gt_pose_depths * pose_vis[0], reduction="sum") / (torch.sum(pose_vis[0]) + 1e-8)
            loss_joint = F.smooth_l1_loss(pred_joint_depths * joint_vis[0], gt_joint_depths * joint_vis[0], reduction="sum") / (torch.sum(joint_vis[0]) + 1e-8)
            loss = {
                "rough_pose":loss_rough_pose,
                "pose": loss_pose,
                "joint": loss_joint,
                # "total": 0.5*loss_rough_pose + loss_pose + 2 * loss_joint,
                "total": 0.5*loss_rough_pose + loss_pose + loss_joint,
            }
        else:
            loss = None

        # === merge pose depth and joint depth
        merged_depths = pred_pose_depths.reshape(batch_size, num_persons, 1) + pred_joint_depths  # [B, Np, Nj]

        output["pred_depths"] = merged_depths  # [B, Np, Nj]

        return output, loss


def get_model(cfg):
    model = MultiViewMultiPersonPoseNet(cfg)
    return model
