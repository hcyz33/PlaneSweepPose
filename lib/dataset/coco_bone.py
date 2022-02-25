import torch

coco_joints_def = {
    0: 'nose',
    1: 'Leye', 2: 'Reye',
    3: 'Lear', 4: 'Rear',
    5: 'Lsho', 6: 'Rsho',
    7: 'Lelb', 8: 'Relb',
    9: 'Lwri', 10: 'Rwri',
    11: 'Lhip', 12: 'Rhip',
    13: 'Lkne', 14: 'Rkne',
    15: 'Lank', 16: 'Rank',
}
coco_bones_def = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # head
    [3, 5], [5, 7], [7, 9],  # left arm
    [4, 6], [6, 8], [8, 10],  # right arm
    [5, 11], [6, 12],  # trunk
    [11, 13], [13, 15],  # left leg
    [12, 14], [14,
     16],  # right leg
]

bone_pairs = [
    [10,5],[5,6],
    [11,8],[8,9],
    [10,12],[12,13],
    [11,14],[14,15]
]

def cal_angle(poses, angle):

    # score =  torch.zeros(batch_size, num_persons, num_joints, num_depth_levels,2)
    # caculate bone vector
    for i in range(len(bone_pairs)):
        bone_1 = poses[:,:,coco_bones_def[bone_pairs[i][0]][0],:] - poses[:,:,coco_bones_def[bone_pairs[i][0]][1],:]
        bone_2 = poses[:,:,coco_bones_def[bone_pairs[i][1]][0],:] - poses[:,:,coco_bones_def[bone_pairs[i][1]][1],:]
        angle[:,:,i] = torch.cosine_similarity(bone_1, bone_2, -1, 1e-8)
    return angle

def cal_bone_length(poses, bone_lengthes):

    # score =  torch.zeros(batch_size, num_persons, num_joints, num_depth_levels,2)
    # caculate bone vector
    for i in range(len(coco_bones_def)):
        bone_lengthes[:,:,i] = torch.sqrt(torch.sum((poses[:,:,coco_bones_def[i][0]] - poses[:,:,coco_bones_def[i][1]]) ** 2, dim=-1))
    return bone_lengthes