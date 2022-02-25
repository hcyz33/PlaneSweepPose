import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
# h36m骨架连接顺序，每个骨架三个维度，分别为：起始关节，终止关节，左右关节标识(1 left 0 right)
human36m_connectivity_dict = [[1,3,0],[1,0,0],[2,4,1],[2,0,0],[0,5,0],[0,6,1],[5,7,0],[7,9,1],[6,8,0],[8,10,1],[5,11,0],
                                [6,12,1],[11,12,0],[11,13,1],[13,15,0],[12,14,1],[14,16,0]]

coco_connectivity_dict = [[0, 1, 0], [0, 2, 1], [1, 3, 0], [2, 4, 1],  # head
    [3, 5, 0], [5, 7, 0], [7, 9, 0],  # left arm
    [4, 6, 1], [6, 8, 1], [8, 10, 1],  # right arm
    [5, 11, 0], [6, 12, 1],  # trunk
    [5, 6, 0], [11, 12, 0],  # trunk
    [11, 13, 0], [13, 15, 0],  # left leg
    [12, 14, 1], [14,
     16, 1],] # right leg
def draw3Dpose(pose_3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    for i in coco_connectivity_dict:
        z, x, y = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if i[2] else rcolor)

    RADIUS = 350 # space around the subject
    # xroot, yroot, zroot = pose_3d[5, 1], pose_3d[5, 2], pose_3d[5, 0]
    xroot, yroot, zroot = (pose_3d[11, 1] + pose_3d[12, 1])/2, (pose_3d[11, 2] + pose_3d[12, 2])/2, (pose_3d[11, 0] + pose_3d[12, 0])/2
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    # ax.set_xlim3d([-1000, 1000])
    # ax.set_zlim3d([-1000, 1000])
    # ax.set_ylim3d([-1000, 1000])

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

def plt_3d(specific_3d_skeleton):
    specific_3d_skeleton = specific_3d_skeleton[:,[2,0,1]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw3Dpose(specific_3d_skeleton, ax)
    # plt.xticks([])
    # plt.yticks([])
    # plt.zticks([])
    plt.show()

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 2, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 2, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 1)


def plt_2d(k2d_projected,img_name = "img", time = 10):
    k2d_projected = k2d_projected
    img_1 = np.ones([280, 380, 3],np.uint8)*100
    draw_pose(k2d_projected,img_1)
    img = cv2.resize(img_1,dsize=(960,540),fx=0.3,fy=0.3)
    cv2.imshow(img_name,img)
    cv2.waitKey(time)


def show_contrast_img(k2d_input, k2d_projected):
    
    k2d_input = k2d_input[:,:,:2].numpy()
    k2d_projected = k2d_projected.numpy()
    for i in range(500):
        img_1 = np.ones([1080, 1920, 3],np.uint8)*100
        img_2 = np.ones([1080, 1920, 3],np.uint8)*200
        draw_pose(k2d_input[i],img_1)
        draw_pose(k2d_projected[i],img_2)
        img = np.concatenate([img_1,img_2],axis=1)

        img = cv2.resize(img,dsize=(1920,540),fx=0.3,fy=0.3)
        cv2.imshow('img',img)
        cv2.waitKey(10)

