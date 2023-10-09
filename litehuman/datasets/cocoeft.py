"""MS COCO Human keypoint dataset."""
import os
import pickle as pk
# import scipy.misc
import cv2
import joblib
import numpy as np
import torch
import torch.utils.data as data
from litehuman.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from litehuman.utils.pose_utils import cam2pixel, pixel2cam, reconstruction_error
from litehuman.utils.presets.simple_transform_3d_cam_eft import SimpleTransform3DCamEFT
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
import json
import copy

s_coco_2_smpl_jt = [
    -1, 11, 12,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

class COCO_EFT_3D(data.Dataset):
    """ COCO Person dataset.
    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/coco'
        Path to the ms coco dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
    """
    CLASSES = ['person']
    num_joints = 17
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    joints_name_coco = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                   'left_shoulder', 'right_shoulder',                           # 6
                   'left_elbow', 'right_elbow',                                 # 8
                   'left_wrist', 'right_wrist',                                 # 10
                   'left_hip', 'right_hip',                                     # 12
                   'left_knee', 'right_knee',                                   # 14
                   'left_ankle', 'right_ankle')                                 # 16
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_29 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    )
    joints_name_14 = (
        'R_Ankle', 'R_Knee', 'R_Hip',           # 2
        'L_Hip', 'L_Knee', 'L_Ankle',           # 5
        'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
        'Neck', 'Head'
    )
    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/coco',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT

        self._loss_type = cfg.LOSS['TYPE']

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', [2200, 2200, 2200])
        # millimeter -> meter
        self.bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]

        self.transformation = SimpleTransform3DCamEFT(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=self._occlusion,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=64,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg,
            loss_type=self._loss_type, scale_mult=1.25)
        self.root_idx_smpl = self.joints_name_29.index('pelvis')
        self.root_idx_17 = self.joints_name_17.index('Pelvis')

        # self.db = self.load_pt()
        self._items, self._labels = self._lazy_load_json()
        print(len(self._items))
        print(len(self._labels))

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(self._labels[idx]['img_id'])
        
        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def load_pt(self):
        db = joblib.load(self._ann_file + '_smpl_annot.pt', 'r')
        return db

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num

    def preprocess_pt_item(self, label, idx):

        # for k, v in label.items():
        #     print(k)
        beta = label['shape'].copy()
        theta = label['pose'].copy().reshape(24, 3, 3)
        theta = matrix_to_axis_angle(torch.from_numpy(theta)).numpy()
        # scalar
        smpl_weight = label['smpl_weight'].copy().reshape(-1)

        joint_cam_17 = label['xyz_17'].reshape((17, 3))
        joint_cam_17 = joint_cam_17 - joint_cam_17[0]
        joint_cam_29 = label['xyz_29'].reshape((29, 3))
        joint_cam_29 = joint_cam_29 - joint_cam_29[0]

        joint_img_17 = np.zeros((17, 3))
        joints_vis_17 = np.zeros((17, 3)) * smpl_weight
        joint_img_29 = np.zeros((29, 3))
        joints_vis_29 = np.ones((29, 3)) * smpl_weight
        joints_vis_xyz_29 = np.ones((29, 3)) * smpl_weight
        gt_joints = label['joints_3d']

        # if smpl_weight[0] < 0.5:
        if float(smpl_weight) < 0.5:
            for i in range(24):
                id1 = i
                id2 = s_coco_2_smpl_jt[i]
                if id2 >= 0:
                    joint_img_29[id1, :2] = gt_joints[id2, :2, 0].copy()
                    joints_vis_29[id1, :2] = gt_joints[id2, :2, 1].copy()
        else:
            uv_29 = label['uv_29']
            joint_img_29[:, :2] = uv_29
            joint_img_29[:, 2] = joint_cam_29[:, 2]

        twist_angle = label['twist_angle'].reshape(23)
        cos = np.cos(twist_angle)
        sin = np.sin(twist_angle)
        phi = np.stack((cos, sin), axis=1)
        phi_weight = np.ones_like(phi) * smpl_weight[0]

        flag = (twist_angle < -10)
        phi_weight[flag, :] = 0

        root_cam = joint_cam_29[0]

        f = np.array([1000.0, 1000.0])
        c = np.array([128.0, 128.0])

        return_label = {
            'bbox': label['bbox'],
            'img_id': idx,
            'img_path': label['img_path'],
            'img_name': label['img_path'],
            'joint_img_17': joint_img_17,
            'joint_vis_17': joints_vis_17,
            'joint_cam_17': joint_cam_17,
            'joint_relative_17': joint_cam_17,
            'joint_img_29': joint_img_29,
            'joint_vis_29': joints_vis_29,
            'joint_vis_xyz_29': joints_vis_xyz_29,
            'joint_cam_29': joint_cam_29,
            'twist_phi': phi,
            'twist_weight': phi_weight,
            'beta': beta,
            'theta': theta,
            'root_cam': root_cam,
            'f': f,
            'c': c,
            'smpl_weight': smpl_weight
        }

        return return_label

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_smpl_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot ...')
            with open(self._ann_file + '_smpl_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
        return items, labels
    
    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)
        # iterate through the annotations
        bbox_scale_list = []
        det_bbox_set = {}
        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v

            image_id = ann['image_id']
            width, height = ann['width'], ann['height']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(ann['bbox']), width, height)
            joint_cam_17 = np.array(ann['smpl_h36m_joints_3d'])
            joint_cam_29 = np.array(ann['smpl_joints_3d_29'])

            joint_img_17 = np.ones_like(joint_cam_17).astype(np.float64)
            joint_img_17[:, :2] = np.array(ann['smpl_h36m_joints_2d'])
            joint_img_17[:, 2] = joint_cam_17[:, 2] - joint_cam_17[self.root_idx_17, 2]
            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_vis_17 = np.ones((17, 3))
            joint_vis_29 = np.ones((29, 3))
            root_cam = joint_cam_29[0]

            joint_img_29 = np.ones_like(joint_cam_29).astype(np.float64)
            joint_img_29[:, :2] = np.array(ann['smpl_joints_2d_29'])
            joint_img_29[:, 2] = joint_cam_29[:, 2] - joint_cam_29[self.root_idx_smpl, 2]
            # print(joint_cam_17)
            # print(joint_img_29)
            # print(joint_cam_29)
            #  break
            beta = np.array(ann['parm_shape'])
            theta = np.array(ann['parm_pose'])

            angle = np.array(ann['twist_angle'])
            angle = np.squeeze(angle, 1)
            cos = np.cos(angle)
            sin = np.sin(angle)
            phi = np.stack((cos, sin), axis=1)
            # phi = np.squeeze(phi, 2)

            phi_weight = (angle > -10) * 1.0
            phi_weight = np.stack([phi_weight, phi_weight], axis=1)

            f = np.array([1000.0, 1000.0])
            c = np.array([128.0, 128.0])

            parm_cam = np.array(ann['parm_cam'])

            abs_path = os.path.join(self._root, 'train2014', ann['file_name'])

            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'width': width,
                'height': height,
                'joint_img_17': joint_img_17,
                'joint_vis_17': joint_vis_17,
                'joint_cam_17': joint_cam_17,
                'joint_relative_17': joint_relative_17,
                'joint_img_29': joint_img_29,
                'joint_vis_29': joint_vis_29,
                'joint_cam_29': joint_cam_29,
                'twist_phi': phi,
                'twist_weight': phi_weight,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': f,
                'c': c,
                'parm_cam': parm_cam
            })
        
        return items, labels

            