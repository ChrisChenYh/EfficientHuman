"""3DPW dataset."""
import copy
import json
import os
import torch
import cv2
import numpy as np
import pickle as pk
import torch.utils.data as data
from litehuman.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from litehuman.utils.pose_utils import pixel2cam, reconstruction_error, cam2pixel
from litehuman.utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam, VirtualmarkersTransform3DSMPLCam)
from pycocotools.coco import COCO
from litehuman.models.layers.smpl.SMPL import SMPL_layer



all_marker_vids = {'smpl': {'ARIEL': 411,
                            'BHEAD': 384,
                            'C7': 3470,
                            'CHIN': 3052,
                            'CLAV': 3171,
                            'FHEAD': 335,
                            'LAEL': 1655,
                            'LANK': 3327,
                            'LAOL': 1736,
                            'LBAK': 1812,
                            'LBCEP': 628,
                            'LBHD': 182,
                            'LBLLY': 1345,
                            'LBSH': 2940,
                            'LBTHI': 988,
                            'LBUM': 3116,
                            'LBUST': 3040,
                            'LBUSTLO': 1426,
                            'LBWT': 3122,
                            'LCHEECK': 239,
                            'LCHST': 595,
                            'LCLAV': 1298,
                            'LCLF': 1103,
                            'LEBHI': 2274,
                            'LEBHM': 2270,
                            'LEBHP': 2193,
                            'LEBHR': 2293,
                            'LEIDIP': 2295,
                            'LELB': 1666,
                            'LELBIN': 1725,
                            'LEMDIP': 2407,
                            'LEPDIP': 2635,
                            'LEPPIP': 2590,
                            'LEPTIP': 2674,
                            'LERDIP': 2518,
                            'LERPIP': 2478,
                            'LERTIP': 2557,
                            'LETMP': 2070,
                            'LETPIPIN': 2713,
                            'LETPIPOUT': 2711,
                            'LFHD': 0,
                            'LFIN': 2174,
                            'LFOOT': 3365,
                            'LFRM': 1568,
                            'LFRM2': 1741,
                            'LFRM2IN': 1953,
                            'LFRMIN': 1728,
                            'LFSH': 1317,
                            'LFTHI': 874,
                            'LFTHIIN': 1368,
                            'LFWT': 857,
                            'LHEE': 3387,
                            'LHEEI': 3432,
                            'LHPS': 2176,
                            'LHTS': 2134,
                            'LIDX1': 2204,
                            'LIDX2': 2283,
                            'LIDX3': 2320,
                            'LIWR': 2112,
                            'LKNE': 1053,
                            'LKNI': 1058,
                            'LMHAND': 2212,
                            'LMID1': 2389,
                            'LMID2': 2406,
                            'LMID3': 2446,
                            'LMT1': 3336,
                            'LMT5': 3346,
                            'LNECK': 298,
                            'LNWST': 1323,
                            'LOWR': 2108,
                            'LPNK1': 2628,
                            'LPNK2': 2634,
                            'LPNK3': 2674,
                            'LPRFWT': 2915,
                            'LRNG1': 2499,
                            'LRNG2': 2517,
                            'LRNG3': 2564,
                            'LRSTBEEF': 3314,
                            'LSCAP': 1252,
                            'LSHN': 1082,
                            'LSHNIN': 1153,
                            'LSHO': 1861,
                            'LSHOUP': 742,
                            'LTHI': 1454,
                            'LTHILO': 850,
                            'LTHM1': 2251,
                            'LTHM2': 2706,
                            'LTHM3': 2730,
                            'LTHM4': 2732,
                            'LTHMB': 2224,
                            'LTIB': 1112,
                            'LTIBIN': 1105,
                            'LTIP': 1100,
                            'LTOE': 3233,
                            'LUPA': 1443,
                            'LUPA2': 1315,
                            'LWPS': 1943,
                            'LWTS': 1922,
                            'MBLLY': 1769,
                            'MBWT': 3022,
                            'MFWT': 3503,
                            'MNECK': 3057,
                            'RAEL': 5087,
                            'RANK': 6728,
                            'RAOL': 5127,
                            'RBAK': 5273,
                            'RBCEP': 4116,
                            'RBHD': 3694,
                            'RBLLY': 4820,
                            'RBSH': 6399,
                            'RBTHI': 4476,
                            'RBUM': 6540,
                            'RBUST': 6488,
                            'RBUSTLO': 4899,
                            'RBWT': 6544,
                            'RCHEECK': 3749,
                            'RCHST': 4085,
                            'RCLAV': 4780,
                            'RCLF': 4589,
                            'RELB': 5135,
                            'RELBIN': 5194,
                            'RFHD': 3512,
                            'RFIN': 5635,
                            'RFOOT': 6765,
                            'RFRM': 5037,
                            'RFRM2': 5210,
                            'RFRM2IN': 5414,
                            'RFRMIN': 5197,
                            'RFSH': 4798,
                            'RFTHI': 4360,
                            'RFTHIIN': 4841,
                            'RFWT': 4343,
                            'RHEE': 6786,
                            'RHEEI': 6832,
                            'RHPS': 5525,
                            'RHTS': 5595,
                            'RIBHI': 5735,
                            'RIBHM': 5731,
                            'RIBHP': 5655,
                            'RIBHR': 5752,
                            'RIDX1': 5722,
                            'RIDX2': 5744,
                            'RIDX3': 5781,
                            'RIIDIP': 5757,
                            'RIIPIP': 5665,
                            'RIMDIP': 5869,
                            'RIMPIP': 5850,
                            'RIPDIP': 6097,
                            'RIPPIP': 6051,
                            'RIRDIP': 5980,
                            'RIRPIP': 5939,
                            'RITMP': 5531,
                            'RITPIPIN': 6174,
                            'RITPIPOUT': 6172,
                            'RITTIP': 6191,
                            'RIWR': 5573,
                            'RKNE': 4538,
                            'RKNI': 4544,
                            'RMHAND': 5674,
                            'RMID1': 5861,
                            'RMID2': 5867,
                            'RMID3': 5907,
                            'RMT1': 6736,
                            'RMT5': 6747,
                            'RNECK': 3810,
                            'RNWST': 4804,
                            'ROWR': 5568,
                            'RPNK1': 6089,
                            'RPNK2': 6095,
                            'RPNK3': 6135,
                            'RPRFWT': 6375,
                            'RRNG1': 5955,
                            'RRNG2': 5978,
                            'RRNG3': 6018,
                            'RRSTBEEF': 6682,
                            'RSCAP': 4735,
                            'RSHN': 4568,
                            'RSHNIN': 4638,
                            'RSHO': 5322,
                            'RSHOUP': 4230,
                            'RTHI': 4927,
                            'RTHILO': 4334,
                            'RTHM1': 5714,
                            'RTHM2': 6168,
                            'RTHM3': 6214,
                            'RTHM4': 6193,
                            'RTHMB': 5686,
                            'RTIB': 4598,
                            'RTIBIN': 4593,
                            'RTIP': 4585,
                            'RTOE': 6633,
                            'RUPA': 4918,
                            'RUPA2': 4794,
                            'RWPS': 5526,
                            'RWTS': 5690,
                            'SACR': 1783,
                            'STRN': 3506,
                            'T10': 3016,
                            'T8': 3508},
                   'smplx': {
                       "CHN1": 8747,
                       "CHN2": 9066,
                       "LEYE1": 1043,
                       "LEYE2": 919,
                       "REYE1": 2383,
                       "REYE2": 2311,
                       "MTH1": 9257,
                       "MTH2": 2813,
                       "MTH3": 8985,
                       "MTH4": 1693,
                       "MTH5": 1709,
                       "MTH6": 1802,
                       "MTH7": 8947,
                       "MTH8": 2905,
                       "RIDX1": 7611,
                       "RIDX2": 7633,
                       "RIDX3": 7667,
                       "RMID1": 7750,
                       "RMID2": 7756,
                       "RMID3": 7781,
                       "RPNK1": 7978,
                       "RPNK2": 7984,
                       "RPNK3": 8001,
                       "RRNG1": 7860,
                       "RRNG2": 7867,
                       "RRNG3": 7884,
                       "RTHM1": 7577,
                       "RTHM2": 7638,
                       "RTHM3": 8053,
                       "RTHM4": 8068,
                       "LIDX1": 4875,
                       "LIDX2": 4897,
                       "LIDX3": 4931,
                       "LMID1": 5014,
                       "LMID2": 5020,
                       "LMID3": 5045,
                       "LPNK1": 5242,
                       "LPNK2": 5250,
                       "LPNK3": 5268,
                       "LRNG1": 5124,
                       "LRNG2": 5131,
                       "LRNG3": 5149,
                       "LTHM1": 4683,
                       "LTHM2": 4902,
                       "LTHM3": 5321,
                       "LTHM4": 5363,
                       "REBRW1": 2178,
                       "REBRW2": 3154,
                       "REBRW4": 2566,
                       "LEBRW1": 673,
                       "LEBRW2": 2135,
                       "LEBRW4": 1429,
                       "RJAW1": 8775,
                       "RJAW4": 8743,
                       "LJAW1": 9030,
                       "LJAW4": 9046,
                       "LJAW6": 8750,
                       "CHIN3": 1863,
                       "CHIN4": 2946,
                       "RCHEEK3": 8823,
                       "RCHEEK4": 3116,
                       "RCHEEK5": 8817,
                       "LCHEEK3": 9179,
                       "LCHEEK4": 2081,
                       "LCHEEK5": 9168,
                       # 'LETPIPOUT': 5321,
                       'LETPIPIN': 5313,
                       'LETMP': 4840,
                       'LEIDIP': 4897,
                       'LEBHI': 4747,
                       'LEMDIP': 5020,
                       'LEBHM': 4828,
                       'LERTIP': 5151,
                       'LERDIP': 5131,
                       'LERPIP': 5114,
                       'LEBHR': 4789,
                       'LEPDIP': 5243,
                       'LEPPIP': 5232,
                       'LEBHP': 4676,
                       'RITPIPOUT': 8057,
                       'RITPIPIN': 8049,
                       'RITMP': 7581,
                       'RIIDIP': 7633,
                       'RIBHI': 7483,
                       'RIMDIP': 7756,
                       'RIBHM': 7564,
                       'RIRDIP': 7867,
                       'RIRPIP': 7850,
                       'RIBHR': 7525,
                       'RIPDIP': 7984,
                       'RIPPIP': 7968,
                       'RIBHP': 7412
                   }
                   }

all_smpl_markers = all_marker_vids['smpl']

# 67 markers
# 20 markers
ORANGE_MARKERS = ['LBAK', 'LSCAP', 'RSCAP', 'LELBIN', 'LNWST', 'RNWST', 'RELBIN',           # 6
                  'LTHI', 'RTHI', 'LKNI', 'RKNI', 'RFTHI', 'LFTHI', 'RFTHIIN', 'LFTHIIN',   # 14
                  'MBLLY', 'RBUST', 'LBUST', 'RCHEECK', 'LCHEECK']                          # 19   

# MAP
# LWRB -> LOWR  RWRB -> ROWR  LWRA -> LIWR  RWRA -> RIWR
# RSHIN -> RSHN LSHIN -> LSHN
YELLOW_MARKERS = ['RFHD', 'LFHD', 'RSHO', 'CLAV', 'LSHO', 'RUPA', 'LUPA', 'STRN',            # 7    (27)
                  'RELB', 'LELB', 'RFRM', 'LFRM', 'RFWT', 'LFWT', 'RIWR', 'LIWR',            # 15   (35)
                  'RTHMB', 'LTHMB', 'RFIN', 'LFIN', 'RTHI', 'LTHI', 'RKNE', 'LKNE',          # 23   (43)
                  'RSHN', 'LSHN', 'RANK', 'LANK', 'RMT1', 'LMT1', 'RMT5', 'LMT5',            # 31   (51)
                  'RRSTBEEF', 'LRSTBEEF', 'RTOE', 'LTOE', 'LHEE', 'RHEE', 'LBWT', 'RBWT',    # 39   (59)   
                  'T10', 'C7', 'LBHD', 'RBHD', 'RBAK', 'LOWR', 'ROWR']                       # 46   (66)                                          

MARKERS_INDEX_67 = ORANGE_MARKERS + YELLOW_MARKERS


class PW3DVM(data.Dataset):
    """ 3DPW dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/pw3d'
        Path to the PW3D dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']

    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 17
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    joints_name_14 = (
        'R_Ankle', 'R_Knee', 'R_Hip',           # 2
        'L_Hip', 'L_Knee', 'L_Ankle',           # 5
        'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
        'Neck', 'Head'
    )
    skeleton = (
        (1, 0), (2, 1), (3, 2),  # 2
        (4, 0), (5, 4), (6, 5),  # 5
        (7, 0), (8, 7),  # 7
        (9, 8), (10, 9),  # 9
        (11, 7), (12, 11), (13, 12),  # 12
        (14, 7), (15, 14), (16, 15),  # 15
    )

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/pw3d',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=True):
        self._cfg = cfg

        self._ann_file = os.path.join(root, 'json', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        self.bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape] # millimeter -> meter

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT

        self._loss_type = cfg.LOSS['TYPE']

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)
        self.classfier = cfg.MODEL.EXTRA.get('WITHCLASSFIER', False)

        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.lshoulder_idx_17 = self.joints_name_17.index('L_Shoulder')
        self.rshoulder_idx_17 = self.joints_name_17.index('R_Shoulder')
        self.root_idx_smpl = self.joints_name_24.index('pelvis')
        self.lshoulder_idx_24 = self.joints_name_24.index('left_shoulder')
        self.rshoulder_idx_24 = self.joints_name_24.index('right_shoulder')
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )       

        self._items, self._labels = self._lazy_load_json()

        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
            self.transformation = SimpleTransform3DSMPL(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type)
        elif cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d_cam':
            self.transformation = SimpleTransform3DSMPLCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type)
        elif cfg.MODEL.EXTRA.PRESET == 'virtualmarkers_smpl_3d_cam':
            self.transformation = VirtualmarkersTransform3DSMPLCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type)        


    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_smpl_vm_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_smpl_vm_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_smpl_vm_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')
        
        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from json annotation files into buffer."""

        items = []
        labels = []

        db = COCO(self._ann_file)
        cnt = 0

        # count = 0
        for aid in db.anns.keys():
            # count += 1
            # print(count)
            ann = db.anns[aid]

            img_id = ann['image_id']

            img = db.loadImgs(img_id)[0]
            width, height = img['width'], img['height']

            sequence_name = img['sequence']
            img_name = img['file_name']
            abs_path = os.path.join(
                self._root, 'imageFiles', sequence_name, img_name)

            beta = np.array(ann['smpl_param']['shape']).reshape(10)
            theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)

            x, y, w, h = ann['bbox']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)
            if xmin > xmax - 5 or ymin > ymax - 5:
                continue

            f = np.array(img['cam_param']['focal'], dtype=np.float32)
            c = np.array(img['cam_param']['princpt'], dtype=np.float32)

            joint_cam_17 = np.array(ann['h36m_joints'], dtype=np.float32).reshape(17, 3)
            # print(joint_cam_17[:1])
            joint_vis_17 = np.ones((17, 3))
            joint_img_17 = np.zeros((17, 3))
            # joint_img_17 = cam2pixel(joint_cam_17 * 1000.0, f, c)
            # joint_img_17[:, 2] /= 1000.0    # [466 839 3.36]
            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]
            
            joint_cam = np.array(ann['smpl_joint_cam'])
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)

            joint_img = np.array(ann['smpl_joint_img'], dtype=np.float32).reshape(24, 3)
            if joint_img.size == 24 * 3:
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :] = joint_img.reshape(24, 3)
            else:
                joint_img_29 = joint_img.reshape(29, 3)

            # 
            joint_img_z_root = joint_img_29[self.root_idx_smpl, 2]
            
            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_img_29[self.root_idx_smpl, 2]

            joint_vis_24 = np.ones((24, 3))
            joint_vis_29 = np.zeros((29, 3))
            joint_vis_29[:24, :] = joint_vis_24

            root_cam = joint_cam_29[self.root_idx_smpl]     # root_cam 0.xxx
            # print(root_cam - joint_cam_17[:1])
            # print(root_cam)
            # generate virtual markers
            vm_theta = theta.copy()
            vm_beta = beta.copy()
            vm_theta = np.reshape(np.array(vm_theta, dtype=np.float32), (1, 24, 3))
            vm_theta = torch.from_numpy(vm_theta)
            vm_beta = np.reshape(np.array(vm_beta, dtype=np.float32), (1, 10))
            vm_beta = torch.from_numpy(vm_beta)
            smpl_output = self.smpl(
                pose_axis_angle=vm_theta,
                betas=vm_beta,
                global_orient=None,
                transl=None,
                return_verts=True,
                pose2rot=True
            )
            gt_vertices = smpl_output.vertices.cpu().numpy()[0]
            gt_smpl_jts_cam = smpl_output.joints_from_verts.cpu().numpy()[0]
            # align mesh xyz to joints_29
            smpl_root = gt_smpl_jts_cam[:1]
            # align mesh xyz to smpl_root
            gt_vertices -= smpl_root

            # align mesh xyz to joints_29
            gt_vertices += joint_cam_29[:1]

            gt_markers_67 = np.array([gt_vertices[all_smpl_markers[key]] for key in MARKERS_INDEX_67])
            # print(gt_markers_67[0, :])

            markers_img_67 = cam2pixel(gt_markers_67*1000.0, f, c)
            markers_img_67[:, 2] = markers_img_67[:, 2]/1000.0 - joint_cam_29[self.root_idx_smpl, 2]
            markers_vis_67 = np.ones((67, 3))

            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': cnt,
                'img_path': abs_path,
                'img_name': img_name,
                'width': width,
                'height': height,
                'joint_img_17': joint_img_17,
                'joint_vis_17': joint_vis_17,
                'joint_cam_17': joint_cam_17,
                'joint_relative_17': joint_relative_17,
                'joint_img_29': joint_img_29,
                'joint_vis_29': joint_vis_29,
                'joint_cam_29': joint_cam_29,
                'markers_img_67': markers_img_67,
                'markers_vis_67': markers_vis_67,
                'markers_cam_67': gt_markers_67,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': f,
                'c': c
            })
            cnt += 1

        return items, labels

    @property
    def joint_pairs_17(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    @property
    def joint_pairs_24(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    @property
    def joint_pairs_29(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))

    @property
    def marker_pairs_67(self):
        return (
            (0, 64), (1, 2), (3, 6), (4, 5), (7, 8),
            (9, 10), (11,12),(13,14),(16,17),(18,19),
            (20,21), (22,24),(25,26),(28,29),(30,31),
            (32,33), (34,35),(36,37),(38,39),(40,41),
            (42,43), (44,45),(46,47),(48,49),(50,51),
            (52,53), (54,55),(56,57),(58,59),(62,63),
            (65,66)
        )

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 1), (2, 3), (4, 5), (7, 8), (9, 10), (11, 12))

    def evaluate_uvd_24(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam'].copy()
            gt_3d_kpt = gt['joint_cam_29'][:24, :].copy()

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd_jts'][:24, :].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[2] + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['img_path']

            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_kp = np.mean(error, axis=0) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000
        metric = 'MPJPE'

        eval_summary = f'UVD_24 error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)
        print(f'UVD_24 error per joint: {tot_err_kp}')

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_24(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_align = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam'].copy()
            gt_3d_kpt = gt['joint_cam_29'][:24, :].copy()

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_24'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # rigid alignment for PA MPJPE
            pred_3d_kpt_align = reconstruction_error(
                pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['img_path']

            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_align = np.mean(error_align) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000

        eval_summary = f'XYZ_24 PA-MPJPE: {tot_err_align:2f} | MPJPE: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_17(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_pa = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_x = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_y = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_z = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        # error for each sequence
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_relative_17']

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_17'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]

            # select eval 14 joints
            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['img_path']

            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_pa = np.mean(error_pa) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000

        eval_summary = f'XYZ_14 PA-MPJPE: {tot_err_pa:2f} | MPJPE: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err_pa
