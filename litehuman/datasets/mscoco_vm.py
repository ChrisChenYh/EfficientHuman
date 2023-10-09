"""MS COCO Human keypoint dataset."""
import copy
import os
import pickle as pk

import cv2
import numpy as np
import torch.utils.data as data
from litehuman.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from litehuman.utils.presets import SimpleTransform, SimpleTransformCam, VirtualmarkersSimpleTransformCam
from litehuman.utils.pose_utils import cam2pixel, pixel2cam, reconstruction_error
from pycocotools.coco import COCO
import json
import torch
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


class MscocoVM(data.Dataset):
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
    joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                   'left_shoulder', 'right_shoulder',                           # 6
                   'left_elbow', 'right_elbow',                                 # 8
                   'left_wrist', 'right_wrist',                                 # 10
                   'left_hip', 'right_hip',                                     # 12
                   'left_knee', 'right_knee',                                   # 14
                   'left_ankle', 'right_ankle')                                 # 16

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/coco',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=True):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._ann_path = os.path.join(root, 'annotations')
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

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
            self.transformation = SimpleTransform(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, dict_output=True)
        elif cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d_cam':
            self.transformation = SimpleTransformCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, dict_output=True, 
                bbox_3d_shape=self.bbox_3d_shape)
        elif cfg.MODEL.EXTRA.PRESET == 'virtualmarkers_smpl_3d_cam':
            self.transformation = VirtualmarkersSimpleTransformCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, dict_output=True, 
                bbox_3d_shape=self.bbox_3d_shape)

        self._items, self._labels = self._lazy_load_json()

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

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

    def _lazy_load_ann_file(self):
        if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _coco = COCO(self._ann_file)
            try:
                with open(self._ann_file + '.pkl', 'wb') as fid:
                    pk.dump(_coco, fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')
            return _coco

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_vm_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_vm_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_vm_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        _coco = self._lazy_load_ann_file()
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        with open(os.path.join(self._ann_path, 'coco_smplifyx_train.json')) as f:
            smplify_results = json.load(f)

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        count = 0
        for entry in _coco.loadImgs(image_ids):
            # count += 1
            # print(count)  118287
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self._root, dirname, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_coco, entry, smplify_results)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _check_load_keypoints(self, coco, entry, smplify_results):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # print(obj.keys()) id
            aid = obj['id']
            if str(aid) in smplify_results:
                # print(smplify_results[aid])
                # smpl_param, cam_param, fitted_2d_pose
                # pose shape trans
                smplify_result = smplify_results[str(aid)]
                # print(smplify_result['smpl_param'].keys())
                # print(1)
            else:
                continue
            # print(smplify_results.keys())
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            markers_3d = np.zeros((67, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            smplify_result['cam_param'].update({'R': np.eye(3).astype(np.float32), 't': np.zeros((3,)).astype(np.float32)})
            smpl_param = smplify_result['smpl_param'] 
            cam_param = smplify_result['cam_param']
            focal_l, center, Rot, transl = cam_param['focal'], cam_param['princpt'], cam_param['R'], cam_param['t']
            cam_param = {'focal': np.array(focal_l).astype(np.float32), 'center': np.array(center).astype(np.float32), 'Rot': np.array(Rot).astype(np.float32), 'transl': np.array(transl).astype(np.float32)}

            # generate virtual markers
            vm_beta = np.array(smpl_param['shape'])
            vm_beta = np.reshape(np.array(vm_beta, dtype=np.float32), (1, 10))
            vm_beta = torch.from_numpy(vm_beta)
            vm_theta = np.array(smpl_param['pose'])
            vm_theta = np.reshape(np.array(vm_theta, dtype=np.float32), (1, 24, 3))
            vm_theta = torch.from_numpy(vm_theta)
            vm_trans = np.array(smpl_param['trans'])
            vm_trans = np.reshape(np.array(vm_trans, dtype=np.float32), (1, 3))
            vm_trans = torch.from_numpy(vm_trans)

            smpl_output = self.smpl(
                pose_axis_angle=vm_theta,
                betas=vm_beta,
                global_orient=None,
                transl=vm_trans,
                return_verts=True,
                pose2rot=True
            )
            gt_vertices = smpl_output.vertices.cpu().numpy()[0]
            gt_smpl_jts_cam = smpl_output.joints_from_verts.cpu().numpy()[0]
            gt_smpl_jts_cam *= 1000.0
            gt_vertices *= 1000.0
            root_cam = gt_smpl_jts_cam[:1]

            # get mrks67_img
            gt_markers_67 = np.array([gt_vertices[all_smpl_markers[key]] for key in MARKERS_INDEX_67])
            f = cam_param['focal']
            c = cam_param['center']
            markers_img_67 = cam2pixel(gt_markers_67, f, c)
            # markers_img_67 = markers_img_67[:, :2]

            markers_3d[:, :, 0] = markers_img_67
            markers_3d[:, :, 1] = 1.0

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d,
                'markers_3d': markers_3d,
                'segmentation': obj['segmentation'],
                'keypoints': obj['keypoints'],
                'root_cam': None,
                'smpl_param': smpl_param,
                'cam_param': cam_param
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32),
                    'markers_3d': np.zeros((self.num_markers, 2, 2), dtype=np.float32)
                })
        return valid_objs

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

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
