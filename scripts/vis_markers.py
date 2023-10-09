"""Image demo script"""
import argparse
import os
import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from litehuman.models import builder
from litehuman.utils.config import update_config
from litehuman.utils.presets import VirtualmarkersTransform3DSMPLCam
from litehuman.utils.vis import get_one_box
from litehuman.models.layers.smpl.SMPL import SMPL_layer
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pytorch3d
import pytorch3d.renderer
from pytorch3d.renderer import TexturesVertex
from scipy.spatial.transform import Rotation

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
MARKERS_INDEX_FRONT = ['RFHD', 'LFHD', 'RCHEECK', 'LCHEECK', 'RSHO', 'CLAV', 'LSHO',
                    'RUPA', 'LUPA', 'STRN', 'RBUST', 'LBUST', 'RELB', 'LELB', 'RELBIN', 'LELBIN',
                    'RFRM', 'LFRM', 'LNWST', 'RNWST', 'MBLLY', 'RFWT', 'LFWT', 'LIWR', 'RIWR',
                    'RTHMB', 'LTHMB', 'RFIN', 'LFIN', 'RTHI', 'LTHI', 'RFTHI', 'LFTHI', 'RFTHIIN', 'LFTHIIN',
                    'RKNE', 'LKNE', 'LKNI', 'RKNI', 'RSHN', 'LSHN', 'RANK', 'LANK', 'RMT1', 'LMT1',
                    'RMT5', 'LMT5', 'RRSTBEEF', 'LRSTBEEF', 'RTOE', 'LTOE']
MARKERS_INDEX_BACK = ['LBHD', 'RBHD', 'C7', 'RBAK', 'LBAK', 'RUPA', 'LUPA', 'LSCAP', 'RSCAP', 'T10',
                    'RELB', 'LELB', 'LELBIN', 'RELBIN', 'RFRM', 'LFRM', 'LNWST', 'RNWST', 'LBWT', 'RBWT',
                    'LIWR', 'LFIN', 'RTHI', 'LTHI', 'LBHD', 'RBHD', 'RKNE', 'LKNE',
                    'RSHN', 'LSHN', 'RANK', 'LANK', 'LHEE', 'RHEE']

det_transform = T.Compose([T.ToTensor()])
def render_mesh(vertices, joints, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # add the translation
    vertices = vertices + translation[:, None, :]
    joints = joints * 2.2 + translation[:, None, :]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)
    vertices = vertices + translation[:, None, :]

    faces = faces.expand(bs, *faces.shape).to(device)

    # vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)
    joints = torch.matmul(rot, joints.transpose(1, 2)).transpose(1, 2)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # mesh_center = torch.mean(vertices, dim=1)
    # camera_position = mesh_center + torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32, device=device)
    # look_at = mesh_center
    # cameras.location = camera_position
    # cameras.look_at = look_at

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    points_raster_settings = pytorch3d.renderer.PointsRasterizationSettings(
        image_size=(height, width),   # (H, W)
        radius=0.02
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    mesh_renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )
    points_renderer = pytorch3d.renderer.PointsRenderer(
        rasterizer=pytorch3d.renderer.PointsRasterizer(
            cameras=cameras,
            raster_settings=points_raster_settings            
        ),
        compositor=pytorch3d.renderer.AlphaCompositor()
    )
    index = [all_smpl_markers[key] for key in MARKERS_INDEX_FRONT]
    vertex_markers = vertices[:, index, :]
    # print(vertex_markers)
    # print(vertex_markers.shape)
    # points_features = torch.tensor([0.0, 0.0, 1.0])
    points_features = torch.zeros_like(vertex_markers)
    points_features[:, :, 2] = 1.0
    joints_features = torch.zeros_like(joints)
    joints_features[:, :, :2] = 1.0
    markers = pytorch3d.structures.Pointclouds(points=vertex_markers, features=points_features)
    joints = pytorch3d.structures.Pointclouds(points=joints, features=joints_features)
    # print(markers)
    # Do rendering
    imgs_markers = points_renderer(markers)
    imgs_mesh = mesh_renderer(mesh)
    imgs_joints = points_renderer(joints)
    # print(imgs_markers.shape)
    # print(imgs_mesh.shape)
    print(imgs_joints.shape)
    # merged_image = imgs_mesh.clone()
    merged_image = imgs_joints
    # merged_image[imgs_markers[..., 3]>imgs_mesh[..., 3]] = imgs_mesh[imgs_markers[..., 3]>imgs_mesh[..., 3]]

    return merged_image

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

parser = argparse.ArgumentParser(description='LiteHuman Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=2,
                    type=int)
parser.add_argument('--img-dir',
                    help='image folder',
                    default='test_image/input',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='test_image/output',
                    type=str)
opt = parser.parse_args()

# device = torch.device('cuda:1')
cfg_file = 'configs/vm_hrnet.yaml'
CKPT = 'model_files/litehuman_ckpt/vm_hrnetw48.pth'
cfg = update_config(cfg_file)
bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2200, 2200, 2200))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]

dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'marker_pairs_67': None,
    'bbox_3d_shape': bbox_3d_shape    
})

transformation = VirtualmarkersTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE']
)

h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
smpl = SMPL_layer(
    './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    h36m_jregressor=h36m_jregressor,
    dtype=torch.float32    
).to(opt.gpu)
det_model = fasterrcnn_resnet50_fpn(pretrained=True)

litehuman_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    litehuman_model.load_state_dict(model_dict)
else:
    litehuman_model.load_state_dict(save_dict)

det_model.cuda(opt.gpu)
litehuman_model.cuda(opt.gpu)

det_model.eval()
litehuman_model.eval()
files = os.listdir(opt.img_dir)
smpl_faces = torch.from_numpy(litehuman_model.smpl.faces.astype(np.int32))
if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

for file in files:
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        # is an image
        if file[:4] == 'res_':
            continue

        # process file name
        img_path = os.path.join(opt.img_dir, file)
        print('processing ', img_path)
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # det_input = det_transform(input_image).to(opt.gpu)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        tight_bbox = get_one_box(det_output)  # xyxy
        if tight_bbox is None:
            continue
        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, tight_bbox)
        # pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_input = pose_input[None, :, :, :].to(opt.gpu)
        
        pose_output = litehuman_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
        )

        pred_beta = pose_output.pred_shape
        pred_theta = pose_output.pred_theta_mats.reshape(-1, 24, 4)
        # print(pred_theta[0, 0, :])
        # rotation_angle = 180.0
        # rotation_matrix = Rotation.from_euler('z', rotation_angle, degrees=True).as_matrix().astype(np.float32)
        # print(rotation_matrix.shape)
        # rotation_matrix = torch.from_numpy(rotation_matrix).to(opt.gpu)
        t_theta = torch.zeros((1, 24, 3), dtype=torch.float32).to(opt.gpu)
        # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
        t_theta[0, 0, :] = torch.from_numpy(-np.pi * np.array([1, 0, 0])).float().to(opt.gpu)
        t_smpl_output = smpl(
            pose_axis_angle=t_theta[:, 1:, :],
            betas=pred_beta,
            global_orient=t_theta[:, [0], :],
            transl=None,
            return_verts=True,
            pose2rot=True
        )
        t_vertices = t_smpl_output.vertices.detach()

        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()

        # Visualization
        image = input_image.copy()
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)

        focal = focal / 256 * bbox_xywh[2]

        vertices = pose_output.pred_vertices.detach()
        joints = pose_output.pred_xyz_jts_24.detach()
        joints = joints.reshape(-1, 24, 3)

        verts_batch = vertices
        # verts_batch = t_vertices 

        transl_batch = transl

        color_batch = render_mesh(
            vertices=verts_batch, joints=joints, faces=smpl_faces,
            translation=transl_batch,
            focal_length=focal, height=image.shape[0], width=image.shape[1])

        color = color_batch[0].cpu().numpy()
        color = color * 255
        image_vis = color.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
        res_path = os.path.join(opt.out_dir, basename)
        cv2.imwrite(res_path, image_vis)

        # valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        # image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        # image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        # color = image_vis_batch[0]
        # valid_mask = valid_mask_batch[0].cpu().numpy()
        # mask = 255 * np.ones([color.shape[0], color.shape[1], 3], np.float32)        
        # input_img = mask
        # alpha = 0.9
        # image_vis = alpha * color[:, :, :3] * valid_mask + (
        #     1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        # image_vis = image_vis.astype(np.uint8)
        # # print(color.shape)

        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        # res_path = os.path.join(opt.out_dir, basename)
        # cv2.imwrite(res_path, image_vis)