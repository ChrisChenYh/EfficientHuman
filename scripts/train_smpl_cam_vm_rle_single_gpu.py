"""Script for multi-gpu training."""
import os
import pickle as pk
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad

from litehuman.datasets import MixDataset, MixDatasetCam, PW3D, MixDataset2Cam, MixDataset3Cam, MixDataset2VMCam, H36mSMPLVM, PW3DVM
from litehuman.models import builder
from litehuman.opt import cfg, logger, opt
from litehuman.utils.env import init_dist
from litehuman.utils.metrics import DataLogger, NullWriter, calc_coord_accuracy, vertice_pve
from litehuman.utils.transforms import flip, get_func_heatmap_to_coord
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

import time

# torch.set_num_threads(64)
num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu


def _init_fn(worker_id):
    np.random.seed(opt.seed)
    random.seed(opt.seed)

def train(opt, train_loader, m, criterion, optimizer, writer, epoch_num):
    loss_logger = DataLogger()
    loss_uvd_logger = DataLogger()
    loss_beta_logger = DataLogger()
    loss_theta_logger = DataLogger()
    loss_twist_logger = DataLogger()
    loss_camtrans_logger = DataLogger()
    loss_camscale_logger = DataLogger()

    acc_uvd_29_logger = DataLogger()
    acc_uvd_67_logger = DataLogger()
    acc_xyz_17_logger = DataLogger()
    m.train()
    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    depth_dim = cfg.MODEL.EXTRA.get('DEPTH_DIM')
    hm_shape = (hm_shape[1], hm_shape[0], depth_dim)
    root_idx_17 = train_loader.dataset.root_idx_17

    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    data_start_time = time.time()
    for j, (inps, labels, img_ids, bboxes) in enumerate(train_loader):
        data_time = time.time() - data_start_time
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu).requires_grad_() for inp in inps]
        else:
            inps = inps.cuda(opt.gpu).requires_grad_()

        for k, _ in labels.items():
            labels[k] = labels[k].cuda(opt.gpu)

        compute_start_time = time.time()

        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')
        root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')

        output = m(inps, labels=labels, trans_inv=trans_inv, intrinsic_param=intrinsic_param, joint_root=root, depth_factor=depth_factor)

        robust_train = cfg.LOSS.ELEMENTS.get('RUBOST_TRAIN', False)
        if robust_train:
            loss_dict = criterion(output, labels, epoch_num=epoch_num)
        else:
            loss_dict = criterion(output, labels)

        pred_uvd_jts = output.pred_uvd_jts
        pred_uvd_mks = output.pred_uvd_mks
        pred_xyz_jts_17 = output.pred_xyz_jts_17
        label_masks_29 = labels['target_weight_29']
        label_masks_17 = labels['target_weight_17']
        label_masks_67 = labels['target_weight_67']

        if pred_uvd_jts.shape[1] == 24 or pred_uvd_jts.shape[1] == 72 :
            pred_uvd_jts = pred_uvd_jts.cpu().reshape(pred_uvd_jts.shape[0], 24, 3)
            gt_uvd_jts = labels['target_uvd_29'].cpu().reshape(pred_uvd_jts.shape[0], 29, 3)[:, :24, :]
            gt_uvd_mask = label_masks_29.cpu().reshape(pred_uvd_jts.shape[0], 29, 3)[:, :24, :]
            acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts, gt_uvd_jts, gt_uvd_mask, hm_shape, num_joints=24)
        else:
            acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts.detach().cpu(), labels['target_uvd_29'].cpu(), label_masks_29.cpu(), hm_shape, num_joints=29)
        acc_uvd_67 = calc_coord_accuracy(pred_uvd_mks.detach().cpu(), labels['target_uvd_67'].cpu(), label_masks_67.cpu(), hm_shape, num_joints=67)
        acc_xyz_17 = calc_coord_accuracy(pred_xyz_jts_17.detach().cpu(), labels['target_xyz_17'].cpu(), label_masks_17.cpu(), hm_shape, num_joints=17, root_idx=root_idx_17)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss_dict.loss.item(), batch_size)
        loss_beta_logger.update(loss_dict.loss_beta.item(), batch_size)
        loss_theta_logger.update(loss_dict.loss_theta.item(), batch_size)
        loss_uvd_logger.update(loss_dict.loss_uvd.item(), batch_size)

        loss_twist_logger.update(loss_dict.loss_twist.item(), batch_size)
        
        loss_camscale_logger.update(loss_dict.loss_scale.item(), batch_size)
        loss_camtrans_logger.update(loss_dict.loss_trans.item(), batch_size)

        acc_uvd_29_logger.update(acc_uvd_29, batch_size)
        acc_uvd_67_logger.update(acc_uvd_67, batch_size)
        acc_xyz_17_logger.update(acc_xyz_17, batch_size)

        loss = loss_dict.loss

        optimizer.zero_grad()
        loss.backward()

        compute_time = time.time() - compute_start_time

        if robust_train:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    clip_grad.clip_grad_norm_(param, 5)

        optimizer.step()

        opt.trainIters += 1
        if opt.log:
            # TQDM
            train_loader.set_description(
                'loss: {loss:.8f} | accuvd29: {accuvd29:.4f} | accuvd67: {accuvd67:.4f} | acc17: {acc17:.4f}'.format(
                    loss=loss_logger.value,
                    accuvd29=acc_uvd_29_logger.value,
                    accuvd67=acc_uvd_67_logger.value,
                    acc17=acc_xyz_17_logger.value)
            )

            global_step = epoch_num * len(train_loader) + j
            writer.add_scalar(tag='train/loss_all', scalar_value=loss_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/loss_uvd', scalar_value=loss_uvd_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/loss_theta', scalar_value=loss_theta_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/loss_beta', scalar_value=loss_beta_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/loss_twist', scalar_value=loss_twist_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/loss_scale', scalar_value=loss_camscale_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/loss_trans', scalar_value=loss_camtrans_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/acc_17', scalar_value=acc_xyz_17_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/acc_29', scalar_value=acc_uvd_29_logger.value, global_step=global_step)
            writer.add_scalar(tag='train/data_time', scalar_value=data_time, global_step=global_step)
            writer.add_scalar(tag='train/compute_time', scalar_value=compute_time, global_step=global_step)

        if opt.wandb:
            wandb.log({
                'train_itr_loss': loss_logger.avg,
                'train_itr_acc29': acc_uvd_29_logger.avg,
                'train_itr_acc17': acc_xyz_17_logger.avg,
                'train_itr_uvd29_loss': loss_uvd_logger.avg,
                'train_itr_beta_loss': loss_beta_logger.avg,
                'train_itr_theta_loss': loss_theta_logger.avg,
                'train_itr_twist_loss': loss_twist_logger.avg,
                'train_itr_camtrans_loss': loss_camtrans_logger.avg,
                'train_itr_camscale_loss': loss_camscale_logger.avg
            })

        data_start_time = time.time()

    if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_xyz_17_logger.avg

def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=32, pred_root=False, test_vertices=True):

    gt_val_sampler = torch.utils.data.SequentialSampler(data_source=gt_val_dataset)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.nThreads, drop_last=False, sampler=gt_val_sampler)
    kpt_pred = {}
    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    mpvpe_logger = DataLogger()

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for inps, labels, img_ids, bboxes in gt_val_loader:
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].cuda(opt.gpu)
            except AttributeError:
                assert k == 'type'
        bboxes = bboxes.cuda(opt.gpu)
        # output = m(inps, trans_inv=trans_inv, intrinsic_param=intrinsic_param, joint_root=root, depth_factor=depth_factor, flip_output=False)
        output = m(inps, flip_test=opt.flip_test, bboxes=bboxes, img_center=labels['img_center'])

        if test_vertices:
            gt_betas = labels['target_beta']
            gt_thetas = labels['target_theta']
            gt_output = m.forward_gt_theta(gt_thetas, gt_betas)
            

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_24 = output.pred_xyz_jts_24.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)
        pred_mesh = output.pred_vertices.reshape(inps.shape[0], -1, 3)

        if test_vertices:
            gt_mesh = gt_output.vertices.reshape(inps.shape[0], -1, 3)
            gt_xyz_jts_17 = gt_output.joints_from_verts.reshape(inps.shape[0], 17, 3) / 2.2

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        pred_uvd_jts = pred_uvd_jts.cpu().data
        pred_mesh = pred_mesh.cpu().data.numpy()

        if test_vertices:
            gt_mesh = gt_mesh.cpu().data.numpy()
            gt_xyz_jts_17 = gt_xyz_jts_17.cpu().data.numpy()

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(pred_xyz_jts_17.shape[0], 17, 3)
        pred_uvd_jts = pred_uvd_jts.reshape(pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(pred_xyz_jts_24.shape[0], 24, 3)
        pred_scores = output.maxvals.cpu().data[:, :29]

        if test_vertices:
            mpvpe = vertice_pve(pred_verts=pred_mesh, target_verts=gt_mesh)
            mpvpe_logger.update(mpvpe*1000, batch_size)

        for i in range(pred_xyz_jts_17.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred_uvd_jts[i], pred_scores[i], hm_shape, bbox, mean_bbox_scale=None)
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
                'vertices': pred_mesh[i],
                'uvd_jts': pose_coords[0],
                'xyz_24': pred_xyz_jts_24_struct[i]
            }

    with open(os.path.join(opt.work_dir, f'test_gt_kpt.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    with open(os.path.join(opt.work_dir, f'test_gt_kpt.pkl'), 'rb') as fid:
        kpt_pred = pk.load(fid)
    os.remove(os.path.join(opt.work_dir, f'test_gt_kpt.pkl'))

    tot_err_17 = gt_val_dataset.evaluate_xyz_17(
        kpt_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json')
    )

    return tot_err_17, mpvpe_logger.avg

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    if opt.seed is not None:
        setup_seed(opt.seed)

    if not opt.log:
        logger.setLevel(50)
        null_writer = NullWriter()
        sys.stdout = null_writer    

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    opt.nThreads = int(opt.nThreads)
    opt.gpu = torch.device('cuda')

    # Model Initialize
    m = preset_model(cfg)

    # model summary
    if opt.params:
        from thop import clever_format, profile
        input = torch.randn(1, 3, 256, 256).cuda(opt.gpu)
        flops, params = profile(m.cuda(opt.gpu), inputs=(input, ))
        macs, params = clever_format([flops, params], "%.3f")
        logger.info(macs, params)    

    # move model to gpu
    m.cuda(opt.gpu)

    # init loss and optimizer
    criterion = builder.build_loss(cfg.LOSS).cuda(opt.gpu)
    optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)

    # set learning rate schedule
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)    

    # set wandb / tensorboard
    if opt.wandb:
        wandb.init(project='LiteHuman')
        wandb.config.update(cfg)
    if opt.log:
        writer = SummaryWriter('.tensorboard/{}/{}-{}'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
    else:
        writer = None

    # init datasets setting
    if cfg.DATASET.DATASET == 'mix_smpl':
        train_dataset = MixDataset(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix_smpl_cam':
        train_dataset = MixDatasetCam(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix2_smpl_cam':
        train_dataset = MixDataset2Cam(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix3_smpl_cam':
        train_dataset = MixDataset3Cam(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix2_smpl_vm_cam':
        train_dataset = MixDataset2VMCam(
            cfg=cfg,
            train=True
        )
    else:
        raise NotImplementedError        

    # get heatmap to coord function
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    # init train_sampler and train_dataset
    train_sampler = torch.utils.data.RandomSampler(data_source=train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=opt.nThreads,
        sampler=train_sampler,
        worker_init_fn=_init_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # get gt val dataset
    if cfg.DATASET.DATASET == 'mix_smpl':
        gt_val_dataset_h36m = MixDataset(
            cfg=cfg,
            train=False)
    elif cfg.DATASET.DATASET == 'mix_smpl_cam' or cfg.DATASET.DATASET == 'mix2_smpl_cam' or cfg.DATASET.DATASET == 'mix3_smpl_cam':
        gt_val_dataset_h36m = MixDatasetCam(
            cfg=cfg,
            train=False)
    elif cfg.DATASET.DATASET == 'mix2_smpl_vm_cam':
        gt_val_dataset_h36m = H36mSMPLVM(
            cfg=cfg,
            train=False,
            ann_file='Sample_20_test_Human36M_smpl'
        )
    else:
        raise NotImplementedError

    gt_val_dataset_3dpw = PW3DVM(
        cfg=cfg,
        ann_file='3DPW_test_new.json',
        train=False)

    # start train epoch
    opt.trainIters = 0
    best_err_h36m = 999
    best_err_3dpw = 999
    if opt.wandb:
        wandb.watch(m, log="all")
    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        # current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')
        # # Training
        # loss, acc17 = train(opt, train_loader, m, criterion, optimizer, writer, i)
        # logger.epochInfo('Train', opt.epoch, loss, acc17)
        # if opt.wandb:
        #     wandb.log({
        #         'train_epoch_loss': loss,
        #         'train_epoch_acc17': acc17
        #     })
        # if opt.log:
        #     writer.add_scalar(tag='train/loss_epoch', scalar_value=loss)
        #     writer.add_scalar(tag='train/acc_epoch', scalar_value=acc17)

        # lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            if opt.log:
                # Save checkpoint
                torch.save(m.state_dict(), './exp/{}/{}-{}/model_{}.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id, opt.epoch))
            
            # Prediction Test
            with torch.no_grad():
                gt_tot_err_h36m, h36m_mpvpe = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord)
                print(h36m_mpvpe)
                gt_tot_err_3dpw, pw3d_mpvpe = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord)
                print(pw3d_mpvpe)
                if opt.log:
                    if gt_tot_err_h36m <= best_err_h36m:
                        best_err_h36m = gt_tot_err_h36m
                        torch.save(m.state_dict(), './exp/{}/{}-{}/best_h36m_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
                    if gt_tot_err_3dpw <= best_err_3dpw:
                        best_err_3dpw = gt_tot_err_3dpw
                        torch.save(m.state_dict(), './exp/{}/{}-{}/best_3dpw_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))

                    logger.info(f'##### Epoch {opt.epoch} | h36m err: {gt_tot_err_h36m} / {best_err_h36m} | 3dpw err: {gt_tot_err_3dpw} / {best_err_3dpw} #####')

                    writer.add_scalar(tag='valid/err_3dpw', scalar_value=gt_tot_err_3dpw)
                    writer.add_scalar(tag='valid/err_h36m', scalar_value=gt_tot_err_h36m)

                if opt.wandb:
                    wandb.log({
                        'val_epoch_h36m_err': gt_tot_err_h36m,
                        'val_epoch_3dpw_err': gt_tot_err_3dpw
                    })
    torch.save(m.state_dict(), './exp/{}/{}-{}/final_DPG.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))

def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model

if __name__ == "__main__":
    main()
