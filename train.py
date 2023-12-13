import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
import warnings

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='./options/train/LOLv2_real.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                # from torch.utils.tensorboard import SummaryWriter
                from tensorboardX import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)

            # print(train_set[0])
            # import pdb; pdb.set_trace()

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            if opt['datasets']['val']['ifval']:
                val_set = create_dataset(dataset_opt)
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)
                if rank <= 0:
                    logger.info('Number of val images in [{:s}]: {:d}'.format(
                        dataset_opt['name'], len(val_set)))
            else:
                logger.info('not validation')
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0
    best_psnr = 0.
    best_ssim = 0.
    #### training

    progressive = opt['datasets']['train'].get('progressive')
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            if progressive:
            ### ------Progressive learning ---------------------
                j = ((current_step > groups) != True).nonzero()[0]
                if len(j) == 0:
                    bs_j = len(groups) - 1
                else:
                    bs_j = j[0]

                mini_gt_size = mini_gt_sizes[bs_j]
                mini_batch_size = mini_batch_sizes[bs_j]

                if logger_j[bs_j]:
                    logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size,
                                                                                              mini_batch_size * torch.cuda.device_count()))
                    logger_j[bs_j] = False

                if mini_batch_size < batch_size:
                    indices = random.sample(range(0, batch_size), k=mini_batch_size)
                    train_data['LQs'] = train_data['LQs'][indices]
                    train_data['GT'] = train_data['GT'][indices]

                if mini_gt_size < gt_size:
                    x0 = int((gt_size - mini_gt_size) * random.random())
                    y0 = int((gt_size - mini_gt_size) * random.random())
                    x1 = x0 + mini_gt_size
                    y1 = y0 + mini_gt_size
                    train_data['LQs'] = train_data['LQs'][:, :, x0:x1, y0:y1]
                    train_data['GT'] = train_data['GT'][:, :, x0:x1, y0:y1]
                ###-------------------------------------------

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['datasets']['val']['ifval']:
                    psnr_rlt = {}  # with border and center frames
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.
                    ssim_rlt = {}  # with border and center frames
                    ssim_rlt_avg = {}
                    ssim_total_avg = 0.
                    idx = 0

                    for val_data in val_loader:
                        folder = val_data['folder'][0]
                        # border = val_data['border'].item()
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []
                        if ssim_rlt.get(folder, None) is None:
                            ssim_rlt[folder] = []

                        idx += 1
                        img_dir = os.path.join(opt['path']['val_images'], folder.split('.')[0])
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        rlt_img = cv2.cvtColor(rlt_img, cv2.COLOR_RGB2BGR)
                        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.jpg'.format(folder.split('.')[0], current_step))
                        util.save_img(rlt_img, save_img_path)

                        # calculate PSNR
                        psnr = util.calculate_psnr(rlt_img, gt_img)
                        psnr_rlt[folder].append(psnr)
                        ssim = util.calculate_ssim(rlt_img, gt_img)
                        ssim_rlt[folder].append(ssim)
                        # pbar.update('Test {} - {}'.format(folder, idx_d))
                    for k, v in psnr_rlt.items():
                        psnr_rlt_avg[k] = sum(v) / len(v)
                        psnr_total_avg += psnr_rlt_avg[k]
                    for k, v in ssim_rlt.items():
                        ssim_rlt_avg[k] = sum(v) / len(v)
                        ssim_total_avg += ssim_rlt_avg[k]
                    psnr_total_avg /= len(psnr_rlt)
                    ssim_total_avg /= len(ssim_rlt)
                    log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                    log_s1 = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                    if psnr_total_avg > best_psnr:
                        best_psnr = psnr_total_avg
                        model.save('best')
                        log_s += ' best psnr : {:.4e}'.format(best_psnr)
                    if ssim_total_avg > best_ssim:
                        best_ssim = ssim_total_avg
                        log_s1 += ' best ssim : {:.4e}'.format(best_ssim)
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)
                    logger.info(log_s)
                    logger.info(log_s1)
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                        for k, v in psnr_rlt_avg.items():
                            tb_logger.add_scalar(k, v, current_step)
                else:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

            #### save models and training states
            # if current_step % opt['logger']['save_checkpoint_freq'] == 0:
            #     if rank <= 0:
            #         logger.info('Saving models and training states.')
            #         model.save(current_step)
            #         model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        # tb_logger.close()


if __name__ == '__main__':
    main()
