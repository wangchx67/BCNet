import os.path as osp
import logging
import argparse

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from utils.CSE import *

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='./options/test/LSRW_Huawei.yml')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

def main():
    save_imgs = True
    model = create_model(opt)
    save_folder = './results/{}'.format(opt['name'])
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')


    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}  # with border and center frames
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}  # with border and center frames
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        cse_rlt = {}  # with border and center frames
        cse_rlt_avg = {}
        cse_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []
            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []
            if cse_rlt.get(folder, None) is None:
                cse_rlt[folder] = []

            model.feed_data(val_data)

            model.test()

            visuals = model.get_current_visuals()

            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
            gt_img = util.tensor2img(visuals['GT'])  # uint8
            input_img = util.tensor2img(visuals['LQ'])

            if save_imgs:
                try:
                    tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    print(osp.join(output_folder, '{}.png'.format(tag)))

                    rlt_img = cv2.cvtColor(rlt_img,cv2.COLOR_RGB2BGR)
                    gt_img = cv2.cvtColor(gt_img,cv2.COLOR_RGB2BGR)
                    input_img = cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR)

                    cv2.imwrite(osp.join(output_folder, '{}.png'.format(tag)), rlt_img)
                    cv2.imwrite(osp.join(GT_folder, '{}.png'.format(tag)), gt_img)

                    cv2.imwrite(osp.join(input_folder, '{}.png'.format(tag)), input_img)

                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()

            # calculate PSNR
            # psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr = peak_signal_noise_ratio(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)
            ## ssim = util.calculate_ssim(rlt_img, gt_img)
            ssim = structural_similarity(rlt_img, gt_img, multichannel=True)
            ssim_rlt[folder].append(ssim)

            img = cv2.cvtColor(rlt_img, cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            cse = CSE(img, gt)
            cse_rlt[folder].append(cse)

            pbar.update('Test {} - {}'.format(folder, idx_d))

        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]
        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]
        for k, v in cse_rlt.items():
            cse_rlt_avg[k] = sum(v) / len(v)
            cse_total_avg += cse_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        cse_total_avg /= len(cse_rlt)
        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # CSE: {:.4e}:'.format(cse_total_avg)
        for k, v in cse_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        print('PSNR:'+str(psnr_all))

        ssim_all = 0
        ssim_count = 0
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        print('SSIM:'+str(ssim_all))

        cse_all = 0
        cse_count = 0
        for k, v in cse_rlt.items():
            cse_all += sum(v)
            cse_count += len(v)
        cse_all = cse_all * 1.0 / cse_count
        print('CSE:' + str(cse_all))



if __name__ == '__main__':
    main()
