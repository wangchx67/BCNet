import os.path as osp
import torch.utils.data as data
import data.util as util
import random


class VideoSameSizeDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoSameSizeDataset, self).__init__()
        self.opt = opt
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}

        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = [subfolder_LQ]
            img_paths_GT = [subfolder_GT]
            if subfolder_LQ.split('/')[-1].split('.')[-1] != 'png' and subfolder_LQ.split('/')[-1].split('.')[-1] != 'jpg'\
                    and subfolder_LQ.split('/')[-1].split('.')[-1] != 'JPG'\
                    and subfolder_LQ.split('/')[-1].split('.')[-1] != 'bmp':
                continue
            # if subfolder_LQ.split('/')[-1].split('.')[-1] != 'jpg':
            #     continue
            else:
                max_idx = len(img_paths_LQ)
                self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                self.data_info['idx'].append('{}/{}'.format(0, len(subfolder_LQ)))

                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        img_LQ_path = self.imgs_LQ[folder][0]
        img_GT_path = self.imgs_GT[folder][0]
        img_LQ_path = [img_LQ_path]
        img_GT_path = [img_GT_path]

        if self.opt['phase'] == 'train':
            # img_LQ = util.read_img_seq(img_LQ_path,self.opt['train_size'])
            img_LQ = util.read_img_seq(img_LQ_path)
            # img_GT = util.read_img_seq(img_GT_path,self.opt['train_size'])
            img_GT = util.read_img_seq(img_GT_path)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            LQ_size,  GT_size= self.opt['train_size']
            _, H, W = img_GT.shape  # real img size

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]

        elif self.opt['phase'] == 'test':
            img_LQ = util.read_img_seq(img_LQ_path)
            img_GT = util.read_img_seq(img_GT_path)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

        else:
            img_LQ = util.read_img_seq(img_LQ_path)
            img_GT = util.read_img_seq(img_GT_path)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]


        return {
            'LQs': img_LQ,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
