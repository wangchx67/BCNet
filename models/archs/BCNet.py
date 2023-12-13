import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2
from models.color_utils.soft_encode_ab import SoftEncodeAB
from models.color_utils.annealed_mean_decode_q import AnnealedMeanDecodeQ
from models.color_utils.cielab import ABGamut, CIELAB, DEFAULT_CIELAB
from models.color_utils.colorspace_transform import rgb2lab,lab2rgb
import kornia

###############################
class BCNet(nn.Module):
    def __init__(self,
                 n_feat=16,
                 chan_factor=2,
                 bias=False,
                 gpus=None,
                 phase='chrominance',
                 path_l = None,
                 color_aug = False,
                 color_space = 'Lab'):
        super(BCNet, self).__init__()

        self.phase = phase
        self.color_space = color_space

        self.lightness = arch_util.lightnessNet(1,1,n_feat,chan_factor,bias)
        self.chrominance = arch_util.ChromNet(2,2,n_feat,chan_factor,bias,color_aug)

        if phase=='lightness':
            for p in self.chrominance.parameters():
                p.requires_grad=False
        elif phase=='chrominance':
            w = torch.load(path_l)
            ww = {}
            for i in w.keys():
                if i[:9] == 'lightness':
                    ww[i[10:]] = w[i]
            self.lightness.load_state_dict(ww)
            for p in self.lightness.parameters():
                p.requires_grad=False
        elif phase == 'default':
            pass
        device = torch.device('cuda' if gpus is not None else 'cpu')
        # en-/decoding
        self.encode_ab = SoftEncodeAB(DEFAULT_CIELAB,
                                      device=device)

        self.decode_q = AnnealedMeanDecodeQ(DEFAULT_CIELAB,
                                            T=0.38,
                                            device=device)

    def forward(self, input,gt,ref = None):
        B, C, H, W = input.shape
        rate = 2 ** 4
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            input = F.pad(input, (0, pad_w, 0, pad_h), "reflect")
            if ref is not None:
                ref = F.pad(ref, (0, pad_w, 0, pad_h), "reflect")
            gt = F.pad(gt, (0, pad_w, 0, pad_h), "reflect")

        # lab_input = rgb2lab(input,norm=True)
        # lab_gt = rgb2lab(gt,norm=True)

        l_input, l_gt = self.get_luminance(input,gt,self.color_space)
        c_input, c_gt = self.get_chrominance(input,gt,self.color_space)

        lab_gt = torch.cat([l_gt,c_gt],dim=1)

        l_out, l_fea = self.lightness(l_input)

        if self.phase=='lightness':
            # l_out = l_out[:, :, :H, :W]
            # lab_gt = lab_gt[:, :, :H, :W]
            lab_output = torch.cat([l_out, c_gt], dim=1)
            lab_output = lab_output[:, :, :H, :W]
            output = self.chrominance_out_luminance(c_gt[:, :, :H, :W],l_out[:, :, :H, :W],self.color_space)
            q_pred=None
            q_actual=None
        else:
            size_q = 128 # 96 | 64 | 128
            if self.color_space != 'Lab':
                ab_out, q_pred = self.chrominance(c_input, l_fea, ref)
                q_actual = None
                lab_output = torch.cat([l_out, ab_out], dim=1)
                lab_output = lab_output[:, :, :H, :W]
                output = self.chrominance_out_luminance(ab_out[:, :, :H, :W],l_out[:, :, :H, :W],self.color_space)
            else:
                ab_gt_resized = F.interpolate(c_gt, (size_q, size_q), mode='bilinear')
                q_actual = self.encode_ab(ab_gt_resized)

                ab_out, q_pred = self.chrominance(c_input,l_fea,ref)

                lab_output = torch.cat([l_out,ab_out],dim=1)

                lab_output = lab_output[:,:,:H,:W]
                output = lab2rgb(lab_output,norm=True)

        return output,q_pred,q_actual,lab_output,lab_gt

    def get_luminance(self, img, gt, color_space):

        if color_space == 'Lab':
            gt_lab = rgb2lab(gt, norm=True)
            img_lab = rgb2lab(img, norm=True)
            img_l = img_lab[:, [0], :, :]
            gt_l = gt_lab[:, [0], :, :]
            return img_l, gt_l
        elif color_space == 'HLS':
            img_hls = kornia.color.rgb_to_hls(img)
            gt_hls = kornia.color.rgb_to_hls(gt)
            img_l = img_hls[:, [1], :, :]
            gt_l = gt_hls[:, [1], :, :]
            return img_l, gt_l
        elif color_space == 'HSV':
            img_hsv = kornia.color.rgb_to_hsv(img)
            gt_hsv = kornia.color.rgb_to_hsv(gt)
            img_l = img_hsv[:, [2], :, :]
            gt_l = gt_hsv[:, [2], :, :]
            return img_l, gt_l
        elif color_space == 'Yuv':
            img_yuv = kornia.color.rgb_to_yuv(img)
            gt_yuv = kornia.color.rgb_to_yuv(gt)
            img_l = img_yuv[:, [0], :, :]
            gt_l = gt_yuv[:, [0], :, :]
            return img_l, gt_l
        elif color_space == 'Luv':
            img_Luv = kornia.color.rgb_to_luv(img)
            gt_Luv = kornia.color.rgb_to_luv(gt)
            img_l = img_Luv[:, [0], :, :]
            gt_l = gt_Luv[:, [0], :, :]
            return img_l / 100., gt_l / 100.

    def get_chrominance(self, img, gt, color_space):

        if color_space == 'Lab':
            gt_lab = rgb2lab(gt, norm=True)
            img_lab = rgb2lab(img, norm=True)
            img_ab = img_lab[:, 1:, :, :]
            gt_ab = gt_lab[:, 1:, :, :]
            return img_ab, gt_ab
        elif color_space == 'HLS':
            img_hls = kornia.color.rgb_to_hls(img)
            gt_hls = kornia.color.rgb_to_hls(gt)
            img_h = img_hls[:, [0], :, :]
            img_s = img_hls[:, [2], :, :]
            gt_h = gt_hls[:, [0], :, :]
            gt_s = gt_hls[:, [2], :, :]
            return torch.cat((img_h / 6., img_s), dim=1), torch.cat((gt_h / 6., gt_s), dim=1)
        elif color_space == 'HSV':
            img_hsv = kornia.color.rgb_to_hsv(img)
            gt_hsv = kornia.color.rgb_to_hsv(gt)
            img_h = img_hsv[:, [0], :, :]
            img_s = img_hsv[:, [1], :, :]
            gt_h = gt_hsv[:, [0], :, :]
            gt_s = gt_hsv[:, [1], :, :]
            return torch.cat((img_h / 6., img_s), dim=1), torch.cat((gt_h / 6., gt_s), dim=1)
        elif color_space == 'Yuv':
            img_yuv = kornia.color.rgb_to_yuv(img)
            gt_yuv = kornia.color.rgb_to_yuv(gt)
            img_u = img_yuv[:, [1], :, :]
            img_v = img_yuv[:, [2], :, :]
            gt_u = gt_yuv[:, [1], :, :]
            gt_v = gt_yuv[:, [2], :, :]
            return torch.cat((img_u, img_v), dim=1), torch.cat((gt_u, gt_v), dim=1)
        elif color_space == 'Yuv':
            img_yuv = kornia.color.rgb_to_luv(img)
            gt_yuv = kornia.color.rgb_to_luv(gt)
            img_u = img_yuv[:, [1], :, :]
            img_v = img_yuv[:, [2], :, :]
            gt_u = gt_yuv[:, [1], :, :]
            gt_v = gt_yuv[:, [2], :, :]
            return torch.cat((img_u, img_v), dim=1), torch.cat((gt_u, gt_v), dim=1)
        elif color_space == 'Luv':
            img_yuv = kornia.color.rgb_to_luv(img)
            gt_yuv = kornia.color.rgb_to_luv(gt)
            img_u = img_yuv[:, [1], :, :]
            img_v = img_yuv[:, [2], :, :]
            gt_u = gt_yuv[:, [1], :, :]
            gt_v = gt_yuv[:, [2], :, :]
            return torch.cat((img_u, img_v), dim=1), torch.cat((gt_u, gt_v), dim=1)

    def chrominance_out_luminance(self,img_ab, out_l,color_space):

        if color_space == 'Lab':
            lab = torch.cat((out_l,img_ab),dim=1)
            return lab2rgb(lab,norm=True)
        elif color_space == 'HLS':
            img_hs = img_ab
            hls = torch.cat((img_hs[:,[0],:,:]*6.,out_l,img_hs[:,[1],:,:]),dim=1)
            return kornia.color.hls_to_rgb(hls)
        elif color_space == 'HSV':
            img_hs = img_ab
            hsv = torch.cat((img_hs[:,[0],:,:]*6.,img_hs[:,[1],:,:],out_l),dim=1)
            return kornia.color.hsv_to_rgb(hsv)
        elif color_space == 'Yuv':
            img_uv = img_ab
            yuv = torch.cat((out_l,img_uv),dim=1)
            return kornia.color.yuv_to_rgb(yuv)
        elif color_space == 'Luv':
            img_uv = img_ab
            luv = torch.cat((out_l*100.,img_uv),dim=1)
            return kornia.color.luv_to_rgb(luv)

class BCNet_wodecoupling(nn.Module):
    def __init__(self,
                 n_feat=16,
                 chan_factor=2,
                 bias=False,
                 gpus=None,
                 phase='chrominance',
                 path_l = None,
                 color_aug = False):
        super(BCNet_wodecoupling, self).__init__()

        self.phase = phase

        self.lightness = arch_util.lightnessNet(1,1,n_feat,chan_factor,bias)
        self.chrominance = arch_util.ChromNet2(2,2,n_feat,chan_factor,bias,color_aug)

        if phase=='lightness':
            for p in self.chrominance.parameters():
                p.requires_grad=False
        elif phase=='chrominance':
            w = torch.load(path_l)
            ww = {}
            for i in w.keys():
                if i[:9] == 'lightness':
                    ww[i[10:]] = w[i]
            self.lightness.load_state_dict(ww)
            for p in self.lightness.parameters():
                p.requires_grad=False
        elif phase == 'default':
            pass
        device = torch.device('cuda' if gpus is not None else 'cpu')
        # en-/decoding
        self.encode_ab = SoftEncodeAB(DEFAULT_CIELAB,
                                      device=device)

        self.decode_q = AnnealedMeanDecodeQ(DEFAULT_CIELAB,
                                            T=0.38,
                                            device=device)

    def forward(self, input,gt,ref = None):
        B, C, H, W = input.shape
        rate = 2 ** 4
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            input = F.pad(input, (0, pad_w, 0, pad_h), "reflect")
            if ref is not None:
                ref = F.pad(ref, (0, pad_w, 0, pad_h), "reflect")
            gt = F.pad(gt, (0, pad_w, 0, pad_h), "reflect")

        lab_input = rgb2lab(input,norm=True)
        lab_gt = rgb2lab(gt,norm=True)

        l_out, l_fea = self.lightness(lab_input[:,[0],:,:])

        if self.phase=='lightness':
            # l_out = l_out[:, :, :H, :W]
            # lab_gt = lab_gt[:, :, :H, :W]
            lab_output = torch.cat([l_out, lab_gt[:,1:,:,:]], dim=1)
            lab_output = lab_output[:, :, :H, :W]
            output=lab2rgb(lab_output,norm=True)
            q_pred=None
            q_actual=None
        elif self.phase == 'chrominance':
            size_q = 128 # 96 | 64 | 128
            ab_gt_resized = F.interpolate(lab_gt[:,1:,:,:], (size_q, size_q), mode='bilinear')
            q_actual = self.encode_ab(ab_gt_resized)

            ab_out, q_pred = self.chrominance(lab_input[:,1:,:,:],l_fea,ref)

            lab_output = torch.cat([l_out,ab_out],dim=1)

            lab_output = lab_output[:,:,:H,:W]
            output = lab2rgb(lab_output,norm=True)

        return output,q_pred,q_actual,lab_output,lab_gt

class BCNet_wodecoupling2(nn.Module):
    def __init__(self,
                 n_feat=16,
                 chan_factor=2,
                 bias=False,
                 gpus=None,
                 phase='chrominance',
                 path_l = None,
                 color_aug = False):
        super(BCNet_wodecoupling2, self).__init__()

        self.phase = phase

        self.lightness = arch_util.lightnessNet2(3,3,n_feat,chan_factor,bias)

    def forward(self, input,gt,ref = None):
        B, C, H, W = input.shape
        rate = 2 ** 4
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            input = F.pad(input, (0, pad_w, 0, pad_h), "reflect")

        l_out, l_fea = self.lightness(input)

        return l_out,None,None,None,None