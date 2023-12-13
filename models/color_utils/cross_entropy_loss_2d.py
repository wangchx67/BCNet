import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from color_utils.soft_encode_ab import SoftEncodeAB
from color_utils.annealed_mean_decode_q import AnnealedMeanDecodeQ
from color_utils.cielab import ABGamut, CIELAB, DEFAULT_CIELAB
from PIL import Image
import torchvision
import numpy as np
from utils import colorspace_transform


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        softmax = log_softmax(outputs, dim=1)

        norm = labels.clone()

        norm[norm != 0] = torch.log(norm[norm != 0])

        return -torch.sum((softmax-norm) * labels) / outputs.shape[0]

if __name__ == "__main__":

    en= SoftEncodeAB(DEFAULT_CIELAB,neighbours=1)
    de = AnnealedMeanDecodeQ(DEFAULT_CIELAB,T=0)

    img = Image.open('../input.jpg')
    img = img.resize((256,256), Image.ANTIALIAS)
    img = (np.asarray(img) / 255.0)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

    gt = Image.open('../gt.jpg')
    gt = gt.resize((256, 256), Image.ANTIALIAS)
    gt = (np.asarray(gt) / 255.0)
    gt = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0)

    lab = colorspace_transform.rgb2lab(img)
    l = (lab[:, [0], :, :])/100.0
    ab = lab[:, 1:, :, :]
    lab_gt = colorspace_transform.rgb2lab(gt)
    l_gt = (lab_gt[:, [0], :, :])/100.0
    ab_gt = lab_gt[:, 1:, :, :]

    q_img=en(ab)
    q_gt=en(ab_gt)

    criterion=CrossEntropyLoss2d()

    q=nn.functional.softmax(q_gt,dim=1)

    loss=criterion(q,q_gt)

    print(loss)