import torch
import numpy as np
import os
from PIL import Image
import torchvision

l_cent = 0.
l_norm = 100.
ab_norm = 110.
# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])
    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.to(rgb.device)

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.to(rgb.device)

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.to(xyz.device)

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.to(xyz_scale.device)

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).to(z_int.device), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.to(out.device)

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc
    return out

def rgb2lab(rgb,norm=False):
    lab = xyz2lab(rgb2xyz(rgb))
    if norm:
        # l_rs = (lab[:, [0], :, :] - l_cent) / 100.0
        # ab_rs = (lab[:, 1:, :, :]+110.) / 220.
        # out = torch.cat((l_rs,ab_rs),dim=1)
        l_rs = (lab[:, [0], :, :] - l_cent) / 100.0
        ab_rs = (lab[:, 1:, :, :]) / 110.
        out = torch.cat((l_rs,ab_rs),dim=1)
    else:
        out=lab

    return out

def lab2rgb(lab_rs,norm=False):
    l = lab_rs[:, [0], :, :]
    ab = lab_rs[:, 1:, :, :]
    if norm:
        # l = lab_rs[:, [0], :, :] * 100 + l_cent
        # ab = lab_rs[:, 1:, :, :] * 220. - 110.
        l = lab_rs[:, [0], :, :] * 100. + l_cent
        ab = lab_rs[:, 1:, :, :] * 110.
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))

    return out

def calc_hist(data_ab):
    w=14
    N, C, H, W = data_ab.shape
    grid_a = torch.linspace(-1, 1, w).view(1, w, 1, 1, 1).expand(N, w, w, H, W).cuda()
    grid_b = torch.linspace(-1, 1, w).view(1, 1, w, 1, 1).expand(N, w, w, H, W).cuda()
    hist_a = torch.max(0.1 - torch.abs(grid_a - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).cuda()) * 10
    hist_b = torch.max(0.1 - torch.abs(grid_b - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).cuda()) * 10
    hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1)
    return hist

def encode_ab(ab):

    ab_gamut=np.load('./utils/ab-gamut.npy',encoding = "latin1")

    q_to_ab=torch.from_numpy(ab_gamut).to(ab.device).type(ab.dtype)
    n, _, h, w = ab.shape

    m = n * h * w

    # find nearest neighbours
    ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)

    cdist = torch.cdist(q_to_ab, ab_.t())

    nns = cdist.argsort(dim=0)[:1, :]

    q = nns.reshape(1,n,h,w).permute(1, 0, 2, 3)

    return q


def decode_q(q):

    ab_gamut=np.load('./utils/ab-gamut.npy',encoding = "latin1")

    q_to_ab = torch.from_numpy(ab_gamut).to(q.device)

    ab=q_to_ab[q]
    ab=ab.squeeze(1).permute(0,3,1,2)

    return ab

if __name__ == "__main__":

    img = Image.open('../../color_dark/1_gt.jpg')
    img = (np.asarray(img) / 255.0)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    lab=rgb2lab(img)
    l=lab[:,[0],:,:]
    ab=lab[:,1:,:,:]
    q=encode_ab(ab)
    ab_=decode_q(q)
    out=torch.cat((l,ab_),dim=1)
    out=lab2rgb(out)
    torchvision.utils.save_image(out, "test.jpg")