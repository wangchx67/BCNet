import torch
import models.archs.BCNet as BCNet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'BCNet':
        netG = BCNet.BCNet(n_feat=opt_net['n_feat'],
                         chan_factor=opt_net['chan_factor'],
                         bias=opt_net['bias'],
                           gpus=opt['gpu_ids'],
                           phase=opt['train']['phase'],
                           path_l = opt['train']['path_lightness'],
                           color_aug=opt_net['color_aug'],
                           color_space = opt_net['color_space'])
    elif which_model == 'BCNet_wodecoupling':
        netG = BCNet.BCNet_wodecoupling(n_feat=opt_net['n_feat'],
                         chan_factor=opt_net['chan_factor'],
                         bias=opt_net['bias'],
                           gpus=opt['gpu_ids'],
                           phase=opt['train']['phase'],
                           path_l = opt['train']['path_lightness'],
                           color_aug=opt_net['color_aug'],)
    elif which_model == 'BCNet_wodecoupling2':
        netG = BCNet.BCNet_wodecoupling2(n_feat=opt_net['n_feat'],
                         chan_factor=opt_net['chan_factor'],
                         bias=opt_net['bias'],
                           gpus=opt['gpu_ids'],
                           phase=opt['train']['phase'],
                           path_l = opt['train']['path_lightness'],
                           color_aug=opt_net['color_aug'],)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

