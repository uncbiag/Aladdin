import torch
import torchvision
from torchvision import transforms
import numpy as np
import glob
from datasets import *
from torch.utils.data import DataLoader
import evalMetrics as metrics
from torch import nn, optim
from tensorboardX import SummaryWriter
import time
import datetime
import os
import argparse
import sys
from losses import *
from atlas_models import SVF_resid
from atlas_utils import *
import SimpleITK as sitk
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(".."))



def show_current_images_2d_map(iS, iT, iW, iSL, iTL, iWL, vizImage, vizName, phiWarped, i):
    if iSL is not None and iTL is not None:
        sp_s = 331
        sp_t = 332
        sp_w = 333
        sp_c = 334
        sp_p = 335
        sp_v = 336
        sp_ls = 337
        sp_lt = 338
        sp_lw = 339
    elif (vizImage is not None) and (phiWarped is not None):
        sp_s = 231
        sp_t = 232
        sp_w = 233
        sp_c = 234
        sp_p = 235
        sp_v = 236
    elif (vizImage is not None):
        sp_s = 231
        sp_t = 232
        sp_w = 233
        sp_c = 234
        sp_v = 235
    elif (phiWarped is not None):
        sp_s = 231
        sp_t = 232
        sp_w = 233
        sp_c = 234
        sp_p = 235
    else:
        sp_s = 221
        sp_t = 222
        sp_w = 223
        sp_c = 224

    font = {'size': 10}

    plt.suptitle('Deformation map')
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(sp_s).set_axis_off()
    plt.imshow(iS.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('source image', font)

    plt.subplot(sp_t).set_axis_off()
    plt.imshow(iT.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('target image', font)

    plt.subplot(sp_w).set_axis_off()
    plt.imshow(iW.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('warped image', font)

    plt.subplot(sp_c).set_axis_off()
    plt.imshow(checkerboard_2d(iW.squeeze().detach().cpu().numpy(), iT.squeeze().detach().cpu().numpy()), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('checkerboard', font)

    if phiWarped is not None:
        plt.subplot(sp_p).set_axis_off()
        plt.imshow(iW.squeeze().detach().cpu().numpy(),cmap='gray')

        plt.contour(phiWarped.squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=0.5)
        plt.contour(phiWarped.squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=0.5)

        plt.colorbar().ax.tick_params(labelsize=3)
        plt.title('warped image + grid', font)

    if vizImage is not None:
        plt.subplot(sp_v).set_axis_off()
        plt.imshow(utils.lift_to_dimension(utils.t2np(vizImage), 2), cmap='gray')
        plt.colorbar().ax.tick_params(labelsize=3)
        plt.title(vizName, font)


    if iSL is not None and iTL is not None:
        plt.subplot(sp_ls).set_axis_off()
        plt.imshow(torch.max(iSL, 1)[1].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('Source Label', font)

        plt.subplot(sp_lt).set_axis_off()
        plt.imshow(torch.max(iTL, 1)[1].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('Target Label', font)

        plt.subplot(sp_lw).set_axis_off()
        plt.imshow(torch.max(iWL, 1)[1].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('Warped Label', font)



    file_name = 'test_affine_defomation_'+str(i)+'.png'
    deformation_path = './deformation_map/'
    join_p = lambda pth1,pth2: os.path.join(pth1, pth2)

    if not os.path.isdir(deformation_path):
        os.mkdir(deformation_path)
    plt.savefig(join_p(deformation_path, file_name), dpi=500)
    plt.clf()


def show_current_images_2d_map_with_affine_projection(iS, iT, iW, iSL, iTL, iWL, vizImage, vizName, phiWarped, iWA, phiAffWarped, i):
    if iSL is not None and iTL is not None:
        sp_s = 331
        sp_t = 332
        sp_w = 333
        sp_c = 334
        sp_p = 335
        sp_v = 336
        sp_ls = 337
        sp_lt = 338
        sp_lw = 339
    elif (vizImage is not None) and (phiWarped is not None):
        sp_s = 231
        sp_t = 232
        sp_w = 233
        sp_c = 234
        sp_p = 235
        sp_v = 236
    elif (vizImage is not None):
        sp_s = 231
        sp_t = 232
        sp_w = 233
        sp_c = 234
        sp_v = 235
    elif (phiWarped is not None):
        sp_s = 241
        sp_t = 242
        sp_w = 243
        sp_c = 245
        sp_p = 247
        sp_ap = 244
        sp_aw = 248
    else:
        sp_s = 221
        sp_t = 222
        sp_w = 223
        sp_c = 224

    font = {'size': 6}

    plt.suptitle('Deformation map')
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(sp_s).set_axis_off()
    plt.imshow(iS.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('source image', font)

    plt.subplot(sp_t).set_axis_off()
    plt.imshow(iT.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('target image', font)

    plt.subplot(sp_w).set_axis_off()
    plt.imshow(iW.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('warped image', font)

    plt.subplot(sp_ap).set_axis_off()
    plt.imshow(iWA.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('affine project', font)

    plt.subplot(sp_c).set_axis_off()
    plt.imshow(checkerboard_2d(iW.squeeze().detach().cpu().numpy(), iT.squeeze().detach().cpu().numpy()), cmap='gray')
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title('checkerboard', font)

    if phiWarped is not None:
        ## diffeomorphic
        plt.subplot(sp_p).set_axis_off()
        plt.imshow(iW.squeeze().detach().cpu().numpy(),cmap='gray')

        plt.contour(phiWarped.squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=0.5)
        plt.contour(phiWarped.squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=0.5)

        plt.colorbar().ax.tick_params(labelsize=3)
        plt.title('warped image + grid', font)

        ## affine projection
        plt.subplot(sp_aw).set_axis_off()
        plt.imshow(iWA.squeeze().detach().cpu().numpy(),cmap='gray')

        plt.contour(phiAffWarped.squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=0.5)
        plt.contour(phiAffWarped.squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=0.5)

        plt.colorbar().ax.tick_params(labelsize=3)
        plt.title('affine project + grid', font)

    if vizImage is not None:
        plt.subplot(sp_v).set_axis_off()
        plt.imshow(utils.lift_to_dimension(utils.t2np(vizImage), 2), cmap='gray')
        plt.colorbar().ax.tick_params(labelsize=3)
        plt.title(vizName, font)


    if iSL is not None and iTL is not None:
        plt.subplot(sp_ls).set_axis_off()
        plt.imshow(torch.max(iSL, 1)[1].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('Source Label', font)

        plt.subplot(sp_lt).set_axis_off()
        plt.imshow(torch.max(iTL, 1)[1].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('Target Label', font)

        plt.subplot(sp_lw).set_axis_off()
        plt.imshow(torch.max(iWL, 1)[1].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('Warped Label', font)



    file_name = 'test_affine_defomation_'+str(i)+'.png'
    deformation_path = './deformation_map/'
    join_p = lambda pth1,pth2: os.path.join(pth1, pth2)

    if not os.path.isdir(deformation_path):
        os.mkdir(deformation_path)
    plt.savefig(join_p(deformation_path, file_name), dpi=500)
    plt.clf()


def checkerboard_2d(I0,I1,nrOfTiles=8):
    """
    Creates a checkerboard between two images

    :param I0: image 0, size XxYxZ
    :param I1: image 1, size XxYxZ
    :param nrOfTiles: number of desired tiles in each direction
    :return: returns tiled image
    """
    sz = I0.shape
    tileSize = int( np.array(sz).min()/nrOfTiles )
    nrOfTileXH = int( np.ceil(sz[0]/tileSize)/2+1 )
    nrOfTileYH = int( np.ceil(sz[1]/tileSize)/2+1 )
    cb_grid = np.kron([[1, 0] * nrOfTileYH, [0, 1] * nrOfTileYH] *nrOfTileXH, np.ones((tileSize, tileSize)))
    # now cut it to the same size
    cb_grid=cb_grid[0:sz[0],0:sz[1]]
    cb_image = I0*cb_grid + I1*(1-cb_grid)
    return cb_image


if __name__ == "__main__":
    pass














