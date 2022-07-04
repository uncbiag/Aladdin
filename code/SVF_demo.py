import torch
import numpy as np
from datasets import *
from torch.utils.data import DataLoader
import evalMetrics as metrics
import os
import sys
from atlas_models import SVF_resid
import SimpleITK as sitk
from atlas_utils import *
from viz_deformation import *
# sys.path.append(os.path.realpath(".."))
import argparse
import glob
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
args = parser.parse_args()

test_single_list, test_pair_list = get_test_list()
SVFNet_test_single = OAI_Atlas_Opt_3D(test_single_list)
SVFNet_test_single_dataloader = DataLoader(SVFNet_test_single, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
SVFNet_test_pair = OAI_Atlas_3D(test_pair_list)
SVFNet_test_pair_dataloader = DataLoader(SVFNet_test_pair, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

model_path = '/playpen-raid/zpd/remote/Atlas/AtlasBuilding/ckpoints/'
model_name = 'YOUR_MODEL_NAME/'

bilinear = Bilinear(zero_boundary=True)


## specify atlas or search the best
best_atlas_epoch = ''
best_atlas_file = glob.glob(model_path+model_name+'atlas_svf_img_epoch_'+best_atlas_epoch+'*.nii.gz')[-1]
best_prob_file = glob.glob(model_path+model_name+'atlas_svf_prob_epoch_'+best_atlas_epoch+'*.pt')[-1]

print(best_atlas_file)

## get initial alignment avg dice and initial atlas
atlas_img = sitk.ReadImage(best_atlas_file)
atlas_seg = torch.load(best_prob_file).cuda(args.gpu)
atlas_img_array = sitk.GetArrayFromImage(atlas_img)
atlas_tensor = torch.from_numpy(atlas_img_array).unsqueeze(0).unsqueeze(0).cuda(args.gpu)

## load model
svf_model_state_dict = torch.load(model_path+model_name+'model_best.pth.tar')
svf_model = SVF_resid(img_sz=np.array([80, 192, 192]), args=args)
torch.cuda.set_device(args.gpu)
svf_model.cuda(args.gpu)
svf_model.load_state_dict(svf_model_state_dict['state_dict'], strict=False)
svf_model.eval()

img_sz=np.array([80, 192, 192])
identity_map = gen_identity_map(img_sz).unsqueeze(0).cuda(args.gpu)


src_img_1 = sitk.ReadImage('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/9684822_image.nii.gz')
src_img_2 = sitk.ReadImage('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/9686590_image.nii.gz')
src_img_1_array = sitk.GetArrayFromImage(src_img_1)
src_img_2_array = sitk.GetArrayFromImage(src_img_2)
src_1_tensor = torch.from_numpy(src_img_1_array).unsqueeze(0).unsqueeze(0).cuda()
src_2_tensor = torch.from_numpy(src_img_2_array).unsqueeze(0).unsqueeze(0).cuda()

src_cat_1_input = torch.cat((atlas_tensor, src_1_tensor), 1)
src_cat_2_input = torch.cat((atlas_tensor, src_2_tensor), 1)
pos_flow_1_src, neg_flow_1_src = svf_model(src_cat_1_input)
pos_flow_2_src, neg_flow_2_src = svf_model(src_cat_2_input)
pos_deform_field_1_src = pos_flow_1_src + identity_map
neg_deform_field_1_src = neg_flow_1_src + identity_map
pos_deform_field_2_src = pos_flow_2_src + identity_map
neg_deform_field_2_src = neg_flow_2_src + identity_map



### pairwise similarity demo
warped_src_1_img_in_atlas_space = bilinear(src_1_tensor, neg_deform_field_1_src)
warped_src_2_img_in_atlas_space = bilinear(src_2_tensor, neg_deform_field_2_src)
warped_atlas_img_in_src_1_space = bilinear(atlas_tensor, pos_deform_field_1_src)
warped_atlas_img_in_src_2_space = bilinear(atlas_tensor, pos_deform_field_2_src)
composite_src_1_to_src_2_field = bilinear(neg_flow_1_src, pos_deform_field_2_src) + pos_deform_field_2_src
composite_src_2_to_src_1_field = bilinear(neg_flow_2_src, pos_deform_field_1_src) + pos_deform_field_1_src
warped_src_1_img_in_src_2_space = bilinear(src_1_tensor, composite_src_1_to_src_2_field)
warped_src_2_img_in_src_1_space = bilinear(src_2_tensor, composite_src_2_to_src_1_field)

# show_current_images_2d_map(iS=src_1_tensor[:,:,40,:,:], iT=atlas_tensor[:,:,40,:,:], iW=warped_src_1_img_in_atlas_space[:,:,40,:,:], iSL=None, iTL=None, iWL=None, vizImage=None, vizName=None, phiWarped=neg_deform_field_1_src[:,1:,40,:,:], i=2021)
# show_current_images_2d_map(iS=src_2_tensor[:,:,40,:,:], iT=atlas_tensor[:,:,40,:,:], iW=warped_src_2_img_in_atlas_space[:,:,40,:,:], iSL=None, iTL=None, iWL=None, vizImage=None, vizName=None, phiWarped=neg_deform_field_2_src[:,1:,40,:,:], i=2022)
# show_current_images_2d_map(iS=atlas_tensor[:,:,40,:,:], iT=src_1_tensor[:,:,40,:,:], iW=warped_atlas_img_in_src_1_space[:,:,40,:,:], iSL=None, iTL=None, iWL=None, vizImage=None, vizName=None, phiWarped=pos_deform_field_1_src[:,1:,40,:,:], i=2023)
# show_current_images_2d_map(iS=atlas_tensor[:,:,40,:,:], iT=src_2_tensor[:,:,40,:,:], iW=warped_atlas_img_in_src_2_space[:,:,40,:,:], iSL=None, iTL=None, iWL=None, vizImage=None, vizName=None, phiWarped=pos_deform_field_2_src[:,1:,40,:,:], i=2024)
# show_current_images_2d_map(iS=src_1_tensor[:,:,40,:,:], iT=src_2_tensor[:,:,40,:,:], iW=warped_src_1_img_in_src_2_space[:,:,40,:,:], iSL=None, iTL=None, iWL=None, vizImage=None, vizName=None, phiWarped=composite_src_1_to_src_2_field[:,1:,40,:,:], i=2025)
# show_current_images_2d_map(iS=src_2_tensor[:,:,40,:,:], iT=src_1_tensor[:,:,40,:,:], iW=warped_src_2_img_in_src_1_space[:,:,40,:,:], iSL=None, iTL=None, iWL=None, vizImage=None, vizName=None, phiWarped=composite_src_2_to_src_1_field[:,1:,40,:,:], i=2026)

plt.subplot(111).set_axis_off()
plt.imshow(src_1_tensor[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/src_1_img.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_1_img_in_atlas_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/warped_src_1_img_in_atlas_space.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_1_img_in_atlas_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.contour(neg_deform_field_1_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.contour(neg_deform_field_1_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.savefig('./deformation_map/warped_src_1_img_in_atlas_space_deformation.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_src_1_img_in_atlas_space[:,:,40,:,:] - atlas_tensor[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_src_1_img_in_atlas_space_difference.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_1_img_in_src_2_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/warped_src_1_img_in_src_2_space.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_1_img_in_src_2_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.contour(composite_src_1_to_src_2_field[:,1:,40,:,:].squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.contour(composite_src_1_to_src_2_field[:,1:,40,:,:].squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.savefig('./deformation_map/warped_src_1_img_in_src_2_space_deformation.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(atlas_tensor[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/atlas_img.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_atlas_img_in_src_1_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/warped_atlas_img_in_src_1_space.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_atlas_img_in_src_1_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.contour(pos_deform_field_1_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.contour(pos_deform_field_1_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.savefig('./deformation_map/warped_atlas_img_in_src_1_space_deformation.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_atlas_img_in_src_1_space[:,:,40,:,:] - src_1_tensor[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_atlas_img_in_src_1_space_difference.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_src_2_img_in_src_1_space[:,:,40,:,:] - src_1_tensor[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_src_2_img_in_src_1_space_difference_with_src_1_tensor.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_atlas_img_in_src_2_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/warped_atlas_img_in_src_2_space.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_atlas_img_in_src_2_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.contour(pos_deform_field_2_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.contour(pos_deform_field_2_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.savefig('./deformation_map/warped_atlas_img_in_src_2_space_deformation.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_atlas_img_in_src_2_space[:,:,40,:,:] - src_2_tensor[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_atlas_img_in_src_2_space_difference.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_src_1_img_in_src_2_space[:,:,40,:,:] - src_2_tensor[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_src_1_img_in_src_2_space_difference_with_src_2_tensor.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(src_2_tensor[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/src_2_img.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_2_img_in_atlas_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/warped_src_2_img_in_atlas_space.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_2_img_in_atlas_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.contour(neg_deform_field_2_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.contour(neg_deform_field_2_src[:,1:,40,:,:].squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.savefig('./deformation_map/warped_src_2_img_in_atlas_space_deformation.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_src_2_img_in_atlas_space[:,:,40,:,:] - atlas_tensor[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_src_2_img_in_atlas_space_difference.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(torch.abs(warped_src_2_img_in_atlas_space[:,:,40,:,:] - warped_src_1_img_in_atlas_space[:,:,40,:,:]).squeeze().detach().cpu().numpy(), cmap='rainbow')
plt.clim(0.0, 0.8)
plt.colorbar().ax.tick_params(labelsize=10)
plt.savefig('./deformation_map/warped_src_2_img_in_atlas_space_difference_with_warped_src_1_img.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_2_img_in_src_1_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.savefig('./deformation_map/warped_src_2_img_in_src_1_space.png', bbox_inches='tight', dpi=500)
plt.close()

plt.subplot(111).set_axis_off()
plt.imshow(warped_src_2_img_in_src_1_space[:,:,40,:,:].squeeze().detach().cpu().numpy(), cmap='gray')
plt.contour(composite_src_2_to_src_1_field[:,1:,40,:,:].squeeze().detach().cpu().numpy()[0, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.contour(composite_src_2_to_src_1_field[:,1:,40,:,:].squeeze().detach().cpu().numpy()[1, :, :], np.linspace(-1, 1, 20), colors='r', linestyles='solid', linewidths=1.0)
plt.savefig('./deformation_map/warped_src_2_img_in_src_1_space_deformation.png', bbox_inches='tight', dpi=500)
plt.close()



