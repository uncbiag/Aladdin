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
sys.path.append(os.path.realpath(".."))
import argparse
import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
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

dice_all_atlas, dice_all_image = 0, 0
img_sz=np.array([80, 192, 192])
identity_map = gen_identity_map(img_sz).unsqueeze(0).cuda(args.gpu)
tmp_img, tmp_seg = 0, 0


with torch.set_grad_enabled(False):
## create avg seg
    for j, (mean_src_imgs, mean_src_segs, _) in enumerate(SVFNet_test_single_dataloader):
        mean_src_imgs, mean_src_segs = mean_src_imgs.cuda(args.gpu), mean_src_segs.cuda(args.gpu)

        src_cat_input = torch.cat((atlas_tensor, mean_src_imgs), 1)
        mean_pos_flow_src, mean_neg_flow_src = svf_model(src_cat_input)
        mean_pos_deform_field_src = mean_pos_flow_src + identity_map
        mean_neg_deform_field_src = mean_neg_flow_src + identity_map

        mean_warped_src_imgs = bilinear(mean_src_imgs, mean_neg_deform_field_src)
        mean_warped_src_segs = bilinear(mean_src_segs, mean_neg_deform_field_src)

        tmp_img += mean_warped_src_imgs
        tmp_seg += mean_warped_src_segs
    mean_img_tensor = tmp_img / len(SVFNet_test_single_dataloader)
    mean_seg_tensor = tmp_seg / len(SVFNet_test_single_dataloader)



