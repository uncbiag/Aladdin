import torch
import numpy as np
from datasets import *
from torch.utils.data import DataLoader
import evalMetrics as metrics
from torch import optim
import os
import sys
from atlas_models import SVF_resid
import SimpleITK as sitk
from atlas_utils import *
sys.path.append(os.path.realpath(".."))
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=3, type=int, help='GPU id to use.')
parser.add_argument('--model-name', default='', type=str, help='model to be evaluated.')
args = parser.parse_args()

test_single_list, test_pair_list = get_test_list()
SVFNet_test_single = OAI_Atlas_Opt_3D(test_single_list)
SVFNet_test_single_dataloader = DataLoader(SVFNet_test_single, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
SVFNet_test_pair = OAI_Atlas_3D(test_pair_list)
SVFNet_test_pair_dataloader = DataLoader(SVFNet_test_pair, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model_path = '/playpen-raid/zpd/remote/Atlas/AtlasBuilding/ckpoints/'
model_name = 'YOUR_MODEL_NAME/'
model_name = args.model_name

bilinear = Bilinear(zero_boundary=True)


## specify atlas or search the best
best_atlas_epoch = ''
best_atlas_file = glob.glob(model_path+model_name+'/atlas_svf_img_epoch_'+best_atlas_epoch+'*.nii.gz')[-1]
best_prob_file = glob.glob(model_path+model_name+'/atlas_svf_prob_epoch_'+best_atlas_epoch+'*.pt')[-1]

print(best_atlas_file)

## get initial alignment avg dice and initial atlas
atlas_img = sitk.ReadImage(best_atlas_file)
atlas_seg = torch.load(best_prob_file).cuda(args.gpu)
atlas_img_array = sitk.GetArrayFromImage(atlas_img)
atlas_tensor = torch.from_numpy(atlas_img_array).unsqueeze(0).unsqueeze(0).cuda(args.gpu)

## load model
svf_model_state_dict = torch.load(model_path+model_name+'/model_best.pth.tar')
svf_model = SVF_resid(img_sz=np.array([80, 192, 192]), args=args)
torch.cuda.set_device(args.gpu)
svf_model.cuda(args.gpu)
svf_model.load_state_dict(svf_model_state_dict['state_dict'], strict=False)
svf_model.eval()

# dice_all = 0
img_sz=np.array([80, 192, 192])
identity_map = gen_identity_map(img_sz).unsqueeze(0).cuda(args.gpu)


dice_all = np.zeros(len(SVFNet_test_single_dataloader))

dice_bone_bridge = []
dice_cartilage_bridge = []

# example of bridge
with torch.set_grad_enabled(False):
    for tar_ind, (tar_imgs, tar_segs, tar_ids) in enumerate(SVFNet_test_single_dataloader):
        tmp_img, tmp_seg = 0, 0
        tar_imgs, tar_segs = tar_imgs.cuda(args.gpu), tar_segs.cuda(args.gpu)
        tar_cat_input = torch.cat((atlas_tensor, tar_imgs), 1)
        tar_pos_flow, _ = svf_model(tar_cat_input)
        for src_ind, (src_imgs, src_segs, src_ids) in enumerate(SVFNet_test_single_dataloader):
            if tar_ids[0] != src_ids[0]:
                src_imgs, src_segs = src_imgs.cuda(args.gpu), src_segs.cuda(args.gpu)
                print('{}/100, {}/100'.format(tar_ind+1, src_ind+1))
                src_cat_input = torch.cat((atlas_tensor, src_imgs), 1)
                _, src_neg_flow = svf_model(src_cat_input)

                trans_st_ts = bilinear(src_neg_flow, (tar_pos_flow + identity_map)) + tar_pos_flow + identity_map
                warped_img = bilinear(src_imgs, trans_st_ts)
                warped_seg = bilinear(src_segs, trans_st_ts)

                tmp_img += warped_img
                tmp_seg += warped_seg
        avg_img = tmp_img / (len(SVFNet_test_single_dataloader)-1)
        avg_seg = tmp_seg / (len(SVFNet_test_single_dataloader)-1)

        avg_seg_array = np.argmax(avg_seg.detach().squeeze().cpu().numpy(), axis=0)
        tar_segs_array = np.argmax(tar_segs.detach().squeeze().cpu().numpy(), axis=0)

        dice_bone_bridge.append(
            (metrics.metricEval('dice', avg_seg_array == 1, tar_segs_array == 1, num_labels=2) +
             metrics.metricEval('dice', avg_seg_array == 3, tar_segs_array == 3, num_labels=2)) / 2.0
        )
        dice_cartilage_bridge.append(
            (metrics.metricEval('dice', avg_seg_array == 2, tar_segs_array == 2, num_labels=2) +
             metrics.metricEval('dice', avg_seg_array == 4, tar_segs_array == 4, num_labels=2)) / 2.0
        )

# print(dice_all.mean(), dice_all.std())
print('avg bone soft dice in bridge space: {:.4f}({:.4f})'.format(np.mean(np.array(dice_bone_bridge)), np.std(np.array(dice_bone_bridge))))
print('avg cartilage soft dice in bridge space: {:.4f}({:.4f})'.format(np.mean(np.array(dice_cartilage_bridge)), np.std(np.array(dice_cartilage_bridge))))










