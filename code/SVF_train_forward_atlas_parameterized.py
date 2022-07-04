import torch
import numpy as np
from datasets import *
from torch.utils.data import DataLoader
import evalMetrics as metrics
from torch import optim
from tensorboardX import SummaryWriter
import time
import datetime
import os
import sys
from atlas_models import SVF_resid
import SimpleITK as sitk
from atlas_utils import *
sys.path.append(os.path.realpath(".."))
import warnings
import argparse
import random
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--atlas-lr', '--atlas-learning-rate', default=1e4, type=float,
                    metavar='ALR', help='initial atlas learning rate')
parser.add_argument('--reg-factor', default=20000.0, type=float, help='regularization factor')
parser.add_argument('--sim-factor', default=10.0, type=float, help='similarity factor')
parser.add_argument('--image-pair-sim-factor', default=0.0, type=float, help='pairwise similarity factor')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--sim-loss', default='SSD', type=str, help='Similarity Loss to use.')
parser.add_argument('--save-per-epoch', default=10, type=int, help='number of epochs to save model.')


if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    train_single_list, valid_single_list = get_train_valid_list_forward_atlas()

    max_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    atlas_lr = args.atlas_lr
    loss_name = args.sim_loss
    best_score = 0.0

    reg_factor = args.reg_factor
    sim_factor = args.sim_factor
    pair_sim_factor = args.image_pair_sim_factor
    smooth_factor = 0.1

    using_affine_init = False

    SVFNet_train = OAI_Atlas_Opt_3D(train_single_list)
    SVFNet_train_dataloader = DataLoader(SVFNet_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    SVFNet_atlas_update = OAI_Atlas_Opt_3D(train_single_list)
    SVFNet_atlas_update_dataloader = DataLoader(SVFNet_atlas_update, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    SVFNet_val = OAI_Atlas_Opt_3D(valid_single_list)
    SVFNet_val_dataloader = DataLoader(SVFNet_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    experiment_name = 'CVPR22_OAI_3D_SVFNet_Forward_Atlas_Parameterize_' + str(loss_name) \
                      + '_affine_init_' + str(int(using_affine_init)) \
                      + '_seed_' + str(args.seed) \
                      + '_reg_' + str(reg_factor) \
                      + '_atlas_sim_' + str(sim_factor) \
                      + '_pair_sim_' + str(pair_sim_factor) \
                      + '_smooth_' + str(smooth_factor) \
                      + '_epoch_' + str(max_epochs) \
                      + '_batchsize_' + str(batch_size) \
                      + '_network_lr_' + str(lr) \
                      + '_atlas_lr_' + str(atlas_lr)
    train_model = SVF_resid(img_sz=np.array([80, 192, 192]), args=args)
    train_model.weights_init()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        train_model.cuda(args.gpu)
    else:
        train_model.cuda()

    optimizer = optim.Adam(train_model.parameters(), lr=lr)

    train_model.train()
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    writer = SummaryWriter(os.path.join('./logs', now_date, experiment_name + '_' + now_time))

    ## get initial alignment avg dice and initial atlas
    atlas_img = sitk.ReadImage('/playpen-raid/zpd/remote/Atlas/AtlasBuilding/initial_avg_atlas/averaged_initial_atlas_img.nii.gz')
    atlas_seg = torch.load('/playpen-raid/zpd/remote/Atlas/AtlasBuilding/initial_avg_atlas/averaged_initial_atlas_prob_seg.pt')
    atlas_img_array = sitk.GetArrayFromImage(atlas_img)
    atlas_tensor = torch.from_numpy(atlas_img_array).unsqueeze(0).unsqueeze(0)

    atlas_tensor.requires_grad = True
    atlas_optimizer = optim.SGD([atlas_tensor], lr=atlas_lr, weight_decay=0)

    bilinear = Bilinear(zero_boundary=False)

    img_sz = np.array([80, 192, 192])
    batch_sz = batch_size
    train_identity_map = gen_identity_map(img_sz).unsqueeze(0).repeat(batch_sz, 1, 1, 1, 1).cuda(args.gpu)
    val_identity_map = gen_identity_map(img_sz).unsqueeze(0).cuda(args.gpu)

    for epoch in range(max_epochs):
        atlas_optimizer.zero_grad()
        atlas_imgs = atlas_tensor.repeat(batch_sz, 1, 1, 1, 1).cuda(args.gpu)
        atlas_segs = atlas_seg.repeat(batch_sz, 1, 1, 1, 1).cuda(args.gpu)
        for i, (src_imgs, src_segs, src_ids) in enumerate(SVFNet_train_dataloader):
            global_step = epoch * len(SVFNet_train_dataloader) + (i + 1) * batch_size
            src_imgs, src_segs = src_imgs.cuda(args.gpu), src_segs.cuda(args.gpu)
            optimizer.zero_grad()

            cat_input = torch.cat((atlas_imgs, src_imgs), 1)
            pos_flow, neg_flow = train_model(cat_input)
            pos_deform_field = pos_flow + train_identity_map
            neg_deform_field = neg_flow + train_identity_map

            svf_warped_atlas_imgs = bilinear(atlas_imgs, pos_deform_field)
            svf_warped_atlas_segs = bilinear(atlas_segs, pos_deform_field)

            ## to evaluate in image space
            if pair_sim_factor != 0.0:
                sec_pos_deform_field = torch.flip(pos_deform_field, dims=[0])
                sec_src_imgs = torch.flip(src_imgs, dims=[0])
                sec_src_segs = torch.flip(src_segs, dims=[0])
                svf_warped_src_imgs_in_image_space = bilinear(src_imgs, (bilinear(neg_flow, sec_pos_deform_field) + sec_pos_deform_field))
                svf_warped_src_segs_in_image_space = bilinear(src_segs, (bilinear(neg_flow, sec_pos_deform_field) + sec_pos_deform_field))


            ## loss
            sim_loss = get_sim_loss(svf_warped_atlas_imgs, src_imgs, loss_name)
            reg_loss = get_reg_loss(pos_flow)
            if pair_sim_factor != 0.0:
                pair_sim_loss = get_pair_sim_loss_image_space(svf_warped_src_imgs_in_image_space, sec_src_imgs, loss_name)

            if pair_sim_factor == 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss
            elif pair_sim_factor != 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + pair_sim_factor * pair_sim_loss

            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/training', loss.item(), global_step=global_step)
            if pair_sim_factor == 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}'.format(
                        epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item())
                )
            elif pair_sim_factor != 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}, pair_sim_factor: {}, pair_sim_loss: {}'.format(
                    epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item(), pair_sim_factor, pair_sim_loss.item())
                )

            del svf_warped_atlas_imgs, svf_warped_atlas_segs, pos_flow, pos_deform_field, neg_flow, neg_deform_field
        with torch.no_grad():
            atlas_tensor.grad = atlas_tensor.grad/len(SVFNet_train_dataloader)
        smooth_loss = smooth_factor * get_first_order_reg_loss(atlas_tensor)
        smooth_loss.backward()
        atlas_optimizer.step()

        ## Validate to save the best atlas and model parameters
        if epoch % args.save_per_epoch == (args.save_per_epoch - 1):
            with torch.set_grad_enabled(False):
                ## create avg seg
                tmp_img, tmp_seg = 0, 0
                dice_all = 0
                atlas_imgs = atlas_tensor.cuda(args.gpu)
                atlas_segs = atlas_seg.cuda(args.gpu)
                for _, (mean_src_imgs, mean_src_segs, _) in enumerate(SVFNet_val_dataloader):
                    mean_src_imgs, mean_src_segs = mean_src_imgs.cuda(args.gpu), mean_src_segs.cuda(args.gpu)

                    src_cat_input = torch.cat((atlas_imgs, mean_src_imgs), 1)
                    _, mean_neg_flow_src = train_model(src_cat_input)
                    mean_neg_deform_field_src = mean_neg_flow_src + val_identity_map

                    mean_warped_src_segs = bilinear(mean_src_segs, mean_neg_deform_field_src)

                    tmp_seg += mean_warped_src_segs
                mean_atlas_seg_tensor = tmp_seg / len(SVFNet_val_dataloader)

                ## inference
                for _, (inf_src_imgs, inf_src_segs, _) in enumerate(SVFNet_val_dataloader):
                    inf_src_imgs, inf_src_segs = inf_src_imgs.cuda(args.gpu), inf_src_segs.cuda(args.gpu)
                    src_cat_input = torch.cat((atlas_imgs, inf_src_imgs), 1)
                    pos_flow, _ = train_model(src_cat_input)

                    pos_deform_field = pos_flow + val_identity_map
                    svf_warped_atlas_segs = bilinear(mean_atlas_seg_tensor, pos_deform_field)

                    dice_all += (1.0 - get_atlas_seg_loss(inf_src_segs, svf_warped_atlas_segs))

                dice_avg = dice_all / len(SVFNet_val_dataloader)
                print("{} epoch, {} iter, training loss: {:.5f}, val dice: {:.5f}".format(epoch, i + 1, loss.item(), dice_avg))
                writer.add_scalar('validation/dice_avg', dice_avg, global_step=global_step)


                if dice_avg > best_score:
                    best_score = dice_avg.item()
                    print('{} epoch, current highest - Dice: {:.5f}'.format(epoch, dice_avg))
                    writer.add_scalar('validation/highest_dice', dice_avg, global_step=global_step)
                    save_model_path = './ckpoints/' + experiment_name + '/'
                    if not os.path.isdir(save_model_path):
                        os.mkdir(save_model_path)
                    best_state = {'epoch': epoch,
                                  'state_dict': train_model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'best_score': best_score,
                                  'global_step': global_step
                                  }
                    torch.save(best_state, save_model_path + 'model_best.pth.tar')
                    tmp_img, tmp_seg, JD_denominator = 0, 0, 0
                    for _, (update_src_imgs, update_src_segs, _) in enumerate(SVFNet_atlas_update_dataloader):
                        update_src_imgs, update_src_segs = update_src_imgs.cuda(args.gpu), update_src_segs.cuda(args.gpu)

                        src_cat_input = torch.cat((atlas_imgs, update_src_imgs), 1)
                        update_pos_flow_src, update_neg_flow_src = train_model(src_cat_input)
                        update_pos_deform_field_src = update_pos_flow_src + val_identity_map
                        update_neg_deform_field_src = update_neg_flow_src + val_identity_map

                        update_warped_src_segs = bilinear(update_src_segs, update_neg_deform_field_src)

                        JD_tensor = torch.from_numpy(jacobian_determinant(update_neg_deform_field_src)).unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        JD_denominator += JD_tensor

                        tmp_seg += (update_warped_src_segs*JD_tensor)
                    atlas_seg = tmp_seg / JD_denominator
                    save_atlas_img_name = save_model_path + 'atlas_svf_img_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.nii.gz'
                    save_atlas_est_name = save_model_path + 'atlas_svf_est_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.nii.gz'
                    save_atlas_prob_name = save_model_path + 'atlas_svf_prob_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.pt'
                    save_updated_atlas(atlas_tensor, atlas_seg, save_atlas_img_name, save_atlas_est_name, save_atlas_prob_name)

            save_model_path = './ckpoints/' + experiment_name + '/'
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            current_state = {'epoch': epoch,
                             'state_dict': train_model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'best_score': best_score,
                             'global_step': global_step
                            }
            torch.save(current_state, save_model_path + 'checkpoint.pth.tar')



    writer.close()