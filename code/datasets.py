from torch.utils.data import Dataset
import torch
import numpy as np
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import DataLoader

class OAI_Atlas_3D(Dataset):
    def __init__(self, list_pair_IDs):
        self.list_pair_IDs = list_pair_IDs

    def __len__(self):
        return len(self.list_pair_IDs)

    def __getitem__(self, id):
        (src_id, tar_id) = self.list_pair_IDs[id]
        target_path = '/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/'
        tar_img = sitk.ReadImage(target_path+tar_id+'_image.nii.gz')
        tar_img_array = sitk.GetArrayFromImage(tar_img)
        tar_seg = sitk.ReadImage(target_path+tar_id+'_masks.nii.gz')
        tar_seg_array = sitk.GetArrayFromImage(tar_seg)
        src_img = sitk.ReadImage(target_path+src_id+'_image.nii.gz')
        src_img_array = sitk.GetArrayFromImage(src_img)
        src_seg = sitk.ReadImage(target_path+src_id+'_masks.nii.gz')
        src_seg_array = sitk.GetArrayFromImage(src_seg)

        sample = {
            'src_img': torch.from_numpy(src_img_array).unsqueeze(0),
            'tar_img': torch.from_numpy(tar_img_array).unsqueeze(0),
            'src_seg': self.mask_to_one_hot(torch.from_numpy(src_seg_array).unsqueeze(0), 5),
            'tar_seg': self.mask_to_one_hot(torch.from_numpy(tar_seg_array).unsqueeze(0), 5),
            'src_id': src_id, 'tar_id': tar_id
        }

        return sample['src_img'], sample['tar_img'], sample['src_seg'], sample['tar_seg'], sample['src_id'], sample['tar_id']

    def mask_to_one_hot(self, mask, n_classes):
        """
        Convert a segmentation mask to one-hot coded tensor
        :param mask: mask tensor of size 1xDxMxN
        :param n_classes: number of classes
        :return: one_hot: CxDxMxN
        """
        one_hot_shape = list(mask.shape)
        one_hot_shape[0] = n_classes

        mask_one_hot = torch.zeros(one_hot_shape)

        mask_one_hot.scatter_(0, mask.long(), 1)

        return mask_one_hot


class OAI_Atlas_Opt_3D(Dataset):
    def __init__(self, list_single_IDs):
        self.list_single_IDs = list_single_IDs

    def __len__(self):
        return len(self.list_single_IDs)

    def __getitem__(self, id):
        src_id = self.list_single_IDs[id]
        target_path = '/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/'
        src_img = sitk.ReadImage(target_path+src_id+'_image.nii.gz')
        src_img_array = sitk.GetArrayFromImage(src_img)
        src_seg = sitk.ReadImage(target_path+src_id+'_masks.nii.gz')
        src_seg_array = sitk.GetArrayFromImage(src_seg)

        sample = {
            'src_img': torch.from_numpy(src_img_array).unsqueeze(0),
            'src_seg': self.mask_to_one_hot(torch.from_numpy(src_seg_array).unsqueeze(0), 5),
            'src_id': src_id
        }

        return sample['src_img'], sample['src_seg'], sample['src_id']

    def mask_to_one_hot(self, mask, n_classes):
        """
        Convert a segmentation mask to one-hot coded tensor
        :param mask: mask tensor of size 1xDxMxN
        :param n_classes: number of classes
        :return: one_hot: CxDxMxN
        """
        one_hot_shape = list(mask.shape)
        one_hot_shape[0] = n_classes

        mask_one_hot = torch.zeros(one_hot_shape)

        mask_one_hot.scatter_(0, mask.long(), 1)

        return mask_one_hot


if __name__ == '__main__':
    pass


