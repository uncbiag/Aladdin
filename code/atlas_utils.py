import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from losses import LNCCLoss, DiceLossMultiClass, BendingEnergyLoss, GradLoss, NCCLoss



class Bilinear(nn.Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, zero_boundary=False, using_scale=False):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.using_scale = using_scale
        """ scale [-1,1] image intensity into [0,1], this is due to the zero boundary condition we may use here """

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]), padding_mode=self.zero_boundary, align_corners=True)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        if self.using_scale:
            output = self.forward_stn((input1 + 1) / 2, input2)
            # print(STNVal(output, ini=-1).sum())
            return output * 2 - 1
        else:
            output = self.forward_stn(input1, input2)
            # print(STNVal(output, ini=-1).sum())
            return output


def get_test_list():
    with open('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/test.txt', 'r') as f_test:
        test_list = list(f_test.read().splitlines())

    final_test_list  = test_list[-100:]
    final_test_pair_list = []
    for i in range(len(final_test_list)):
        for j in range(i+1, len(final_test_list)):
            final_test_pair_list.append((final_test_list[i], final_test_list[j]))

    return final_test_list, final_test_pair_list


def get_train_valid_list_forward_atlas():
    with open('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/train.txt', 'r') as f_train:
        train_list = list(f_train.read().splitlines())
    with open('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/valid.txt', 'r') as f_valid:
        valid_list = list(f_valid.read().splitlines())
    with open('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/test.txt', 'r') as f_test:
        test_list = list(f_test.read().splitlines())

    final_train_list = train_list + test_list[:-100]
    final_valid_list = valid_list
    final_test_list  = test_list[-100:]


    return final_train_list, final_valid_list


def identity_map(sz, dtype=np.float32):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index
    spacing = 1./ (np.array(sz)-1)

    for d in range(dim):
        id[d] *= spacing[d]
        id[d] = id[d]*2 - 1

    return torch.from_numpy(id.astype(np.float32))


def not_normalized_identity_map(sz):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))


def gen_identity_map(img_sz, resize_factor=1.,normalized=True):
    """
    given displacement field,  add displacement on grid field  todo  now keep for reproduce  this function will be disabled in the next release, replaced by spacing version
    """
    dim = 3
    if isinstance(resize_factor, list):
        img_sz = [int(img_sz[i] * resize_factor[i]) for i in range(dim)]
    else:
        img_sz = [int(img_sz[i] * resize_factor) for i in range(dim)]
    if normalized:
        grid = identity_map(img_sz)
    else:
        grid = not_normalized_identity_map(img_sz)
    return grid


def gen_identity_ap():
    """
    get the idenityt affine parameter

    :return:
    """
    affine_identity = torch.zeros(12).cuda()
    affine_identity[0] = 1.
    affine_identity[4] = 1.
    affine_identity[8] = 1.

    return affine_identity



def get_sim_loss(warped, target, loss_name):
    """
    compute the similarity loss

    :param loss_fn: the loss function
    :param output: the warped image
    :param target: the target image
    :return: the similarity loss average on batch
    """
    # loss_fn = self.ncc if self.epoch < self.epoch_activate_extern_loss else loss_fn
    if loss_name == 'LNCC':
        sim_criterion = LNCCLoss()
    elif loss_name == 'NCC':
        sim_criterion = NCCLoss()
    elif loss_name == 'SSD':
        sim_criterion = nn.MSELoss(size_average=True)
    else:
        raise ValueError('Undefined loss for similarity measure')
    sim_loss = sim_criterion(warped, target)

    return sim_loss / warped.shape[0]


def get_pair_sim_loss(warped_img, loss_name):
    batch_size = warped_img.shape[0]
    if loss_name == 'LNCC':
        pair_criterion = LNCCLoss()
    elif loss_name == 'NCC':
        pair_criterion = NCCLoss()
    elif loss_name == 'SSD':
        pair_criterion = nn.MSELoss(size_average=True)
    else:
        raise ValueError('Undefined loss for similarity measure')
    pair_loss = pair_criterion(warped_img[:int(batch_size/2)], warped_img[int(batch_size/2):])

    return pair_loss / (batch_size/2.)


def get_pair_sim_loss_image_space(warped_img1, warped_img2, loss_name):
    if loss_name == 'LNCC':
        pair_criterion = LNCCLoss()
    elif loss_name == 'NCC':
        pair_criterion = NCCLoss()
    elif loss_name == 'SSD':
        pair_criterion = nn.MSELoss(size_average=True)
    else:
        raise ValueError('Undefined loss for similarity measure')
    pair_loss = pair_criterion(warped_img1, warped_img2)

    return pair_loss / warped_img1.shape[0]



def get_atlas_seg_loss(warped_segs, atlas_segs):
    """
    compute the similarity loss

    :param loss_fn: the loss function
    :param output: the warped image
    :param target: the target image
    :return: the similarity loss average on batch
    """
    batch_size = warped_segs.shape[0]
    seg_criterion = DiceLossMultiClass(n_class=5, weight_type='Uniform', no_bg=True)
    seg_loss = seg_criterion(warped_segs, atlas_segs)

    return seg_loss / batch_size


def get_sym_loss(rec_src_phi_warped, rec_tar_phi_warped, n_batch):
    """
    compute the symmetric loss,
    :math: `loss_{sym} = \|(\varphi^{s t})^{-1} \circ(\varphi^{t s})^{-1}-i d\|_{2}^{2}`

    :param rec_phiWarped:the transformation map, including two direction ( s-t, t-s in batch dimension)
    :return: mean(`loss_{sym}`)
    """
    src_A_map = rec_src_phi_warped[:n_batch]
    src_B_map = rec_src_phi_warped[n_batch:]
    tar_B_map = rec_tar_phi_warped[:n_batch]
    tar_A_map = rec_tar_phi_warped[n_batch:]

    return torch.mean((src_A_map-tar_A_map)**2 + (src_B_map - tar_B_map)**2)


def get_reg_loss(disp_flow):
    reg_criterion = BendingEnergyLoss()
    reg_loss = reg_criterion(disp_flow)

    return reg_loss

def get_first_order_reg_loss(disp_flow):
    reg_criterion = GradLoss(penalty='l2')
    reg_loss = reg_criterion(disp_flow)

    return reg_loss

def jacobian_determinant(deform_field):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    deform_map_np = deform_field.permute([0, 2, 3, 4, 1]).squeeze().detach().cpu().numpy()
    volshape = deform_map_np.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute gradients
    # specify the voxel spacing!!!
    J = np.gradient(deform_map_np, 2.0/79.0, 2.0/191.0, 2.0/191.0, 1.0)

    # 3D flow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]



def save_updated_atlas(atlas_img, atlas_seg, save_atlas_img_name, save_atlas_est_name, save_atlas_prob_name):
    torch.save(atlas_seg, save_atlas_prob_name)
    atlas_img_np = atlas_img.detach().squeeze().cpu().numpy()
    atlas_seg_np = atlas_seg.detach().squeeze().cpu().numpy()
    atlas_est_np = torch.max(atlas_seg, 1)[1].detach().squeeze().cpu().numpy()
    tmp_img = sitk.ReadImage('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/9001104_image.nii.gz')
    tmp_seg = sitk.ReadImage('/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/9001104_masks.nii.gz')
    atlas_img_nii = sitk.GetImageFromArray(atlas_img_np.astype('float32'))
    atlas_img_nii.CopyInformation(tmp_img)
    sitk.WriteImage(atlas_img_nii, save_atlas_img_name)
    atlas_est_nii = sitk.GetImageFromArray(atlas_est_np.astype('float32'))
    atlas_est_nii.CopyInformation(tmp_seg)
    sitk.WriteImage(atlas_est_nii, save_atlas_est_name)
