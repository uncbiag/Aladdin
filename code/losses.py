# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################

class Loss(object):
    """
    implementation of loss function
    current support list:
    "l1": Lasso
    "mse": mean square error
    'ncc': normalize cross correlation
    'lncc': localized normalized lncc (here, we implement the multi-kernel localized normalized lncc)
    """
    def __init__(self,opt):
        super(Loss,self).__init__()
        cont_loss_type = opt['tsk_set']['loss']['type']
        class_num = opt['tsk_set']['seg']['class_num']

        if cont_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif cont_loss_type == 'mse':
            self.criterion = nn.MSELoss(size_average=True)
        elif cont_loss_type =='ncc':
            self.criterion = NCCLoss()
        elif cont_loss_type =='lncc':
            lncc =  LNCCLoss()
            lncc.initialize()
            self.criterion =lncc
        elif cont_loss_type =='empty':
            self.criterion = None
        elif cont_loss_type =='ce':
            ce_opt = opt['tsk_set']['loss']['ce']
            ce_opt['class_num'] = class_num
            self.criterion = CrossEntropyLoss(ce_opt)
        elif cont_loss_type == 'focal_loss':
            focal_loss = FocalLoss()
            focal_loss.initialize(class_num, alpha=None, gamma=2, size_average=True)
            self.criterion = focal_loss
        elif cont_loss_type == 'dice_loss':
            dice_loss =  DiceLoss()
            dice_loss.initialize(class_num,None)
            self.criterion =dice_loss
        elif cont_loss_type == 'gdice_loss':
            dice_loss =  GeneralizedDiceLoss()
            dice_loss.initialize(class_num,None)
            self.criterion =dice_loss
        elif cont_loss_type == 'tdice_loss':
            dice_loss =  TverskyLoss()
            dice_loss.initialize(class_num,None)
            self.criterion =dice_loss
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)



    def get_loss(self,output, gt, inst_weights=None, train=False):
        if self.criterion is not None:
            return self.criterion(output,gt)





class NCCLoss(nn.Module):
    """
    A implementation of the normalized cross correlation (NCC)
    """
    def forward(self,input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1).view(input.shape[0],1)
        target_minus_mean = target - torch.mean(target, 1).view(input.shape[0],1)
        nccSqr = ((input_minus_mean * target_minus_mean).mean(1)) / torch.sqrt(
                    ((input_minus_mean ** 2).mean(1)) * ((target_minus_mean ** 2).mean(1)))
        nccSqr =  nccSqr.mean()

        return (1 - nccSqr)*input.shape[0]


class LNCCLoss(nn.Module):
    """This is an generalized LNCC; we implement multi-scale (means resolution)
    multi kernel (means size of neighborhood) LNCC.

    :param: resol_bound : type list,  resol_bound[0]> resol_bound[1] >... resol_bound[end]
    :param: kernel_size_ratio: type list,  the ratio of the current input size
    :param: kernel_weight_ratio: type list,  the weight ratio of each kernel size, should sum to 1
    :param: stride: type_list, the stride between each pixel that would compute its lncc
    :param: dilation: type_list

    Settings in json::

        "similarity_measure": {
                "develop_mod_on": false,
                "sigma": 0.5,
                "type": "lncc",
                "lncc":{
                    "resol_bound":[-1],
                    "kernel_size_ratio":[[0.25]],
                    "kernel_weight_ratio":[[1.0]],
                    "stride":[0.25,0.25,0.25],
                    "dilation":[1]
                }

    For multi-scale multi kernel, e.g.,::

        "resol_bound":[64,32],
        "kernel_size_ratio":[[0.0625,0.125, 0.25], [0.25,0.5], [0.5]],
        "kernel_weight_ratio":[[0.1,0.3,0.6],[0.3,0.7],[1.0]],
        "stride":[0.25,0.25,0.25],
        "dilation":[1,2,2] #[2,1,1]

    or for single-scale single kernel, e.g.,::

        "resol_bound":[-1],
        "kernel_size_ratio":[[0.25]],
        "kernel_weight_ratio":[[1.0]],
        "stride":[0.25],
        "dilation":[1]


    Multi-scale is controlled by "resol_bound", e.g resol_bound = [128, 64], it means if input size>128, then it would compute multi-kernel
    lncc designed for large image size,  if 64<input_size<128, then it would compute multi-kernel lncc desiged for mid-size image, otherwise,
    it would compute the multi-kernel lncc designed for small image.
    Attention! we call it multi-scale just because it is designed for multi-scale registration or segmentation problem.
    ONLY ONE scale would be activated during computing the similarity, which depends on the current input size.

    At each scale, corresponding multi-kernel lncc is implemented, here multi-kernel means lncc with different window sizes
    Loss = w1*lncc_win1 + w2*lncc_win2 ... + wn*lncc_winn, where /sum(wi) =1
    for example. when (image size) S>128, three windows sizes can be used, namely S/16, S/8, S/4.
    for easy notation, we use img_ratio to refer window size, the example here use the parameter [1./16,1./8,1.4]

    In implementation, we compute lncc by calling convolution function, so in this case, the [S/16, S/8, S/4] refers
    to the kernel size of convolution function.  Intuitively,  we would have another two parameters,
    stride and dilation. For each window size (W), we recommend using W/4 as stride. In extreme case the stride can be 1, but
    can large increase computation.   The dilation expand the reception field, set dilation as 2 would physically twice the window size.
    """

    def initialize(self, kernel_sz = [9,9,9], voxel_weights = None):
        pass


    def __stepup(self,img_sz, use_multi_scale=True):
        max_scale  = min(img_sz)
        if use_multi_scale:
            if max_scale>128:
                self.scale = [int(max_scale/16), int(max_scale/8), int(max_scale/4)]
                self.scale_weight = [0.1, 0.3, 0.6]
                self.dilation = [2,2,2]
            elif max_scale>64:
                self.scale = [int(max_scale / 4), int(max_scale / 2)]
                self.scale_weight = [0.3,0.7]
                self.dilation = [2,2]
            else :
                self.scale = [int(max_scale / 2)]
                self.scale_weight = [1.0]
                self.dilation = [1]
        else:
            self.scale_weight =  [int(max_scale/4)]
            self.scale_weight = [1.0]
        self.num_scale = len(self.scale)
        self.kernel_sz = [[scale for _ in range(3)] for scale in self.scale]
        self.step = [[max(int((ksz + 1) / 4),1) for ksz in self.kernel_sz[scale_id]] for scale_id in range(self.num_scale)]
        self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]

        self.conv = F.conv3d


    def forward(self, input, target):
        self.__stepup(img_sz=list(input.shape[2:]))
        input_2 = input ** 2
        target_2 = target ** 2
        input_target = input * target
        lncc_total = 0.
        for scale_id in range(self.num_scale):
            input_local_sum = self.conv(input, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                        stride=self.step[scale_id]).view(input.shape[0], -1)
            target_local_sum = self.conv(target, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                         stride=self.step[scale_id]).view(input.shape[0],
                                                                          -1)
            input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                          stride=self.step[scale_id]).view(input.shape[0],
                                                                           -1)
            target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                           stride=self.step[scale_id]).view(
                input.shape[0], -1)
            input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id], stride=self.step[scale_id]).view(
                input.shape[0], -1)

            input_local_sum = input_local_sum.contiguous()
            target_local_sum = target_local_sum.contiguous()
            input_2_local_sum = input_2_local_sum.contiguous()
            target_2_local_sum = target_2_local_sum.contiguous()
            input_target_local_sum = input_target_local_sum.contiguous()

            numel = float(np.array(self.kernel_sz[scale_id]).prod())

            input_local_mean = input_local_sum / numel
            target_local_mean = target_local_sum / numel

            cross = input_target_local_sum - target_local_mean * input_local_sum - \
                    input_local_mean * target_local_sum + target_local_mean * input_local_mean * numel
            input_local_var = input_2_local_sum - 2 * input_local_mean * input_local_sum + input_local_mean ** 2 * numel
            target_local_var = target_2_local_sum - 2 * target_local_mean * target_local_sum + target_local_mean ** 2 * numel

            lncc = cross * cross / (input_local_var * target_local_var + 1e-5)
            lncc = 1 - lncc.mean()
            lncc_total += lncc * self.scale_weight[scale_id]

        return lncc_total*(input.shape[0])



class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def initialize(self, class_num, alpha=None, gamma=2, size_average=True, verbose=True):
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha

        self.alpha = torch.squeeze(self.alpha)
        if verbose:
            print("the alpha of focal loss is  {}".format(alpha))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, weight= None, inst_weights=None, train=None):
        """

        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, inputs.size(1))
        targets = targets.view(-1)

        P = F.softmax(inputs,dim=1)
        ids = targets.view(-1)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]

        log_p = - F.cross_entropy(inputs, targets,reduce=False)
        probs = F.nll_loss(P, targets,reduce=False)
        # print(probs)
        # print(log_p)
        # print(torch.pow((1 - probs), self.gamma))
        # print(alpha)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class DiceLoss(nn.Module):
    def initialize(self, class_num, weight = None):
        self.class_num = class_num
        self.class_num = class_num
        if weight is None:
            self.weight =torch.ones(class_num, 1)/self.class_num
        else:
            self.weight = weight
        self.weight = torch.squeeze(self.weight)

    def forward(self,input, target, inst_weights=None,train=None):
        """
        input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
        target is a Bx....   range 0,1....N_label
        """
        in_sz = input.size()
        from functools import reduce
        extra_dim = reduce(lambda x,y:x*y,in_sz[2:])
        targ_one_hot = torch.zeros(in_sz[0],in_sz[1],extra_dim).cuda()
        targ_one_hot.scatter_(1,target.view(in_sz[0],1,extra_dim),1.)
        target = targ_one_hot.view(in_sz).contiguous()
        probs = F.softmax(input,dim=1)
        num = probs*target
        num = num.view(num.shape[0],num.shape[1],-1)
        num = torch.sum(num, dim=2)

        den1 = probs#*probs
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)

        den2 = target#*target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)
        # print("den1:{}".format(sum(sum(den1))))
        # print("den2:{}".format(sum(sum(den2/den1))))


        dice = 2 * (num / (den1 + den2))
        dice = self.weight.expand_as(dice) * dice
        dice_eso = dice
        # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
        return dice_total


class DiceLossMultiClass(nn.Module):
    """Dice loss from two inputs of segmentation (between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Uniform', no_bg=False, softmax=False, eps=1e-7):
        super(DiceLossMultiClass, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.eps = eps
        self.no_bg = no_bg
        self.softmax = softmax  # if the source inputs are in 0~1 range
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size BxDxMxN;
                             or class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[-3:] == target.squeeze().shape[-3:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        if self.softmax:
            source = F.softmax(source, dim=1)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if len(target.shape) == len(shape)-1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            target_flat = None
            raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))

        # does not consider background
        if self.no_bg:
            source_flat = source_flat[:, 1:, :]
            target_flat = target_flat[:, 1:, :]

        #
        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)

        if self.weight_type == 'Simple':
            # weights = (target_volume.float().sqrt() + self.eps).reciprocal()
            weights = (target_volume.float()**(1./3.) + self.eps).reciprocal()
            # temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            # max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            # weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        elif self.weight_type == 'Volume':
            weights = (target_volume + self.eps).float().reciprocal()
            # weights = 1/(target_volume ** 2+self.eps)
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * max_weights, weights)
        elif self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1] - int(self.no_bg))
        else:
            raise ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        weights = weights / weights.max()
        # print(weights)
        weights = weights.to(source.device)

        intersection = (source_flat * target_flat).sum(2)
        scores = (2. * (intersection.float()) + self.eps) / (
                (source_volume.float() + target_volume.float()) + 2 * self.eps)

        return 1 - (weights*scores).sum()/weights.sum()


class GeneralizedDiceLoss(nn.Module):
    def initialize(self, class_num, weight=None):
        self.class_num = class_num
        if weight is None:
            self.weight =torch.ones(class_num, 1)
        else:
            self.weight = weight

        self.weight = torch.squeeze(self.weight)
    def forward(self,input, target,inst_weights=None,train=None):
        """
        input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
        target is a Bx....   range 0,1....N_label
        """
        in_sz = input.size()
        from functools import reduce
        extra_dim = reduce(lambda x,y:x*y,in_sz[2:])
        targ_one_hot = torch.zeros(in_sz[0],in_sz[1],extra_dim).cuda()
        targ_one_hot.scatter_(1,target.view(in_sz[0],1,extra_dim),1.)
        target = targ_one_hot.view(in_sz).contiguous()
        probs = F.softmax(input,dim=1)
        num = probs*target
        num = num.view(num.shape[0],num.shape[1],-1)
        num = torch.sum(num, dim=2)  # batch x ch

        den1 = probs
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)  # batch x ch

        den2 = target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)  # batch x ch
        # print("den1:{}".format(sum(sum(den1))))
        # print("den2:{}".format(sum(sum(den2/den1))))
        weights = self.weight.expand_as(den1)

        dice = 2 * (torch.sum(weights*num,dim=1) / torch.sum(weights*(den1 + den2),dim=1))
        dice_eso = dice
        # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total


class TverskyLoss(nn.Module):
    def initialize(self, class_num, weight=None, alpha=0.5, beta=0.5):
        self.class_num = class_num
        if weight is None:
            self.weight = torch.ones(class_num, 1)/self.class_num
        else:
            self.weight = weight

        self.weight = torch.squeeze(self.weight)
        self.alpha = alpha
        self.beta = beta
        print("the weight of Tversky loss is  {}".format(weight))

    def forward(self,input, target,inst_weights=None, train=None):
        """
        input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
        target is a Bx....   range 0,1....N_label
        """
        in_sz = input.size()
        from functools import reduce
        extra_dim = reduce(lambda x,y:x*y,in_sz[2:])
        targ_one_hot = torch.zeros(in_sz[0],in_sz[1],extra_dim).cuda()
        targ_one_hot.scatter_(1,target.view(in_sz[0],1,extra_dim),1.)
        target = targ_one_hot.view(in_sz).contiguous()
        probs = F.softmax(input,dim=1)
        num = probs*target
        num = num.view(num.shape[0],num.shape[1],-1)
        num = torch.sum(num, dim=2)


        den1 = probs*(1-target)
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)

        den2 = (1-probs)*target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)
        # print("den1:{}".format(sum(sum(den1))))
        # print("den2:{}".format(sum(sum(den2/den1))))


        dice = 2 * (num / (num + self.alpha*den1 + self.beta*den2))
        dice = self.weight.expand_as(dice) * dice

        dice_eso = dice
        #dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total


class CrossEntropyLoss(nn.Module):
    def __init__(self, opt, imd_weight=None):
        # To Do,  add dynamic weight
        super(CrossEntropyLoss,self).__init__()
        no_bg = opt[('no_bg',False,'exclude background')]
        weighted = opt[('weighted',False,'  weighted the class')]
        reduced = opt[('reduced',True,'  reduced the class')]
        self.mask = None #opt[('mask',None, 'masked other label')]
        class_num = opt['class_num']

        if no_bg:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        if weighted:
            class_weight = opt['class_weight']if imd_weight is None else imd_weight
            if class_weight is not None and not (len(class_weight)< class_num):
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weight, reduce = reduced)
                self.mask=None
            else:  # this is the case for using random mask, the class weight here refers to the label need be masked
                self.mask = class_weight
                print("the current mask is {}".format(self.mask))
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduce = reduced)
        self.n_class = class_num

    def forward(self, input, gt, inst_weights= None, train=False):
        """
        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        if self.mask is not None and train:
            for m in self.mask:
                gt[gt==m]=0
        if len(input.shape)==5:
            output_flat = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_class)
        else:
            output_flat = input
        truths_flat = gt.view(-1)
        if inst_weights is None:
            return self.loss_fn(output_flat,truths_flat)
        else:
            return torch.mean( inst_weights.view(-1)*self.loss_fn(output_flat,truths_flat))


class GradLoss(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(GradLoss, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class BendingEnergyLoss(nn.Module):
    """
    regularization loss of bending energy of a 3d deformation field
    """

    def __init__(self, norm='L2', spacing=(1, 1, 1), normalize=True):
        super(BendingEnergyLoss, self).__init__()
        self.norm = norm
        self.spacing = torch.tensor(spacing).float()
        self.normalize = normalize
        if self.normalize:
            self.spacing /= self.spacing.min()

    def forward(self, input):
        """
        :param norm: 'L1' or 'L2'
        :param input: Nx3xDxHxW
        :return:
        """
        self.spacing = self.spacing.to(input.device)
        spatial_dims = torch.tensor(input.shape[2:]).float().to(input.device)
        if self.normalize:
            spatial_dims /= spatial_dims.min()

        # according to
        # f''(x) = [f(x+h) + f(x-h) - 2f(x)] / h^2
        # f_{x, y}(x, y) = [df(x+h, y+k) + df(x-h, y-k) - df(x+h, y-k) - df(x-h, y+k)] / 2hk

        ddx = torch.abs(input[:, :, 2:, 1:-1, 1:-1] + input[:, :, :-2, 1:-1, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1])\
                 .view(input.shape[0], input.shape[1], -1)

        ddy = torch.abs(input[:, :, 1:-1, 2:, 1:-1] + input[:, :, 1:-1, :-2, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]) \
                 .view(input.shape[0], input.shape[1], -1)

        ddz = torch.abs(input[:, :, 1:-1, 1:-1, 2:] + input[:, :, 1:-1, 1:-1, :-2] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]) \
                 .view(input.shape[0], input.shape[1], -1)

        dxdy = torch.abs(input[:, :, 2:, 2:, 1:-1] + input[:, :, :-2, :-2, 1:-1] -
                         input[:, :, 2:, :-2, 1:-1] - input[:, :, :-2, 2:, 1:-1]).view(input.shape[0], input.shape[1], -1)

        dydz = torch.abs(input[:, :, 1:-1, 2:, 2:] + input[:, :, 1:-1, :-2, :-2] -
                         input[:, :, 1:-1, 2:, :-2] - input[:, :, 1:-1, :-2, 2:]).view(input.shape[0], input.shape[1], -1)

        dxdz = torch.abs(input[:, :, 2:, 1:-1, 2:] + input[:, :, :-2, 1:-1, :-2] -
                         input[:, :, 2:, 1:-1, :-2] - input[:, :, :-2, 1:-1, 2:]).view(input.shape[0], input.shape[1], -1)


        if self.norm == 'L2':
            ddx = (ddx ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0]**2)) ** 2
            ddy = (ddy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1]**2)) ** 2
            ddz = (ddz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2]**2)) ** 2
            dxdy = (dxdy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0] * self.spacing[1])) ** 2
            dydz = (dydz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1] * self.spacing[2])) ** 2
            dxdz = (dxdz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2] * self.spacing[0])) ** 2

        d = (ddx.mean() + ddy.mean() + ddz.mean() + 2*dxdy.mean() + 2*dydz.mean() + 2*dxdz.mean()) / 9.0
        return d



class LaplaceOperator(nn.Module):
    """
    regularization loss of Laplace of a 3d velocity field
    """

    def __init__(self):
        super(LaplaceOperator, self).__init__()

    def forward(self, input):
        """
        :param norm: 'L1' or 'L2'
        :param input: Nx3xDxHxW
        :return:
        """

        # according to
        # f''(x) = [f(x+h) + f(x-h) - 2f(x)] / h^2
        # f_{x, y}(x, y) = [df(x+h, y+k) + df(x-h, y-k) - df(x+h, y-k) - df(x-h, y+k)] / 2hk

        input = F.pad(input, (1, 1, 1, 1, 1, 1), mode='replicate')

        ddx = input[:, :, 2:, 1:-1, 1:-1] + input[:, :, :-2, 1:-1, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]

        ddy = input[:, :, 1:-1, 2:, 1:-1] + input[:, :, 1:-1, :-2, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]

        ddz = input[:, :, 1:-1, 1:-1, 2:] + input[:, :, 1:-1, 1:-1, :-2] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]


        return ddx + ddy + ddz




