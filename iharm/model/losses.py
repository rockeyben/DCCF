import torch
import torch.nn as nn
import math
from iharm.utils import misc


class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=misc.get_dims_with_exclusion(label.dim(), 0))
        return loss


class MaskWeightedMSE(Loss):
    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedMSE, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        label = label.view(pred.size())
        reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)
        if pred.device != label.device:
            label = label.to(pred.device)
            mask = mask.to(pred.device)
        
        #print(pred.device, label.device)
        loss = (pred - label) ** 2
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask[:, 0:1, :, :], dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss


class MaskWeightedCosine(Loss):
    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedCosine, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        label = label.view(pred.size())
        reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)

        #loss = 1.0 - torch.cos(pred * 255.0 / 180 * 2 * math.pi  - label * 255.0 / 180 * 2 * math.pi )
        loss = 1.0 - torch.cos(pred * 2 * math.pi - label * 2 * math.pi)
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss

class MaskWeightedTV(Loss):

    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedTV, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        batch_size = pred.size()[0]
        h_x = pred.size()[2]
        w_x = pred.size()[3]
        pred = pred * mask[:, 0:1, :, :]
        reduce_dims = misc.get_dims_with_exclusion(pred.dim(), 0)

        h_tv = torch.sum(torch.pow((pred[:,:,1:,:]-pred[:,:,:h_x-1,:]),2), dim=reduce_dims)
        w_tv = torch.sum(torch.pow((pred[:,:,:,1:]-pred[:,:,:,:w_x-1]),2), dim=reduce_dims)
        
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask[:, 0:1, :, :], dim=reduce_dims), self.min_area)
        return h_tv/delimeter+w_tv/delimeter

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
