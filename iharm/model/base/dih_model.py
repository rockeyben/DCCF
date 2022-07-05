from numpy.lib.arraypad import _pad_dispatcher, _pad_simple
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
import time
from torchvision.ops import roi_align as tv_roi_align

from iharm.model.modeling.conv_autoencoder import ConvEncoder, DeconvDecoder, DeconvDecoderMhead, \
                DeconvDecoderUpsample, DeconvDecoderUpsamplePconv, DeconvDecoderUpsampleTrans
from iharm.model.modeling.basic_blocks import ConvBlock
from iharm.model.modeling.transUnet import Transformer
from iharm.model.modeling.ViT_configs import get_b16_config, get_r50_b16_config
from iharm.model.modifiers import LRMult
from iharm.model.ps_filters import *




class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap, use_cpu=False): 
        # print(bilateral_grid.shape, guidemap.shape)
        # Nx12x8x16x16
        N, _, H, W = guidemap.shape
        N, C, lh, lw = bilateral_grid.shape
        bilateral_grid = bilateral_grid.view(N, C // 8, 8, lh, lw)

        device = bilateral_grid.get_device()
        
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)

        return coeff.squeeze(2)

class GuideNN(nn.Module):
    def __init__(self):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, stride=1, padding=0, norm_layer=None, activation=nn.Sigmoid) #nn.Tanh

    def forward(self, x):
        return self.conv2(self.conv1(x))#.squeeze(1)

class CurveGuideNN(nn.Module):
    def __init__(self, npts=16):
        super(CurveGuideNN, self).__init__()
        ccm_ = np.identity(3, dtype=np.float32) + np.random.randn(1).astype(np.float32)*1e-4
        bias_ = np.zeros((3,), dtype=np.float32)
        ccm_ = ccm_[np.newaxis, :, :, np.newaxis, np.newaxis]
        bias_ = bias_[np.newaxis, :, np.newaxis, np.newaxis]
        shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
        shifts_ = shifts_[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        shifts_ = np.tile(shifts_, (1, 3, 1, 1, 1))
        slopes_ = np.zeros((1, 3, npts, 1, 1), dtype=np.float32)
        slopes_[:, :, 0, :, :] = 1.0

        self.lower_bound = -2.11
        self.upper_bound = 2.66
        self.npts = npts
        self.relu = nn.ReLU()
        self.ccm = nn.Parameter(torch.FloatTensor(ccm_))
        self.bias = nn.Parameter(torch.FloatTensor(bias_))
        self.shifts = nn.Parameter(torch.FloatTensor(shifts_))
        self.slopes = nn.Parameter(torch.FloatTensor(slopes_))
        self.conv2 = ConvBlock(3, 1, kernel_size=1, stride=1, padding=0, norm_layer=None, activation=nn.Sigmoid) #nn.Tanh
        self.register_parameter('ccm', self.ccm)
        self.register_parameter('ccm_b', self.bias)
        self.register_parameter('curve_a', self.shifts)
        self.register_parameter('curve_b', self.slopes)

    def forward(self, x, use_cpu=False):
        #print(self.ccm, self.bias, self.shifts, self.slopes)
        #print(self.ccm.shape, self.bias.shape, self.shifts.shape, self.slopes.shape)
        #print(self.shifts)
        B, C, H, W = x.shape
        x = (x - self.lower_bound) / (self.upper_bound - self.lower_bound)
        output = x.unsqueeze(2).expand(B, 3, 3, H, W)
        output = torch.sum(output*self.ccm, dim=1) + self.bias
        output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
        output = torch.sum(self.slopes*self.relu(output - self.shifts), dim=2)
        
        output = self.conv2(output)
        
        return output


class DeepImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = DeconvDecoder(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        comp, attention_map, output = self.decoder(intermediates, image, mask)
        return {'images': comp,
                'attention_map': attention_map,
                'outputs' : output}


class DeepImageHarmonizationMhead(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonizationMhead, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = DeconvDecoderMhead(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, image, hsv, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output_rgb, output_h, output_s, output_v = self.decoder(intermediates, image, hsv, mask)

        return {'images': output_rgb, 'images_h': output_h, 'images_s' : output_s, 'images_v': output_v}


class DeepImageHarmonizationUpsample(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonizationUpsample, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = DeconvDecoderUpsample(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)
        self.upsample_layer = nn.Upsample(size=(768, 1024), mode='bicubic')
        self.get_filters = nn.Conv2d(32, 12, kernel_size=1) # apply affine transform on RGB


    def apply_filter(self, image, filters, map):
        '''
        image: (B, 3, H, W), filters: (B, 12, H, W), map: (B, 1, H, W)
        '''
        B, C, H, W = image.shape
        rot = filters[:, :9, :, :].view(B, 3, 3, H, W)
        bias = filters[:, 9:, :, :]
        output = image.unsqueeze(2).expand(B, 3, 3, H, W)
        output = torch.sum(rot*output, dim=1) + bias
        output = output * map + (1 - map) * image
        return output


    def forward(self, image, image_fullres, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        latent, attention_map = self.decoder(intermediates, image, mask)
        filters_lowres = self.get_filters(latent)

        filters_fullres = self.upsample_layer(filters_lowres)
        attention_map_fullres = self.upsample_layer(attention_map)

        output_lowres = self.apply_filter(image, filters_lowres, attention_map)
        output_fullres = self.apply_filter(image_fullres, filters_fullres, attention_map_fullres)

        return {'images': output_lowres, 
                'images_fullres' : output_fullres,
                'filters' : filters_lowres,
                'attention_map' : attention_map}

class DeepImageHarmonizationUpsampleNonlinear(nn.Module):
    
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode='', decoder_type='default',
        upsample_type='bicubic', apply_mode='bicubic', npts=5, rotNorm=True, up_size=(768, 1024), use_attn=True,
    ):
        super(DeepImageHarmonizationUpsampleNonlinear, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.apply_mode = apply_mode
        self.decoder = DeconvDecoderUpsample(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)
        self.upsample_type = upsample_type
        self.upsample_layer = nn.Upsample(size=up_size, mode='bicubic')
        if upsample_type == 'dbl': # Deep Bilateral Learning(HDRNet)
            self.slice_layer = Slice()
            self.guide_layer = CurveGuideNN()
        elif upsample_type == 'conv':
            self.refine_layer = nn.Conv2d(4, 1, kernel_size=1)
        self.use_attn = use_attn

        # apply affine transform and nonlnear transform on RGB
        self.npts = npts
        self.lower_bound = -2.11
        self.upper_bound = 2.66
        if upsample_type == 'dbl':
            self.get_filters = nn.Conv2d(32, 12*8, kernel_size=1)
        elif upsample_type == 'bicubic' or upsample_type == 'conv':
            self.get_filters = nn.Conv2d(32, 12+self.npts*3, kernel_size=1)
        
        if apply_mode == 'hsv':
            self.get_filters = nn.Conv2d(32, 12+self.npts+1+7, kernel_size=1)
        self.relu = nn.ReLU()


    def get_shifts(self, shape, lb, ub, npts, device):
        B, C, H, W = shape
        shifts_ = torch.linspace(0, 1 - 1.0/npts, npts, device=device)
        shifts_ = shifts_.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
        shifts_ = shifts_.expand(B, C, npts, H, W)
        return shifts_


    def apply_lum(self, image, filters, map):
        B, C, H, W = image.shape
        matrix = filters[:, :12, :, :]
        #matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
        rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
        bias = matrix[:, 9:12, :, :]

        slopes = filters[:, 12:12+self.npts, :, :].view(B, 1, self.npts, H, W)
        lum_min = filters[:, 12+self.npts:12+self.npts+1, :, :]
        lum_min = torch.clamp(lum_min, -0.2, 0.2)
        shifts = self.get_shifts((B, C, H, W), self.lower_bound, self.upper_bound, self.npts, image.device)

        sat_params = filters[:, 12+self.npts+1:, :, :]       

        # 把图片归一到[0,1]的区间
        output = (image - self.lower_bound) / (self.upper_bound - self.lower_bound)
        
        # V curve
        output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
        output = torch.sum(slopes*self.relu(output - shifts), dim=2) + lum_min
        output = torch.clamp(output, 0, 1)

        output = torch.clamp(output, 0, 1)
        # 把图片投影回原来的区间
        output = output * (self.upper_bound - self.lower_bound) + self.lower_bound

        comp = output * map + (1 - map) * image
        return comp

    def apply_sat(self, image, filters, map):
        B, C, H, W = image.shape
        matrix = filters[:, :12, :, :]
        #matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
        rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
        bias = matrix[:, 9:12, :, :]

        slopes = filters[:, 12:12+self.npts, :, :].view(B, 1, self.npts, H, W)
        lum_min = filters[:, 12+self.npts:12+self.npts+1, :, :]
        lum_min = torch.clamp(lum_min, -0.2, 0.2)
        shifts = self.get_shifts((B, C, H, W), self.lower_bound, self.upper_bound, self.npts, image.device)

        sat_params = filters[:, 12+self.npts+1:, :, :]       

        # 把图片归一到[0,1]的区间
        output = (image - self.lower_bound) / (self.upper_bound - self.lower_bound)
        
        # V curve
        output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
        output = torch.sum(slopes*self.relu(output - shifts), dim=2) + lum_min
        output = torch.clamp(output, 0, 1)

        output = apply_saturation_filter(output, sat_params)

        output = torch.clamp(output, 0, 1)
        # 把图片投影回原来的区间
        output = output * (self.upper_bound - self.lower_bound) + self.lower_bound

        comp = output * map + (1 - map) * image
        return comp

    def apply_filter(self, image, filters, map, return_raw=False):
        '''
        image: (B, 3, H, W), filters: (B, 12, H, W), map: (B, 1, H, W)
        '''
        if self.apply_mode == 'bicubic':
            B, C, H, W = image.shape
            matrix = filters[:, :12, :, :]
            matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
            rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
            bias = matrix[:, 9:12, :, :]

            slopes = filters[:, 12:, :, :].view(B, 3, self.npts, H, W)
            shifts = self.get_shifts((B, C, H, W), self.lower_bound, self.upper_bound, self.npts, image.device)        

            # 把图片归一到[0,1]的区间
            output = (image - self.lower_bound) / (self.upper_bound - self.lower_bound)
            # 用多个relu的加权和来近似一个非线性曲线
            output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
            output = torch.sum(slopes*self.relu(output - shifts), dim=2)
            # 线性变换，直接矩阵乘3x4的旋转矩阵
            output = output.unsqueeze(2).expand(B, 3, 3, H, W)
            output = torch.sum(rot*output, dim=1) + bias
            # 把图片投影回原来的区间
            output = output * (self.upper_bound - self.lower_bound) + self.lower_bound
        elif self.apply_mode == 'woRotNorm' and self.upsample_type == 'bicubic':
            B, C, H, W = image.shape
            matrix = filters[:, :12, :, :]
            #matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
            rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
            bias = matrix[:, 9:12, :, :]

            slopes = filters[:, 12:, :, :].view(B, 3, self.npts, H, W)
            shifts = self.get_shifts((B, C, H, W), self.lower_bound, self.upper_bound, self.npts, image.device)        

            # 把图片归一到[0,1]的区间
            output = (image - self.lower_bound) / (self.upper_bound - self.lower_bound)
            # 用多个relu的加权和来近似一个非线性曲线
            output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
            output = torch.sum(slopes*self.relu(output - shifts), dim=2)
            output = torch.clip(output, 0, 1)
            output = output * (self.upper_bound - self.lower_bound) + self.lower_bound
            # 线性变换，直接矩阵乘3x4的旋转矩阵
            output = output.unsqueeze(2).expand(B, 3, 3, H, W)
            output = torch.sum(rot*output, dim=1) + bias
            # 把图片投影回原来的区间
        elif self.apply_mode == 'dbl':
            B, C, H, W = image.shape
            rot = filters[:, :9, :, :].view(B, 3, 3, H, W)
            bias = filters[:, 9:12, :, :]
            output = image.unsqueeze(2).expand(B, 3, 3, H, W)
            output = torch.sum(rot*output, dim=1) + bias
        elif self.apply_mode == 'hsv':
            B, C, H, W = image.shape
            matrix = filters[:, :12, :, :]
            #matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
            rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
            bias = matrix[:, 9:12, :, :]

            slopes = filters[:, 12:12+self.npts, :, :].view(B, 1, self.npts, H, W)
            lum_min = filters[:, 12+self.npts:12+self.npts+1, :, :]
            lum_min = torch.clamp(lum_min, -0.2, 0.2)
            shifts = self.get_shifts((B, C, H, W), self.lower_bound, self.upper_bound, self.npts, image.device)

            sat_params = filters[:, 12+self.npts+1:, :, :]       

            # 把图片归一到[0,1]的区间
            output = (image - self.lower_bound) / (self.upper_bound - self.lower_bound)
            
            # V curve
            #st = time.time()
            output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
            output = torch.sum(slopes*self.relu(output - shifts), dim=2) + lum_min
            output = torch.clamp(output, 0, 1)
            #ed = time.time()
            #V_time = ed - st
            # S curve
            #st = time.time()
            output = apply_saturation_filter(output, sat_params)
            #ed = time.time()
            #S_time = ed - st
            # H curve
            #st = time.time()
            output = output.unsqueeze(2).expand(B, 3, 3, H, W)
            output = torch.sum(rot*output, dim=1) + bias
            #ed = time.time()
            #H_time = ed - st
            #print('%.3f, %.3f, %.3f' % (V_time, S_time, H_time))
            output = torch.clamp(output, 0, 1)
            # 把图片投影回原来的区间
            output = output * (self.upper_bound - self.lower_bound) + self.lower_bound

        if return_raw:
            comp = output * map + (1 - map) * image
            return comp, output
        else:
            comp = output * map + (1 - map) * image
            return comp


    def slice(self, filters, image, use_cpu=False):
        gpu_device = filters.get_device()
        if use_cpu:
            guide_map = self.guide_layer(image.to('cpu'))
            coeff = self.slice_layer(filters.to('cpu'), guide_map)
            coeff = coeff.to(gpu_device)
        else: 
            guide_map = self.guide_layer(image)
            coeff = self.slice_layer(filters, guide_map)
            
        return coeff


    def forward(self, image, image_fullres, mask, backbone_features=None, test=False, rois=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        latent, attention_map = self.decoder(intermediates, image, mask)

        filters_lowres = self.get_filters(latent)
        if self.upsample_type == 'bicubic':
            filters_fullres = self.upsample_layer(filters_lowres)
        elif self.upsample_type == 'conv':
            filters_fullres = self.upsample_layer(filters_lowres)
        elif self.upsample_type == 'dbl':
            filters_fullres = self.slice(filters_lowres, image_fullres)
            filters_lowres = self.slice(filters_lowres, image)
            
        output_lowres = self.apply_filter(image, filters_lowres, attention_map if self.use_attn else mask)

        if test:
            return {
                'images' : output_lowres,
                'filters_fullres' : filters_fullres,
            }

        #print(filters_fullres.shape, filters_lowres.shape, image.shape)
        
        attention_map_fullres = self.upsample_layer(attention_map)
        mask_fullres = self.upsample_layer(mask)
        #if self.upsample_type == 'conv':
        #    attention_map_fullres = torch.sigmoid(3.0*self.refine_layer(torch.cat([image_fullres, attention_map_fullres], axis=1)))

        output_fullres = self.apply_filter(image_fullres, filters_fullres,  attention_map_fullres if self.use_attn else mask_fullres)

        fake_fgs = None
        real_fgs = None

        #print(rois)
        if rois is not None:
            B = image.size(0)
            bi = torch.Tensor(np.arange(0, B)).unsqueeze(1).to(rois.device)
            rois = torch.cat([bi, rois], axis=1)
            fake_fgs = tv_roi_align(output_fullres, rois, output_size=128)
            real_fgs = tv_roi_align(image_fullres, rois, output_size=128)

        #print(fake_fgs.shape, real_fgs.shape)        

        return {'images': output_lowres, 
                'images_fullres' : output_fullres,
                'filters' : filters_lowres,
                'attention_map' : attention_map,
                'attention_map_fullres' : attention_map_fullres,
                'fake_fgs' : fake_fgs,
                'real_fgs' : real_fgs}

class DeepImageHarmonizationUpsampleNonlinearTrans(nn.Module):

    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode='', decoder_type='default'
    ):
        super(DeepImageHarmonizationUpsampleNonlinearTrans, self).__init__()
        print('this is DeepImageHarmonizationUpsampleNonlinearTrans')
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode, pad_mode='final',
        )
        vit_config = get_r50_b16_config()
        self.transformer = Transformer(vit_config, 32, False, 128)
        self.transformer.apply(LRMult(0.1))
        self.transformer.embeddings.patch_embeddings.apply(LRMult(1.0))
        self.decoder = DeconvDecoderUpsampleTrans(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)
        self.upsample_layer = nn.Upsample(size=(768, 1024), mode='bicubic')
        # apply affine transform and nonlnear transform on RGB
        self.npts = 5
        self.lower_bound = -2.11
        self.upper_bound = 2.66
        self.get_filters = nn.Conv2d(32, 12 + 3*self.npts, kernel_size=1)
        self.relu = nn.ReLU()


    def get_shifts(self, shape, lb, ub, npts, device):
        B, C, H, W = shape
        shifts_ = torch.linspace(0, 1 - 1.0/npts, npts, device=device)
        shifts_ = shifts_.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
        shifts_ = shifts_.expand(B, C, npts, H, W)
        return shifts_

    def apply_filter(self, image, filters, map):
        '''
        image: (B, 3, H, W), filters: (B, 12, H, W), map: (B, 1, H, W)
        '''
        B, C, H, W = image.shape
        matrix = filters[:, :12, :, :]
        matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
        rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
        bias = matrix[:, 9:12, :, :]

        slopes = filters[:, 12:, :, :].view(B, 3, self.npts, H, W)
        shifts = self.get_shifts((B, C, H, W), self.lower_bound, self.upper_bound, self.npts, image.device)        

        # 把图片归一到[0,1]的区间
        output = (image - self.lower_bound) / (self.upper_bound - self.lower_bound)
        # 用多个relu的加权和来近似一个非线性曲线
        output = output.unsqueeze(2).expand(B, 3, self.npts, H, W)
        output = torch.sum(slopes*self.relu(output - shifts), dim=2)
        # 线性变换，直接矩阵乘3x4的旋转矩阵
        output = output.unsqueeze(2).expand(B, 3, 3, H, W)
        output = torch.sum(rot*output, dim=1) + bias
        # 把图片投影回原来的区间
        output = output * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        output = output * map + (1 - map) * image
        return output


    def forward(self, image, image_fullres, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        trans_embed = None
        for _, skips in enumerate(intermediates):
            if _ == 0:
                skips, attn_weights, features = self.transformer(skips)
                B, n_patch, hidden = skips.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
                h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
                skips = skips.permute(0, 2, 1)
                skips = skips.contiguous().view(B, hidden, h, w)
                trans_embed = skips
        
        latent, attention_map = self.decoder(intermediates, image, mask, trans_embed)

        filters_lowres = self.get_filters(latent)
        filters_fullres = self.upsample_layer(filters_lowres)
        attention_map_fullres = self.upsample_layer(attention_map)

        output_lowres = self.apply_filter(image, filters_lowres, attention_map)
        output_fullres = self.apply_filter(image_fullres, filters_fullres, attention_map_fullres)

        return {'images': output_lowres, 
                'images_fullres' : output_fullres,
                'filters' : filters_lowres,
                'attention_map' : attention_map}



