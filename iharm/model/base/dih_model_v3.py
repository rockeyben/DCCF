from numpy.lib.arraypad import _pad_dispatcher, _pad_simple
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial
import time
import kornia

from iharm.model.modeling.unet import UNetEncoder, UNetDecoderUpsample
from iharm.model.ops import MaskedChannelAttention
from iharm.model.base.ssam_model import SpatialSeparatedAttention
from iharm.model.modeling.conv_autoencoder import ConvEncoder, DeconvDecoderUpsample
from iharm.model.modeling.basic_blocks import ConvBlock
from iharm.model.ps_filters import apply_saturation_filter, get_hue_channel, get_lum_channel, get_sat_channel, luminosity_blend, rgb_selective_color,\
        rgb_to_hsv, hsv_to_rgb, get_rgb_hue_channle

def debug(t):
    print(torch.min(t), torch.max(t))


class DeepImageHarmonizationUpsampleHSL_V3(nn.Module):
    
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode='',
        npts=5, norm_rotation=True, up_size=(768, 1024), use_disentangle=False,
        use_attn=False, h_method='xinzhi_hsl', s_method='xinzhi', tune_method='default', use_refine=False, use_detach=True,
        use_tanh=True, use_hr=True, hue_filters=9, use_hr_inter=False, hue_norm_rotation=True,
        use_blur_L=False, detach_refiner=True, use_dbl=False, clamp_sat_modify=False, guide_method='NN', backbone_type='idih', attention_mid_k=2.0,
    ):
        super(DeepImageHarmonizationUpsampleHSL_V3, self).__init__()
        self.depth = depth
        
        if backbone_type == 'idih':
            self.encoder = ConvEncoder(
                depth, ch,
                norm_layer, batchnorm_from, max_channels,
                backbone_from, backbone_channels, backbone_mode
            )
            self.decoder = DeconvDecoderUpsample(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)
        elif backbone_type == 'ssam':
            print('depth', depth)
            self.encoder = UNetEncoder(
                depth, ch,
                norm_layer, batchnorm_from, max_channels,
                backbone_from, backbone_channels, backbone_mode
            )
            self.decoder =  UNetDecoderUpsample(
                depth, self.encoder.block_channels,
                norm_layer,
                attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
                attend_from=attend_from,
                image_fusion=image_fusion
            )

        self.norm_rotation = norm_rotation
        self.upsample_layer = nn.Upsample(size=up_size, mode='bicubic')
        self.record = use_disentangle
        self.use_attn = use_attn
        self.h_method = h_method
        self.s_method = s_method
        self.tune_method = tune_method
        self.use_refine = use_refine
        self.detach = use_detach
        self.use_tanh = use_tanh
        self.use_hr = use_hr
        self.use_hr_inter = use_hr_inter 
        self.hue_norm_rotation = hue_norm_rotation
        self.use_blur_L = use_blur_L
        self.detach_refiner = detach_refiner
        self.clamp_sat_modify = clamp_sat_modify
        self.guide_method = guide_method

        self.npts = npts
        self.hue_filters_num = hue_filters
        self.lum_filters_num = 1 + self.npts
        self.sat_filters_num = 1
        
        self.hue_base = 0
        self.hue_end = self.hue_filters_num
        self.lum_base = self.hue_end
        self.lum_end = self.hue_end + self.lum_filters_num
        self.sat_base = self.lum_end
        self.sat_end = self.sat_base + self.sat_filters_num
        self.refine_base = self.sat_end
        self.refine_end = self.refine_base + 12

        self.get_filters = nn.Conv2d(32, self.hue_filters_num+self.lum_filters_num+self.sat_filters_num, kernel_size=1)
        if self.use_refine:
            self.get_refine = nn.Conv2d(32, 12, kernel_size=1)

        self.relu = nn.ReLU()

        self.mean = torch.Tensor([.485, .456, .406]) ##TODO normalize
        self.std = torch.Tensor([.229, .224, .225])

        self.mean = self.mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = self.std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def get_shifts(self, shape, npts, device):
        B, C, H, W = shape
        shifts_ = torch.linspace(0, 1 - 1.0/npts, npts, device=device)
        shifts_ = shifts_.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
        shifts_ = shifts_.expand(B, C, npts, H, W)
        return shifts_

    def get_hsv(self, image):
        
        image = self.map_to_01(image)
        # print(torch.max(image))
        image_hsv = rgb_to_hsv(image)
        return image_hsv

    def map_to_01(self, image):
        # output: [0, 1]
        self.mean = self.mean.to(image.device)
        self.std = self.std.to(image.device)
        return image * self.std + self.mean

    def map_to_norm(self, image):
        # output: [-2.11, 2.16], 和经过dataset正则化后的图片一致
        self.mean = self.mean.to(image.device)
        self.std = self.std.to(image.device)
        return (image - self.mean) / self.std



    def apply_lum(self, image, filters, input_hsv=None):
        # input image must be in [0,1]
        
        B, C, H, W = image.shape

        slopes = filters[:, self.lum_base:self.lum_end-1, :, :].view(B, 1, self.npts, H, W)
        lum_min = filters[:, self.lum_end-1:self.lum_end, :, :]
        shifts = self.get_shifts((B, 1, H, W), self.npts, image.device)
        if self.use_tanh:
            lum_min = torch.tanh(lum_min) * 0.2
            slopes = torch.tanh(slopes)
        else:
            lum_min = torch.clamp(lum_min, -0.2, 0.2)

        # V curve
        if self.tune_method == 'default':
            output = image.unsqueeze(2).expand(B, 3, self.npts, H, W)
            output = torch.sum(slopes*self.relu(output - shifts), dim=2) + lum_min
            output = torch.clamp(output, 0, 1)
        elif self.tune_method == 'merge':
            #input_hsv = rgb_to_hsv(image)
            output_hsv = input_hsv.clone()
            input_v = input_hsv[:, 2:3, :, :]
            #if self.use_blur_L:
            #    input_v = kornia.gaussian_blur2d(input_v, (5,5), (1.5, 1.5))
            output_v = input_v.unsqueeze(2).expand(B, 1, self.npts, H, W)
            output_v = torch.sum(slopes*self.relu(output_v - shifts), dim=2) + lum_min
            output_v = torch.clamp(output_v, 0, 1)
            output_hsv[:, 2:3, :, :] = output_v
            output = hsv_to_rgb(output_hsv)
            output = torch.clamp(output, 0, 1)

        return output, output_hsv

    def apply_sat(self, image, filters, input_hsv=None):
        # input image must be in [0,1]
        sat_params = filters[:, self.sat_base:self.sat_end, :, :]
        if self.use_tanh:
            sat_params = torch.tanh(sat_params)
        else:
            if self.clamp_sat_modify:
                sat_params = torch.clamp(sat_params, -1, 1)  
            else:
                sat_params = torch.clamp(sat_params, 0, 1)  
        # S curve, import from ihamr.ps_filter
        output = apply_saturation_filter(image, sat_params)
        output = torch.clamp(output, 0, 1)

        if self.tune_method == 'merge':
            #input_hsv = rgb_to_hsv(image)
            output_hsv = input_hsv.clone()
            output_s = get_sat_channel(output, method='default')
            output_hsv[:, 1:2, :, :] = output_s
            output = hsv_to_rgb(output_hsv)
            output = torch.clamp(output, 0, 1)
            
        return output, output_hsv

    def apply_hue(self, image, filters, input_hsv=None):
        # input image must be in [0,1]
        B, C, H, W = image.shape
        matrix = filters[:, self.hue_base:self.hue_end, :, :]
        if self.hue_norm_rotation:
            matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
        rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
        output = image.unsqueeze(2).expand(B, 3, 3, H, W)
        if self.hue_filters_num == 12:
            bias = matrix[:, 9:12, :, :]
            output = torch.sum(rot*output, dim=1) + bias
        elif self.hue_filters_num == 9:
            output = torch.sum(rot*output, dim=1)
        output = torch.clamp(output, 0, 1)
        if self.tune_method == 'merge':
            #input_hsv = rgb_to_hsv(image)
            #input_hsv = input_hsv.detach()
            output_hsv = input_hsv.clone()
            output_h = get_rgb_hue_channle(output)
            output_hsv[:, 0:1, :, :] = output_h
            output = hsv_to_rgb(output_hsv)
            output = torch.clamp(output, 0, 1)
        return output, output_hsv
    

    def apply_refine(self, image, filters):
        B, C, H, W = image.shape
        matrix = filters
        if self.norm_rotation:
            matrix = F.normalize(matrix, p=2, dim=1) # 正则化，把旋转矩阵做L2正则
        rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)
        bias = matrix[:, 9:12, :, :]

        output = image.unsqueeze(2).expand(B, 3, 3, H, W)
        output = torch.sum(rot*output, dim=1) + bias
        output = torch.clamp(output, 0, 1)
        return output


    def get_hue(self, image_rgb, method=None):
        if not method:
            method = self.h_method
        image_rgb = self.map_to_01(image_rgb)
        image_bgr = torch.flip(image_rgb, [1])
        return get_hue_channel(image_bgr, method=method)

    def get_sat(self, image_rgb, method=None):
        if not method:
            method = self.s_method
        image_rgb = self.map_to_01(image_rgb)
        image_bgr = torch.flip(image_rgb, [1])
        return get_sat_channel(image_bgr, method=method)

    def get_lum(self, image_rgb, use_blur=False):
        use_blur = self.use_blur_L
        image_rgb = self.map_to_01(image_rgb)
        image_bgr = torch.flip(image_rgb, [1])
        return get_lum_channel(image_bgr, use_blur=use_blur)


    def apply_filter(self, image, filters, refine_filters=None, record=False, detach=False):
        '''
        image: (B, 3, H, W), filters: (B, C, H, W), map: (B, 1, H, W)
        '''
        inter_results = dict()
        output = self.map_to_01(image)

        output_hsv = rgb_to_hsv(output)

        image_bgr = torch.flip(output, [1])
        if record:
            inter_results['input_rgb'] = output
            inter_results['input_hsv'] = output_hsv
            inter_results['input_Hmap'] = get_hue_channel(image_bgr, method=self.h_method)
            inter_results['input_Smap'] = get_sat_channel(image_bgr, method=self.s_method)
            inter_results['input_Lmap'] = get_lum_channel(image_bgr)
            pass
        # V curve
        if detach:
            output, output_hsv = self.apply_lum(output.detach(), filters, input_hsv=output_hsv.detach())
        else:
            output, output_hsv = self.apply_lum(output, filters, input_hsv=output_hsv)

        if record:
            stage1_bgr = torch.flip(output, [1])
            inter_results['stage1_filter'] = filters[:, self.lum_base:self.lum_end, :, :]
            inter_results['stage1_Lmap'] = get_lum_channel(stage1_bgr, use_blur=self.use_blur_L)
            if self.tune_method == 'default':
                inter_results['stage1_Smap'] = get_sat_channel(stage1_bgr, method=self.s_method)
                inter_results['stage1_Hmap'] = get_hue_channel(stage1_bgr, method=self.h_method)
            inter_results['stage1_output'] = self.map_to_norm(output)
            inter_results['stage1_output_hsv'] = output_hsv

        # S curve
        if detach:
            output, output_hsv = self.apply_sat(output.detach(), filters, input_hsv=output_hsv.detach())
        else:
            output, output_hsv = self.apply_sat(output, filters, input_hsv=output_hsv)
        if record:
            stage2_bgr = torch.flip(output, [1])
            inter_results['stage2_Smap'] = get_sat_channel(stage2_bgr, self.s_method)
            inter_results['stage2_filter'] = filters[:, self.sat_base:self.sat_end, :, :]
            if self.tune_method == 'default':
                inter_results['stage2_Lmap'] = get_lum_channel(stage2_bgr)
                inter_results['stage2_Hmap'] = get_hue_channel(stage2_bgr, method=self.h_method)
            inter_results['stage2_output'] = self.map_to_norm(output)
            inter_results['stage2_output_hsv'] = output_hsv
        # H curve
        if detach:
            output, output_hsv  = self.apply_hue(output.detach(), filters, input_hsv=output_hsv.detach())
        else:
            output, output_hsv  = self.apply_hue(output, filters, input_hsv=output_hsv)

        if record:
            stage3_bgr = torch.flip(output, [1])
            inter_results['stage3_Hmap'] = get_hue_channel(stage3_bgr, method=self.h_method)
            inter_results['stage3_filter'] = filters[:, self.hue_base:self.hue_end, :, :]
            if self.tune_method == 'default':
                inter_results['stage3_Lmap'] = get_lum_channel(stage3_bgr)
                inter_results['stage3_Smap'] = get_sat_channel(stage3_bgr, method=self.s_method)
            inter_results['stage3_output_hsv'] = output_hsv
        inter_results['stage3_output'] = self.map_to_norm(output)
        
        if self.use_refine:
            if self.detach_refiner:
                output = self.apply_refine(output.detach(), refine_filters)
            else:
                output = self.apply_refine(output, refine_filters)

        output = self.map_to_norm(output)

        return output, inter_results

    def forward(self, image, image_fullres, mask, mask_fullres=None, backbone_features=None, test=False):
        record=self.record
        H = image_fullres.size(2)
        W = image_fullres.size(3)
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        latent, attention_map = self.decoder(intermediates, image, mask) # attn map 不再使用

        filters_latent = self.get_filters(latent)
        refine_latent = None
        if self.use_refine:
            refine_latent = self.get_refine(latent)
        
        filters_lowres = filters_latent
        refine_filters = refine_latent
        output_lowres, inter_result_lowres = self.apply_filter(image, filters_lowres, refine_filters=refine_filters, record=record, detach=self.detach)

        if self.use_attn:
            output_lowres = output_lowres * attention_map + image * (1 - attention_map)
        else:
            output_lowres = output_lowres * mask + image * (1 - mask)

        # final_hsv = self.get_hsv(self.map_to_01(output_lowres))
        # inter_result_lowres['final_hsv'] = final_hsv
        
        if test:
            return {
                'images' : output_lowres,
                'filters_latent' : filters_latent,
                'refine_latent' : refine_latent,
                'filters' : filters_lowres,
                'refine_filters' : refine_filters,
                'attention_map' : attention_map,
                'inter_result' : inter_result_lowres}

        outputs = dict()
        outputs['images'] = output_lowres
    
        outputs['stage1_filter'] = inter_result_lowres['stage1_filter']
        outputs['stage2_filter'] = inter_result_lowres['stage2_filter']
        outputs['stage3_filter'] = inter_result_lowres['stage3_filter']
        outputs['stage3_output'] = inter_result_lowres['stage3_output'] * mask + image * (1 - mask)

    
        if self.use_hr:
            filters_fullres = F.upsample(filters_lowres, size=(H, W), mode='bicubic')
            refine_filters_fullres = None
            if self.use_refine:
                refine_filters_fullres = F.upsample(refine_filters, size=(H, W), mode='bicubic')
            
            output_fullres, inter_result_fullres = self.apply_filter(image_fullres, filters_fullres, refine_filters=refine_filters_fullres, record=self.use_hr_inter, detach=self.detach)
            if self.use_attn:
                attention_map_fullres = F.upsample(attention_map, size=(H, W), mode='bicubic')
                output_fullres = output_fullres * attention_map_fullres + \
                                image_fullres * (1 - attention_map_fullres)
            else:
                output_fullres = output_fullres * mask_fullres + image_fullres * (1 - mask_fullres)
            outputs['images_fullres'] = output_fullres
            outputs['stage3_output_fullres'] = inter_result_fullres['stage3_output'] * mask_fullres + image_fullres * (1 - mask_fullres)
        
        if record:
            if self.tune_method == 'default':
                for key, value in inter_result_lowres.items():
                    if 'Lmap' in key:
                        outputs[key] = value * mask + inter_result_lowres['input_Lmap'] * (1 - mask)
                    elif 'Smap' in key:
                        outputs[key] = value * mask + inter_result_lowres['input_Smap'] * (1 - mask)                
                    elif 'Hmap' in key:
                        outputs[key] = value * mask + inter_result_lowres['input_Hmap'] * (1 - mask)
            else:
                for key, value in inter_result_lowres.items():
                    if 'Lmap' in key:
                        outputs[key] = value
                    elif 'Smap' in key:
                        outputs[key] = value                
                    elif 'Hmap' in key:
                        outputs[key] = value
                if self.use_hr_inter:
                    outputs['stage1_Lmap_fullres'] = inter_result_fullres['stage1_Lmap']
                    outputs['stage2_Smap_fullres'] = inter_result_fullres['stage2_Smap']
                    outputs['stage3_Hmap_fullres'] = inter_result_fullres['stage3_Hmap']
            if test:
                outputs['stage1_filter'] = inter_result_lowres['stage1_filter']
                outputs['stage2_filter'] = inter_result_lowres['stage2_filter']
                outputs['stage3_filter'] = inter_result_lowres['stage3_filter']
                outputs['input_rgb'] = inter_result_lowres['input_rgb']
                outputs['input_hsv'] = inter_result_lowres['input_hsv']
                outputs['stage1_output'] = inter_result_lowres['stage1_output'] * mask + image * (1 - mask)
                outputs['stage2_output'] = inter_result_lowres['stage2_output'] * mask + image * (1 - mask)
                outputs['stage3_output'] = inter_result_lowres['stage3_output'] * mask + image * (1 - mask)

        return outputs 
