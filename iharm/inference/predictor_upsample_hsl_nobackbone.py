import torch
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor


class PredictorUpsampleHSLNoBackbone(object):
    def __init__(self, net, device, with_flip=False,
                 mean=(.485, .456, .406), std=(.229, .224, .225)):
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()
        
        if hasattr(net, 'depth'):
            size_divisor = 2 ** (net.depth + 1)
        else:
            size_divisor = 1

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)

        self.transforms = [
            ToTensor(self.device),
            NormalizeTensor(mean, std, self.device),
        ]

        if with_flip:
            self.transforms.append(AddFlippedTensor())

    def predict(self, image, image_fullres, raw_image, target_image, target_image_fullres, mask, mask_fullres, return_numpy=True, use_dbl=False):
        rc = False
        with torch.no_grad():
            raw_mask = mask
            # print(image.shape)
            for transform in self.transforms:
                image, mask = transform.transform(image, mask)
            input_mask = mask
            for transform in self.transforms:
                image_fullres, mask = transform.transform(image_fullres, raw_mask)
            for transform in self.transforms:
                raw_image, mask = transform.transform(raw_image, raw_mask)
            for transform in self.transforms:
                target_image, mask = transform.transform(target_image, raw_mask)
            for transform in self.transforms:
                target_image_fullres, mask = transform.transform(target_image_fullres, raw_mask)

            B, _, H, W = raw_image.shape
            upsample_layer = torch.nn.Upsample(size=(H, W), mode='bicubic').to(self.device)
            inter_result = dict()
            self.net.record=False
            output = self.net(image, image_fullres, input_mask, mask_fullres, test=True)
            predicted_image = output['images']
            filters = output['filters']
            refine_filters = output['refine_filters']
            
            filters_fullres = upsample_layer(filters)
            refine_filters_fullres = upsample_layer(refine_filters)
                
            output_fullres, inter_fullres = self.net.apply_filter(raw_image, filters_fullres, refine_filters=refine_filters_fullres, record=rc)
            
            stage3_fullres = inter_fullres['stage3_output']
                
            if self.net.use_attn:
                attn_map = output['attention_map']
                attn_map_fullres = upsample_layer(attn_map)
                output_fullres = output_fullres * attn_map_fullres + raw_image * (1 - attn_map_fullres)

            
            # inter_result = output['inter_result']
            # inter_result['gt_Lmap'] = self.net.get_lum(target_image)
            # inter_result['gt_Smap'] = self.net.get_sat(target_image)
            # inter_result['gt_Hmap'] = self.net.get_hue(target_image, method='xinzhi_hsl')
            # inter_result['stage3_Hmap'] = self.net.get_hue(inter_result['stage3_Hmap'], method='xinzhi_hsl')
            # inter_result['stage3_output'] = self.net.map_to_01(inter_result['stage3_output'])
            # inter_result['stage3_output_fullres'] = self.net.map_to_01(stage3_fullres)
            
            if rc:
                stage2_fullres = inter_fullres['stage2_output']
                stage1_fullres = inter_fullres['stage1_output']
                inter_result['gt_Lmap_fullres'] = self.net.get_lum(target_image_fullres)
                inter_result['gt_Smap_fullres'] = self.net.get_sat(target_image_fullres)
                inter_result['gt_Hmap_fullres'] = self.net.get_hue(target_image_fullres)

                inter_result['stage1_output_fullres'] = self.net.map_to_01(stage2_fullres)
                inter_result['stage2_output_fullres'] = self.net.map_to_01(stage1_fullres)
                inter_result['input_Lmap_fullres'] = inter_fullres['input_Lmap']
                inter_result['input_Smap_fullres'] = inter_fullres['input_Smap']
                inter_result['input_Hmap_fullres'] = inter_fullres['input_Hmap']
                inter_result['stage1_Lmap_fullres'] = inter_fullres['stage1_Lmap']
                inter_result['stage2_Smap_fullres'] = inter_fullres['stage2_Smap']
                inter_result['stage3_Hmap_fullres'] = inter_fullres['stage3_Hmap']

                inter_result['input_rgb_fullres'] = inter_fullres['input_rgb']
                inter_result['input_hsv_fullres'] = inter_fullres['input_hsv']
                inter_result['stage1_output_hsv_fullres'] = inter_fullres['stage1_output_hsv']
                inter_result['stage2_output_hsv_fullres'] = inter_fullres['stage2_output_hsv']
                inter_result['stage1_filter_fullres'] = inter_fullres['stage1_filter']
                inter_result['stage2_filter_fullres'] = inter_fullres['stage2_filter']
                inter_result['stage3_filter_fullres'] = inter_fullres['stage3_filter']
            

            for transform in reversed(self.transforms):
                predicted_image = transform.inv_transform(predicted_image)

            for transform in reversed(self.transforms):
                output_fullres = transform.inv_transform(output_fullres)

            predicted_image = torch.clamp(predicted_image, 0, 255)
            output_fullres = torch.clamp(output_fullres, 0, 255)
            # print(predicted_image.shape, output_fullres.shape)

        if return_numpy:
            return predicted_image.cpu().numpy(), output_fullres.cpu().numpy(), inter_result    
        else:
            return predicted_image, output_fullres, inter_result
