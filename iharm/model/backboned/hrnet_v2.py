import torch.nn as nn

from iharm.model.modeling.hrnet_ocr import HighResolutionNet
from iharm.model.backboned.ih_model_v2 import IHModelWithBackboneHSL
from iharm.model.modifiers import LRMult
from iharm.model.modeling.basic_blocks import MaxPoolDownSize

class HRNetIHModelHSL(IHModelWithBackboneHSL):
    def __init__(
        self,
        base_config,
        downsize_hrnet_input=False, mask_fusion='sum',
        lr_mult=0.1, cat_hrnet_outputs=True, pyramid_channels=-1,
        ocr=64, width=18, small=True,
        mode='cat',
        **base_kwargs
    ):
        params = base_config['params']
        params.update(base_kwargs)
        depth = params['depth']

        backbone = HRNetBB(
            cat_outputs=cat_hrnet_outputs,
            pyramid_channels=pyramid_channels,
            pyramid_depth=min(depth - 2 if not downsize_hrnet_input else depth - 3, 4),
            width=width, ocr=ocr, small=small,
            lr_mult=lr_mult,
        )

        params.update(dict(
            backbone_from=3 if downsize_hrnet_input else 2,
            backbone_channels=backbone.output_channels,
            backbone_mode=mode
        ))
        base_model = base_config['model'](**params)

        super(HRNetIHModelHSL, self).__init__(base_model, backbone, downsize_hrnet_input, mask_fusion)



class HRNetBB(nn.Module):
    def __init__(
        self,
        cat_outputs=True,
        pyramid_channels=256, pyramid_depth=4,
        width=18, ocr=64, small=True,
        lr_mult=0.1,
    ):
        super(HRNetBB, self).__init__()
        self.cat_outputs = cat_outputs
        self.ocr_on = ocr > 0 and cat_outputs
        self.pyramid_on = pyramid_channels > 0 and cat_outputs

        self.hrnet = HighResolutionNet(width, 2, ocr_width=ocr, small=small)
        self.hrnet.apply(LRMult(lr_mult))
        if self.ocr_on:
            self.hrnet.ocr_distri_head.apply(LRMult(1.0))
            self.hrnet.ocr_gather_head.apply(LRMult(1.0))
            self.hrnet.conv3x3_ocr.apply(LRMult(1.0))

        hrnet_cat_channels = [width * 2 ** i for i in range(4)]
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        elif self.ocr_on:
            self.output_channels = [ocr * 2]
        elif self.cat_outputs:
            self.output_channels = [sum(hrnet_cat_channels)]
        else:
            self.output_channels = hrnet_cat_channels

        if self.pyramid_on:
            downsize_in_channels = ocr * 2 if self.ocr_on else sum(hrnet_cat_channels)
            self.downsize = MaxPoolDownSize(downsize_in_channels, pyramid_channels, pyramid_channels, pyramid_depth)

    def forward(self, image, mask, mask_features):
        if not self.cat_outputs:
            return self.hrnet.compute_hrnet_feats(image, mask_features, return_list=True)

        outputs = list(self.hrnet(image, mask, mask_features))
        if self.pyramid_on:
            outputs = self.downsize(outputs[0])
        return outputs

    def load_pretrained_weights(self, pretrained_path):
        self.hrnet.load_pretrained_weights(pretrained_path)
