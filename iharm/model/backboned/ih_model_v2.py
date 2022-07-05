import torch
import torch.nn as nn

from iharm.model.ops import SimpleInputFusion, ScaleLayer

class IHModelWithBackboneHSL(nn.Module):
    def __init__(
        self,
        model, backbone,
        downsize_backbone_input=False,
        mask_fusion='sum',
        backbone_conv1_channels=64,
    ):

        super(IHModelWithBackboneHSL, self).__init__()
        self.downsize_backbone_input = downsize_backbone_input
        self.mask_fusion = mask_fusion

        self.backbone = backbone
        self.model = model

        if mask_fusion == 'rgb':
            self.fusion = SimpleInputFusion()
        elif mask_fusion == 'sum':
            self.mask_conv = nn.Sequential(
                nn.Conv2d(1, backbone_conv1_channels, kernel_size=3, stride=2, padding=1, bias=True),
                ScaleLayer(init_value=0.1, lr_mult=1)
            )

    def forward(self, image, image_fullres, mask, mask_fullres, test=False):
        backbone_image = image
        backbone_mask = torch.cat((mask, 1.0 - mask), dim=1)
        if self.downsize_backbone_input:
            backbone_image = nn.functional.interpolate(
                backbone_image, scale_factor=0.5,
                mode='bilinear', align_corners=True
            )
            backbone_mask = nn.functional.interpolate(
                backbone_mask, backbone_image.size()[2:],
                mode='bilinear', align_corners=True
            )
        backbone_image = (
            self.fusion(backbone_image, backbone_mask[:, :1])
            if self.mask_fusion == 'rgb' else
            backbone_image
        )
        backbone_mask_features = self.mask_conv(backbone_mask[:, :1]) if self.mask_fusion == 'sum' else None
        backbone_features = self.backbone(backbone_image, backbone_mask, backbone_mask_features)

        output = self.model(image, image_fullres, mask, mask_fullres, backbone_features, test)
        return output