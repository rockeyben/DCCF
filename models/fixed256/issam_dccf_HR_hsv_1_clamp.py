from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDatasetUpsample
from iharm.data.hdataset import HDatasetUpsample
from iharm.data.transforms import HCompose
from iharm.engine.upsample_dccf_backbone_trainer import UpsampleDCCFTrainer
from iharm.mconfigs import BMCONFIGS
from iharm.model import initializer
from iharm.model.base import DeepImageHarmonizationUpsampleHSL_V3
from iharm.model.losses import MaskWeightedCosine, MaskWeightedMSE, MaskWeightedTV
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric, MSEMetric
from iharm.utils.log import logger


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (768, 1024)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])

    ccfg = BMCONFIGS['dccf_improved_ssam256_HR_clamp']
    ccfg['params']['up_size'] = model_cfg.crop_size
    model = DeepImageHarmonizationUpsampleHSL_V3(**ccfg['params'])
    model_cfg.h_method = ccfg['params']['h_method']

    # model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    # model.backbone.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18_SMALL)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))

    return model, model_cfg

def train(model, cfg, model_cfg, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    wl = 1.0
    blur = False
    use_hr = BMCONFIGS['dccf_improved_ssam256_HR_clamp']['params']['use_hr'] # True

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeightedMSE(min_area=100)
    loss_cfg.pixel_loss_weight = 1.0

    loss_cfg.pixel_loss_fullres = MaskWeightedMSE(min_area=1000,  pred_name='images_fullres',
            gt_image_name='target_images_fullres', gt_mask_name='masks_fullres')
    loss_cfg.pixel_loss_fullres_weight = wl * 5.0

    loss_cfg.stage1_l_loss = MaskWeightedMSE(min_area=100, pred_name='stage1_Lmap',
            gt_image_name='gt_Lmap', gt_mask_name='masks')
    loss_cfg.stage1_l_loss_weight = wl * 1

    loss_cfg.stage2_s_loss = MaskWeightedMSE(min_area=100, pred_name='stage2_Smap',
            gt_image_name='gt_Smap', gt_mask_name='masks')
    loss_cfg.stage2_s_loss_weight = wl * 1

    loss_cfg.stage3_h_loss = MaskWeightedMSE(min_area=100, pred_name='stage3_Hmap',
            gt_image_name='gt_Hmap', gt_mask_name='masks')
    loss_cfg.stage3_h_loss_weight = wl * 1

    loss_cfg.stage3_rgb_loss = MaskWeightedMSE(min_area=100, pred_name='stage3_output',
            gt_image_name='target_images', gt_mask_name='masks')
    loss_cfg.stage3_rgb_loss_weight = 1

    loss_cfg.stage3_rgb_loss_fullres = MaskWeightedMSE(min_area=1000,
                                                       pred_name='stage3_output_fullres',
                                                       gt_image_name='target_images_fullres', gt_mask_name='masks_fullres')
    loss_cfg.stage3_rgb_loss_fullres_weight = 1.0

    loss_cfg.stage1_tv_loss = MaskWeightedTV(min_area=100, pred_name='stage1_filter',
            gt_image_name='target_images', gt_mask_name='masks')
    loss_cfg.stage1_tv_loss_weight = 1

    loss_cfg.stage2_tv_loss = MaskWeightedTV(min_area=100, pred_name='stage2_filter',
            gt_image_name='target_images', gt_mask_name='masks')
    loss_cfg.stage2_tv_loss_weight = 1

    loss_cfg.stage3_tv_loss = MaskWeightedTV(min_area=100, pred_name='stage3_filter',
            gt_image_name='target_images', gt_mask_name='masks')
    loss_cfg.stage3_tv_loss_weight = 1

    num_epochs = 70

    low_res_size = (256, 256)


    train_augmentator_1 = HCompose([
        RandomResizedCrop(*crop_size, scale=(0.5, 1.0)),
        HorizontalFlip(),
    ])
    train_augmentator_2 = HCompose([
        Resize(*low_res_size)
    ])

    val_augmentator_1 = HCompose([
        Resize(*crop_size)
    ])
    val_augmentator_2 = HCompose([
        Resize(*low_res_size)
    ])



    trainset = ComposeDatasetUpsample(
        [
            HDatasetUpsample(cfg.HFLICKR_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HDAY2NIGHT_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HCOCO_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HADOBE5K_RS_PATH, split='train', blur_target=blur),
        ],
        augmentator_1=train_augmentator_1,
        augmentator_2=train_augmentator_2,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05,
        use_hr=use_hr
    )

    valset = ComposeDatasetUpsample(
        [
            HDatasetUpsample(cfg.HFLICKR_PATH, split='test', blur_target=blur, mini_val=False),
            #HDatasetUpsample(cfg.HDAY2NIGHT_PATH, split='test', blur_target=blur, mini_val=False),
            #HDatasetUpsample(cfg.HCOCO_PATH, split='test', blur_target=blur, mini_val=False),
        ],
        augmentator_1=val_augmentator_1,
        augmentator_2=val_augmentator_2,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1,
        use_hr=use_hr
    )


    optimizer_params = {
        'lr': 5e-4,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    if cfg.local_rank == 0:
        print(optimizer_params)

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50], gamma=0.2)
    trainer = UpsampleDCCFTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[
            DenormalizedPSNRMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            ),
            DenormalizedMSEMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            ),
            DenormalizedMSEMetric(
                'stage3_output', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            ),
            MSEMetric(
                'stage1_Lmap', 'gt_Lmap',
            ),
            MSEMetric(
                'stage2_Smap', 'gt_Smap',
            ),
            MSEMetric(
                'stage3_Hmap', 'gt_Hmap',
            )
        ],
        checkpoint_interval=1,
        image_dump_interval=1000
    )

    if cfg.local_rank == 0:
        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        if epoch >= 40:
            trainer.loss_cfg.pixel_loss_fullres_weight = 1.0
            trainer.loss_cfg.stage1_l_loss_weight = 0.01
            trainer.loss_cfg.stage2_s_loss_weight = 0.01
            trainer.loss_cfg.stage3_h_loss_weight = 0.01
            trainer.loss_cfg.stage3_rgb_loss_weight = 0.01
            trainer.loss_cfg.stage3_rgb_loss_fullres_weight = 0.01

            trainer.loss_cfg.stage1_tv_loss_weight = 0.01
            trainer.loss_cfg.stage2_tv_loss_weight = 0.01
            trainer.loss_cfg.stage3_tv_loss_weight = 0.01


        trainer.training(epoch)
        trainer.validation(epoch)
