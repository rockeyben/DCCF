from time import perf_counter, time
from tqdm import trange
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from guided_filter_pytorch.guided_filter import GuidedFilter
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, N, AvgPredictTime

def to_image(x):
    return x[0].permute((1,2,0)).cpu().numpy() * 255

def to_eval(x):
    return x[0].permute((1,2,0))

def evaluate_dataset_rawsize(dataset, predictor, metrics_hub_lowres, metrics_hub_fullres, upsample_method=None, visdir=None):
    r = 8
    gpu = True
    eps = 1e-8
    gf = GuidedFilter(r, eps).cuda()

    for sample_i in trange(len(dataset), desc=f'Testing on {metrics_hub_lowres.name}'):

        sample = dataset.get_sample(sample_i)
        raw_input = sample['image']
        raw_mask = sample['object_mask']
        raw_target = sample['target_image']

        sample = dataset.augment_sample(sample)

        sample_mask = sample['object_mask']
        predict_start = time()
        pred, attention_map, raw_output = predictor.predict(sample['image'], sample_mask, return_numpy=False)
        torch.cuda.synchronize()

        metrics_hub_lowres.update_time(time() - predict_start)
        metrics_hub_fullres.update_time(time() - predict_start)
        lowres_image = torch.as_tensor(sample['target_image'], dtype=torch.float32).to(predictor.device)
        lowres_mask = torch.as_tensor(sample['object_mask'], dtype=torch.float32).to(predictor.device)
        with torch.no_grad():
            metrics_hub_lowres.compute_and_add(pred, lowres_image, lowres_mask)

        h, w = raw_mask.shape

        if upsample_method == 'GF':
            pred = TF.resize(pred.permute(2, 0, 1), (h, w))
            raw_guide = torch.as_tensor(raw_input, dtype=torch.float32).permute(2, 0, 1)[None].cuda() / 255.0
            pred = gf(raw_guide, pred[None] / 255.0)
            pred = pred[0].permute(1, 2, 0).cuda()
            pred = torch.clip(pred*255.0, 0, 255)
        elif upsample_method == 'bilinear':
            pred = TF.resize(pred.permute(2, 0, 1), (h, w)).permute(1, 2, 0)
        elif upsample_method == 'rawsize':
            pass
        elif upsample_method == 'BGU':
            imname = dataset.dataset_samples[sample_i]
            imname = imname.replace('.jpg', '')
            low_res_pred = pred.cpu().numpy()
            if visdir:
                cv2.imwrite(os.path.join(visdir, '%s_low_res_out.png' % imname),
                                low_res_pred[:, :, ::-1])
                cv2.imwrite(os.path.join(visdir, '%s_low_res_in.png' % imname),
                                sample['image'][:, :, ::-1])
            continue

        target_image = torch.as_tensor(raw_target, dtype=torch.float32).to(predictor.device)
        sample_mask = torch.as_tensor(raw_mask, dtype=torch.float32).to(predictor.device)
        with torch.no_grad():
            fullres_result = metrics_hub_fullres.compute_and_add(pred, target_image, sample_mask)

        if visdir and sample_i % 1 == 0:
            mse = fullres_result[1]
            psnr = fullres_result[2]
            txt = '%.2f %.2f' % (mse, psnr)
            pred = pred.cpu().numpy()
            sample_mask = sample_mask.cpu().numpy()
            sample_mask = sample_mask[:, :, np.newaxis]
            pred = pred * sample_mask + raw_target * ( 1 - sample_mask)
            imname = dataset.dataset_samples[sample_i]
            imname = imname.replace('.jpg', '')
            vis_image = pred[:, :, ::-1]
            
            cv2.imwrite(os.path.join(visdir, '%s.png' % imname), vis_image)
            

def evaluate_dataset_upsample_hsl_refine(dataset, predictor, metrics_hub_lowres, metrics_hub_fullres, visdir=None):

    for sample_i in trange(len(dataset), desc=f'Testing on {metrics_hub_lowres.name}'):

        bdata = dataset.get_sample(sample_i)
        imname = dataset.dataset_samples[sample_i]

        imname = imname.replace('.jpg', '')
        
        raw_image = bdata['image']
        raw_target = bdata['target_image']
        raw_mask = bdata['object_mask']
        sample = dataset.augment_sample(bdata, dataset.augmentator_2)
        sample_fullres = dataset.augment_sample(bdata, dataset.augmentator_1)
        sample_mask = sample['object_mask']
        raw_image_lowres = sample['image']
        image_fullres = sample_fullres['image']
        sample_mask_fullres = sample_fullres['object_mask']
        
        predict_start = time()
        pred, pred_fullres, inter_result = predictor.predict(sample['image'], image_fullres, raw_image, sample['target_image'], raw_target, 
                sample_mask, sample_mask_fullres, return_numpy=False)
        
        torch.cuda.synchronize()
        metrics_hub_lowres.update_time(time() - predict_start)
        metrics_hub_fullres.update_time(time() - predict_start)

        target_image = torch.as_tensor(sample['target_image'], dtype=torch.float32).to(predictor.device)
        sample_mask = torch.as_tensor(sample_mask, dtype=torch.float32).to(predictor.device)

        raw_target = torch.as_tensor(raw_target, dtype=torch.float32).to(predictor.device)
        raw_mask = torch.as_tensor(raw_mask, dtype=torch.float32).to(predictor.device)
        with torch.no_grad():
            metrics_hub_lowres.compute_and_add(pred, target_image, sample_mask)
            fullres_result = metrics_hub_fullres.compute_and_add(pred_fullres, raw_target, raw_mask)


        if visdir and sample_i % 1 == 0:
            mse = fullres_result[1]
            psnr = fullres_result[2]
            txt = '%.2f %.2f' % (mse, psnr)

            raw_mask = raw_mask.cpu().numpy()
            raw_mask = np.stack([raw_mask]*3, axis=2) * 255
            pred_fullres = pred_fullres.cpu().numpy()

            # cv2.imwrite(os.path.join(visdir, '%s.jpg' % imname), pred_fullres[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # print(os.path.join(visdir, '%s.jpg' % imname), visdir)
            # cv2.imwrite(os.path.join(visdir, '%s_input_fullres.jpg' % imname), raw_image[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_gt_fullres.jpg' % imname), bdata['target_image'][:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_mask_fullres.jpg' % imname), raw_mask * 255)
            
            pred = pred.cpu().numpy()
            low_mask = sample['object_mask'][...,np.newaxis]
            raw_mask = bdata['object_mask'][..., np.newaxis]

            # vis_l_input = to_image(inter_result['input_Lmap_fullres'])
            # vis_s_input = to_image(inter_result['input_Smap_fullres'])
            # vis_h_input = to_image(inter_result['input_Hmap_fullres'])
            # vis_s1_l = to_image(inter_result['stage1_Lmap_fullres']) * raw_mask + vis_l_input * (1 - raw_mask)
            # vis_s2_s = to_image(inter_result['stage2_Smap_fullres']) * raw_mask + vis_s_input * (1 - raw_mask)
            # vis_s3_h = to_image(inter_result['stage3_Hmap_fullres']) * raw_mask + vis_h_input * (1 - raw_mask)
            # vis_l_gt = to_image(inter_result['gt_Lmap_fullres'])
            # vis_s_gt = to_image(inter_result['gt_Smap_fullres'])
            # vis_h_gt = to_image(inter_result['gt_Hmap_fullres'])
            # vis_s1_rgb = to_image(inter_result['stage1_output_fullres']) * raw_mask + raw_image * (1 - raw_mask)
            # vis_s2_rgb = to_image(inter_result['stage2_output_fullres']) * raw_mask + raw_image * (1 - raw_mask)
            # vis_s3_rgb = to_image(inter_result['stage3_output_fullres']) * raw_mask + raw_image * (1 - raw_mask)
            # print(inter_result['stage3_output'].min(), inter_result['stage3_output'].max())
            
            # vis_s3_rgb = inter_result['stage3_output'] * low_mask + raw_image_lowres * (1 - low_mask)

            # vis_l_input_lowres = to_image(inter_result['input_Lmap']) 
            # vis_s_input_lowres = to_image(inter_result['input_Smap'])
            # vis_h_input_lowres = to_image(inter_result['input_Hmap'])

            # vis_s1_l_lowres = to_image(inter_result['stage1_Lmap']) * low_mask + vis_l_input_lowres * (1 - low_mask)
            # vis_s2_s_lowres = to_image(inter_result['stage2_Smap']) * low_mask + vis_s_input_lowres * (1 - low_mask)
            # vis_s3_h_lowres = to_image(inter_result['stage3_Hmap']) * low_mask + vis_h_input_lowres * (1 - low_mask)

            # cv2.imwrite(os.path.join(visdir, '%s_input_Lmap.jpg' % imname), vis_l_input[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_input_Smap.jpg' % imname), vis_s_input[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_input_Hmap.jpg' % imname), vis_h_input[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_gt_Lmap.jpg' % imname), vis_l_gt[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_gt_Smap.jpg' % imname), vis_s_gt[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_gt_Hmap.jpg' % imname), vis_h_gt[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage1_Lmap.jpg' % imname), vis_s1_l[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage2_Smap.jpg' % imname), vis_s2_s[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage3_Hmap.jpg' % imname), vis_s3_h[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage1_Lmap_lowres.jpg' % imname), vis_s1_l_lowres[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage2_Smap_lowres.jpg' % imname), vis_s2_s_lowres[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage3_Hmap_lowres.jpg' % imname), vis_s3_h_lowres)
            # cv2.imwrite(os.path.join(visdir, '%s_stage1_rgb.jpg' % imname), vis_s1_rgb[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage2_rgb.jpg' % imname), vis_s2_rgb[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_stage3_rgb.jpg' % imname), vis_s3_rgb[:, :, ::-1])
            cv2.imwrite(os.path.join(visdir, '%s_pred_lowres.jpg' % imname), pred[:, :, ::-1])

            # cv2.imwrite(os.path.join(visdir, '%s_input_fullres.jpg' % imname), raw_image[:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_gt_fullres.jpg' % imname), bdata['target_image'][:, :, ::-1])
            # cv2.imwrite(os.path.join(visdir, '%s_mask_fullres.jpg' % imname), raw_mask * 255)
            
