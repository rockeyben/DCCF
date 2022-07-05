from os import fchdir
from collections import defaultdict
import argparse
import os
import os.path as osp
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import torch
from tqdm import tqdm
import scipy
import seaborn as sns

sys.path.insert(0, '.')
from iharm.inference.predictor_upsample_interactive import PredictorUpsampleInteractive
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger
from iharm.utils.exp import load_config_file
from math import fabs, sqrt,cos,sin,radians
from iharm.model.ps_filters import hsv_to_rgb, get_rgb_hue_channle, apply_saturation_filter, get_sat_channel
from albumentations import HorizontalFlip, Resize, RandomResizedCrop
from iharm.data.transforms import HCompose
import matplotlib.pyplot as plt

def get_shifts(shape, npts, device):
    B, C, H, W = shape
    shifts_ = torch.linspace(0, 1 - 1.0/npts, npts, device=device)
    shifts_ = shifts_.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
    shifts_ = shifts_.expand(B, C, npts, H, W)
    return shifts_

def to_image(x):
    return x[0].permute((1,2,0)).cpu().numpy() * 255

def generate_rotation(degrees):
    matrix = [0 for i in range(12)]

    cosA = cos(radians(degrees))
    sinA = sin(radians(degrees))
    matrix[0] = cosA + (1.0 - cosA) / 3.0
    matrix[1] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
    matrix[2] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
    matrix[3] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
    matrix[4] = cosA + 1./3.*(1.0 - cosA)
    matrix[5] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
    matrix[6] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
    matrix[7] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
    matrix[8] = cosA + 1./3. * (1.0 - cosA)

    return matrix



def plot_curve(params, plot_path, name):
    t_min = params[-1]
    slopes = params[:-1]
    print(t_min)
    print(slopes)
    shifts = np.linspace(0, 1 - 1 / len(slopes), len(slopes))
    print(shifts)

    xx = np.linspace(0, 1, len(slopes)+1)
    yy = []
    for xi in range(len(xx)):
        x = xx[xi]
        y = np.sum(slopes * np.maximum(x - shifts, 0)) + t_min
        yy.append(y)
    
    plt.figure()
    ax = plt.axes()
    plt.xticks([0,1])
    plt.yticks([0,1])
    ax.autoscale(False)
    ax.plot(xx, np.array(yy), 'o-', color='k')
    plt.savefig(str(plot_path / f'{name}.png'))


def plot_multi_curve(params, plot_path, name):
    plt.figure()
    ax = plt.axes()
    plt.xticks([0,1])
    plt.yticks([0,1])
    ax.autoscale(False)
    for param in params:
        t_min = param[-1]
        slopes = param[:-1]
        shifts = np.linspace(0, 1 - 1 / len(slopes), len(slopes))
        xx = np.linspace(0, 1, len(slopes)+1)
        yy = []
        for xi in range(len(xx)):
            x = xx[xi]
            y = np.sum(slopes * np.maximum(x - shifts, 0)) + t_min
            yy.append(y)
        ax.plot(xx, np.array(yy), 'o-')
    plt.savefig(str(plot_path / f'{name}.png'))
    



def apply_naive_L(image, params):
    frelu = torch.nn.ReLU()
    B, C, H, W = image.shape
    npts = len(params)
    params = torch.Tensor(params).to(image.device)
    params = params.view(1, npts, 1, 1).expand(B, npts, H, W)
    
    lum_min = params[:, npts-1:, :, :]
    slopes = params[:, :npts-1, :, :]
    shifts = get_shifts((B, 1, H, W), npts-1, image.device)
    
    output = image.unsqueeze(2).expand(B, 3, npts-1, H, W)
    output = torch.sum(slopes*frelu(output - shifts), dim=2) + lum_min
    output = torch.clamp(output, 0, 1)
    return output

def apply_ours_L(image, input_hsv, filters, params, alpha):
    frelu = torch.nn.ReLU()
    B, C, H, W = image.shape
    npts = len(params)
    params = torch.Tensor(params).to(image.device)
    params = params.view(1, npts, 1, 1).expand(B, npts, H, W)
    params = filters * (1 - alpha) + params * alpha
    
    lum_min = params[:, npts-1:, :, :]
    slopes = params[:, :npts-1, :, :]
    shifts = get_shifts((B, 1, H, W), npts-1, image.device)
    
    output_hsv = input_hsv.clone()
    input_v = input_hsv[:, 2:3, :, :]
    output_v = input_v.unsqueeze(2).expand(B, 1, npts-1, H, W)
    output_v = torch.sum(slopes*frelu(output_v - shifts), dim=2) + lum_min
    output_v = torch.clamp(output_v, 0, 1)
    output_hsv[:, 2:3, :, :] = output_v
    output = hsv_to_rgb(output_hsv)
    output = torch.clamp(output, 0, 1)
    
    return output

def apply_naive_S(image, sigma):
    B, C, H, W = image.shape
    sigma = torch.Tensor([sigma]).to(image.device)
    sigma = sigma.view(1, 1, 1, 1).expand(B, 1, H, W)

    # S curve, import from ihamr.ps_filter
    output = apply_saturation_filter(image, sigma)
    output = torch.clamp(output, 0, 1)

    return output

def apply_ours_S(image, input_hsv, filters, sigma, alpha):
    B, C, H, W = image.shape
    sigma = torch.Tensor([sigma]).to(image.device)
    sigma = sigma.view(1, 1, 1, 1).expand(B, 1, H, W)
    
    sigma = filters * (1 - alpha) + sigma * alpha

    # S curve, import from ihamr.ps_filter
    output = apply_saturation_filter(image, sigma)
    output = torch.clamp(output, 0, 1)
    
    output_hsv = input_hsv.clone()
    output_s = get_sat_channel(output, method='default')
    output_hsv[:, 1:2, :, :] = output_s
    output = hsv_to_rgb(output_hsv)
    output = torch.clamp(output, 0, 1)
    return output
            

def apply_naive_H(image, input_hsv, rot):
    B, C, H, W = image.shape

    rot = torch.Tensor(rot).to(image.device)
    rot = rot.view(1, 12, 1, 1).expand(B, 12, H, W)

    rot = rot[:,:9,:,:].view(B, 3, 3, H, W)
    output = image.unsqueeze(2).expand(B, 3, 3, H, W)
    output = torch.sum(rot*output, dim=1)

    output = torch.clamp(output, 0, 1)

    return output 

def apply_ours_H(image, input_hsv, filters, rot, alpha):
    B, C, H, W = image.shape
    matrix = filters
    guide_rot = matrix[:, :9, :, :].view(B, 3, 3, H, W)

    rot = torch.Tensor(rot).to(image.device)
    user_rot = rot.view(1, 12, 1, 1).expand(B, 12, H, W)
    user_rot_bias = user_rot.view(B, 3, 4, H, W)
    user_rot = user_rot[:,:9,:,:].view(B, 3, 3, H, W)

    output = image.unsqueeze(2).expand(B, 3, 3, H, W)
    bias = matrix[:, 9:12, :, :]
    
    final_rot = guide_rot
    final_rot = guide_rot * (1 - alpha) + user_rot * alpha
    # final_rot = torch.clamp(guide_rot * user_rot, -1, 1)
    # final_rot = guide_rot * user_rot
    # final_rot[:, 2:3 , :, :] = guide_rot[:, 2:3 , :, :] * (1 - alpha) + user_rot[:, 2:3 , :, :] * alpha

    output = torch.sum(final_rot*output, dim=1) + bias
    output = torch.clamp(output, 0, 1)

    output_hsv = input_hsv
    output_h = get_rgb_hue_channle(output)
    output_hsv[:, 0:1, :, :] = output_h
    output = hsv_to_rgb(output_hsv)
    output = torch.clamp(output, 0, 1)
    return output 


def compute_dist(a):
    dist = np.sqrt(((a[:, None] - a[:, :, None])**2).sum(0))
    return dist

def amplify_detail(image, mask, filters, plot_path):
    fg_pos_list = np.where(mask == 1)
    plot_points = 3
    image = image.astype(np.uint8).copy()
    
    filters = filters.cpu().numpy()
    
    filter_vecs = []
    for fp in range(len(fg_pos_list[0])):
        if fp % 100 == 0:
            x = fg_pos_list[0][fp]
            y = fg_pos_list[1][fp]
            filter_vecs.append(filters[0, :, x, y])
    
    filter_vecs = np.array(filter_vecs)
    dist = scipy.spatial.distance.cdist(filter_vecs, filter_vecs)
    
    bm = dist.max()
    bi = np.unravel_index(dist.argmax(), dist.shape)
    
    '''
    for rr in range(plot_points):
        random_idx = np.random.randint(len(fg_pos_list[0]))
        r_x = fg_pos_list[0][random_idx]
        r_y = fg_pos_list[1][random_idx]
        print(r_x, r_y)
        cv2.drawMarker(image, (int(r_y), int(r_x)), (0, 255, 0), markerType=rr, thickness=5)
        plot_curve(filters[0,:,r_x,r_y], plot_path, 'detail'+str(rr))
    '''
    curves = []
    for rr in range(2):
        r_x = fg_pos_list[0][bi[rr]* 100]
        r_y = fg_pos_list[1][bi[rr]* 100]
        print(r_x, r_y)
        cv2.drawMarker(image, (int(r_y), int(r_x)), (0, 255, 0), markerType=rr, thickness=5)
        plot_curve(filters[0,:,r_x,r_y], plot_path, 'detail'+str(rr))
        curves.append(filters[0,:,r_x,r_y])
    plot_multi_curve(curves, plot_path, 'two_curve')
    cv2.imwrite(str(plot_path / f'marker_image.png'), 
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    args, cfg = parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    #device = 'cpu'
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    predictor = PredictorUpsampleInteractive(net, device)

    image_names = os.listdir(args.images)
    image_names = [x for x in image_names if 'input' in x]

    def _save_image(image_name, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(cfg.RESULTS_PATH / f'{image_name}'),
            rgb_image#[cv2.IMWRITE_JPEG_QUALITY, 85]
        )

    logger.info(f'Save images to {cfg.RESULTS_PATH}')

    resize_shape = (args.resize, ) * 2

    resize_op1 = Resize(768, 1024)
    resize_op = Resize(256, 256)
    for image_name in tqdm(image_names):
        image_path = osp.join(args.images, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=0.2, fy=0.2)
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape
        raw_image = image.astype(np.float32)

        if resize_shape[0] > 0:
            #image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
            #image_fullres = resize_op1(image=image)['image']
            image = resize_op(image=image)['image']
            image_fullres = resize_op1(image=raw_image)['image']
            #image_fullres = cv2.resize(image, (1024, 768), cv2.INTER_LINEAR)

        mask_path = osp.join(args.masks, image_name.replace('input.jpg', 'mask.png'))
        mask_image = cv2.imread(mask_path)
        mask_image = cv2.resize(mask_image, None, fx=0.2, fy=0.2)
        mask = mask_image[:, :, 0] / 255.0
        mask = mask.astype(np.float32)
        if resize_shape[0] > 0:
            mask_image_low = cv2.resize(mask_image, resize_shape, cv2.INTER_LINEAR)
        mask_low = mask_image_low[:, :, 0] / 255.0
        mask_low = mask_low.astype(np.float32)
        pred, pred_fullres, inter_result = predictor.predict(image, image_fullres, raw_image, image , raw_image, 
                mask_low, mask, return_numpy=True)
        
        # pred_fullres = to_image(pred_fullres)
        _save_image('pred_' + image_name, pred_fullres)

        hue_filters = inter_result['stage3_filter_fullres']
        sat_filters = inter_result['stage2_filter_fullres']
        lum_filters = inter_result['stage1_filter_fullres']
        
        stage1_output = inter_result['stage1_output_fullres']
        stage2_output = inter_result['stage2_output_fullres']
        stage3_output = inter_result['stage3_output_fullres']
        
        input_rgb = inter_result['input_rgb_fullres']
        input_hsv = inter_result['input_hsv_fullres']
        stage1_output_hsv = inter_result['stage1_output_hsv_fullres']
        stage2_output_hsv = inter_result['stage2_output_hsv_fullres']

        mask = mask[:, :, np.newaxis]

        I_channel = args.channel
        
        if I_channel == 'H':
            viss = []
            viss_naive = []
            rotations = [60, 120, 180, 240, 300]
            for i in range(len(rotations)):
                rot = generate_rotation(rotations[i])
                pred_naive = apply_naive_H(stage2_output, input_hsv, rot)
                pred_naive = to_image(pred_naive) * mask  + raw_image * (1 - mask)

                vis_i = [raw_image]
                vis_n = [raw_image]

                div = 5
                alpha_base = 0.1
                for alpha in range(div+1):
                    if alpha == 0:
                        ta  = 0
                    else:
                        ta = alpha_base + alpha* (1-alpha_base)/div
                    pred_ours = apply_ours_H(stage2_output, stage2_output_hsv, hue_filters, rot, ta)
                    pred_ours = to_image(pred_ours) * mask + raw_image * (1 - mask)
                    _save_image('theta_%d_alpha_%.2f_%s' % (rotations[i], ta, image_name), pred_ours)
                    
                    interp_naive = pred_fullres * (1 - ta) + pred_naive * ta
                    _save_image('naive_theta_%d_alpha_%.2f_%s'% (rotations[i], ta, image_name), interp_naive)
                    
                    vis_i.append(pred_ours)
                    vis_n.append(interp_naive)
                    
                vis_i = np.concatenate(vis_i, 1)
                viss.append(vis_i)
                vis_n = np.concatenate(vis_n, 1)
                viss_naive.append(vis_n)
            
            _save_image('merge_ours_' + image_name, np.concatenate(viss, 0))
            _save_image('merge_naive_' + image_name, np.concatenate(viss_naive, 0))
            
        elif I_channel == 'S':
            viss_naive = []
            viss_ours = []
            sigmas = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]
            # sigmas = [0.5]
            input_Smap = inter_result['input_Smap_fullres']
            pred_Smap = inter_result['stage2_Smap_fullres']
            for i in range(len(sigmas)):
                sigma =  sigmas[i]
                pred_naive = apply_naive_S(stage1_output, sigma)
                pred_naive = to_image(pred_naive) * mask + raw_image * (1 - mask)
                pred_ours = apply_ours_S(stage1_output, stage1_output_hsv, sat_filters, sigma, 0.5)
                pred_ours = to_image(pred_ours) * mask + raw_image * (1 - mask)
                
                viss_naive.append(pred_naive)
                viss_ours.append(pred_ours)
                _save_image('naive'+str(i)+image_name, pred_naive)
                _save_image('ours'+str(i)+image_name, pred_ours)

            # print(input_Smap.shape, pred_Smap.shape)
            # input_S_vis = to_image(input_Smap)
            # pred_S_vis = to_image(pred_Smap) * mask + input_S_vis * (1 - mask)
            # diff = (((pred_S_vis - input_S_vis) * 2 + 270.0) / 2) * mask
            # print(np.min(diff), np.max(diff))
            # diff = diff[:, :, 0]
            # diff = cv2.GaussianBlur(diff, (11, 11), 0)
            # diff = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_BONE)
            
            # S_filter = to_image((sat_filters + 1) / 2.0) * mask
            # print(np.min(S_filter), np.max(S_filter))
            # S_filter = cv2.applyColorMap(S_filter[:, :, 0].astype(np.uint8), cv2.COLORMAP_BONE)
            
            # legend = np.arange(0, 255)
            # legend = np.stack([legend] * 20)
            # print(legend.shape)
            # legend = cv2.applyColorMap(legend.astype(np.uint8), cv2.COLORMAP_BONE)
            # _save_image('legend.png', legend)

            # _save_image('input_S.png', input_S_vis)
            # _save_image('pred_S.png', pred_S_vis)
            # _save_image('diff_S.png', diff * mask)
            # _save_image('sigma_S.png', S_filter * mask)
            # viss_naive = np.concatenate(viss_naive, 1)
            # viss_ours = np.concatenate(viss_ours, 1)
            # viss = np.concatenate([viss_naive, viss_ours], 0)
            # _save_image(image_name, viss)
        elif I_channel == 'L':
            viss_naive = [raw_image]
            viss_ours = [raw_image]
            lumParam = [np.random.random(9)*0.5 - 0.1 for _ in range(1)]
            avg_curve = torch.mean(lum_filters, axis=[0, 2, 3])
            # lumParam = [np.array(avg_curve.cpu())]
            lumParam = [np.random.random(9)*0.5 - 0.15 for _ in range(8)]

            lum_level = []
            for i in range(len(lumParam)):
                params = lumParam[i]
                pred_naive = apply_naive_L(input_rgb, params)
                pred_naive = to_image(pred_naive) * mask + raw_image * (1 - mask)
                pred_ours = apply_ours_L(input_rgb, input_hsv, lum_filters, params, 0.5)
                pred_ours = to_image(pred_ours) * mask + raw_image * (1 - mask)
                # amplify_detail(pred_ours, mask, lum_filters, cfg.RESULTS_PATH)
                viss_naive.append(pred_naive)
                viss_ours.append(pred_ours)
                lum_level.append(np.mean(pred_naive))
                
            level_idx = np.argsort(lum_level)
            for li, _ in enumerate(level_idx):
                plot_curve(lumParam[_], cfg.RESULTS_PATH, li)
            viss_naive_sort = [viss_naive[_+1] for _ in level_idx]
            viss_ours_sort = [viss_ours[_+1] for _ in level_idx]
            viss_naive = np.concatenate(viss_naive_sort, 1)
            viss_ours = np.concatenate(viss_ours_sort, 1)
            viss = np.concatenate([viss_naive, viss_ours], 0)
            _save_image(image_name, viss)
            for li in range(len(viss_naive_sort)):
                _save_image('naive'+str(li)+image_name, viss_naive_sort[li])
                _save_image('ours'+str(li)+image_name, viss_ours_sort[li])
                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument(
        '--images', type=str,
        help='Path to directory with .jpg images to get predictions for.'
    )
    parser.add_argument(
        '--masks', type=str,
        help='Path to directory with .png binary masks for images, named exactly like images without last _postfix.'
    )
    parser.add_argument(
        '--resize', type=int, default=256,
        help='Resize image to a given size before feeding it into the network. If -1 the network input is not resized.'
    )
    parser.add_argument(
        '--original-size', action='store_true', default=False,
        help='Resize predicted image back to the original size.'
    )
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml', help='The path to the config file.')
    parser.add_argument(
        '--results-path', type=str, default='',
        help='The path to the harmonized images. Default path: cfg.EXPS_PATH/predictions.'
    )
    parser.add_argument(
        '--channel', type=str, default='L',
        help='visualization channel: [H, S, L]'
    )

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
    cfg.RESULTS_PATH = Path(args.results_path) if len(args.results_path) else cfg.EXPS_PATH / 'predictions'
    cfg.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(cfg)
    return args, cfg


if __name__ == '__main__':
    main()
