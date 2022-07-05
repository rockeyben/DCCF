from math import degrees
import torch
import torch.nn as nn
import time
import kornia

COLOR_RANGES = {
            'red':[315, 345, 15, 45],
            'yellow' : [15, 45, 75, 105],
            'green':[75, 105, 135, 165],
            'cyans':[135, 165, 195, 225],
            'blue':[195, 225, 255, 285],
            'magenta':[255, 285, 315, 345]
        }
COLORS = [
            'red','green', 'blue', 'cyans', 'magenta', 'yellow'
        ]


def clip_by_tensor(t, t_min, t_max):
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def get_bgr_delta( img_bgr, alpha_c, alpha_m ,alpha_y, alpha_k, omiga):
    # 计算r通道的增量
    delta_r = (-1 - alpha_c) * alpha_k - alpha_c
    min_r = - 1.0 * img_bgr[:, 2:3, :, :]
    max_r = 1.0 + min_r

    delta_rr = clip_by_tensor(delta_r * omiga, min_r * omiga, max_r * omiga)

    # 计算g通道的增量
    delta_g = (-1 - alpha_m) * alpha_k - alpha_m
    min_g = - 1.0 * img_bgr[:, 1:2, :, :]
    max_g = 1.0 + min_g
    delta_gg = clip_by_tensor(delta_g * omiga, min_g * omiga, max_g * omiga)

    # 计算b通道的增量
    delta_b = (-1 - alpha_y) * alpha_k - alpha_y
    min_b = - 1.0 * img_bgr[:, 0:1, :, :]
    max_b = 1.0 + min_b
    delta_bb = clip_by_tensor(delta_b * omiga, min_b * omiga, max_b * omiga)

    return delta_bb, delta_gg, delta_rr

def black_selective_color( img_bgr, alpha_c, alpha_m, alpha_y, alpha_k):
    """
    对黑色区域进行选择性颜色处理
    """
    N = 1.0
    half_n = 0.5
    max_v, _ = torch.max(img_bgr, axis=1, keepdim=True)
    omiga = (N / 2 - max_v) * 2

    delta_bb, delta_gg, delta_rr = get_bgr_delta(img_bgr, alpha_c,
                                                        alpha_m,
                                                        alpha_y,
                                                        alpha_k,
                                                        omiga)

    # 生成对应色彩的增量蒙板(float类型)
    mask = ((img_bgr[:, 0:1, :, :] < half_n) & (
                img_bgr[:, 1:2, :, :] < half_n) & (img_bgr[:, 2:3, :, :] < half_n))

    black_plus_map = torch.cat([delta_bb * mask, delta_gg * mask , delta_rr * mask], 1)

    return black_plus_map


def neutrals_selective_color( img_bgr, alpha_c, alpha_m, alpha_y, alpha_k):
    """
    对灰度色彩区域进行选择性颜色处理
    """
    N = 1.0
    half_n = 0.5
    min_v, _ = torch.min(img_bgr, axis=1, keepdim=True)
    max_v, _ = torch.max(img_bgr, axis=1, keepdim=True)
    omiga = N - (torch.abs(max_v - half_n) + torch.abs(min_v - half_n))

    delta_bb, delta_gg, delta_rr = get_bgr_delta(img_bgr, alpha_c,
                                                        alpha_m,
                                                        alpha_y,
                                                        alpha_k,
                                                        omiga)

    # 生成对应色彩的增量蒙板(float类型)
    mask = (((img_bgr[:, 0:1, :, :] > 0) | (img_bgr[:, 1:2, :, :] > 0) | (
                img_bgr[:, 2:3, :, :] > 0)) & ((img_bgr[:, 0:1, :, :] < N) | (
                img_bgr[:, 1:2, :, :] < N) | (img_bgr[:, 2:3, :, :] < N)))

    neutrals_plus_map = torch.cat([delta_bb * mask, delta_gg * mask , delta_rr * mask], 1)
    return neutrals_plus_map


def white_selective_color( img_bgr, alpha_c, alpha_m, alpha_y, alpha_k):
    """
    在白色通道进行可选颜色的调整
    """
    N = 1.0
    min_v, _ = torch.min(img_bgr, axis=1, keepdim=True)
    omiga = (min_v - N / 2) * 2

    delta_bb, delta_gg, delta_rr = get_bgr_delta(img_bgr, alpha_c,
                                                        alpha_m,
                                                        alpha_y,
                                                        alpha_k,
                                                        omiga)

    # 生成对应色彩的增量蒙板(float类型)
    mask = ((img_bgr[:, 0:1, :, :] > N /2) & (img_bgr[:, 1:2, :, :] > N /2) & (img_bgr[:, 2:3, :, :] > N /2))

    white_plus_map = torch.cat([delta_bb * mask, delta_gg * mask , delta_rr * mask], 1)
    return white_plus_map

def cmy_selective_color( img_bgr, channel, alpha_c, alpha_m, alpha_y, alpha_k):
    """
    对黄色、洋红、青色的颜色进行处理
    """
    min_v, _ = torch.min(img_bgr, axis=1, keepdim=True)
    mid_v, _ = torch.median(img_bgr, axis=1, keepdim=True)
    omiga = mid_v - min_v

    # 计算r通道的增量
    delta_bb, delta_gg, delta_rr = get_bgr_delta(img_bgr, alpha_c,
                                                        alpha_m,
                                                        alpha_y,
                                                        alpha_k,
                                                        omiga)
    # 生成对应色彩的增量蒙板(float类型)
    mask = (min_v == img_bgr[:,channel:channel+1, :, :])

    cmy_plus_map = torch.cat([delta_bb * mask, delta_gg * mask , delta_rr * mask], 1)     
    return cmy_plus_map


def rgb_selective_color( img_bgr, channel, alpha_c, alpha_m, alpha_y, alpha_k):
    """
    对rgb色彩区域做选择性增强
    """
    max_v, _ = torch.max(img_bgr, axis=1, keepdim=True)
    mid_v, _ = torch.median(img_bgr, axis=1, keepdim=True)
    omiga = max_v - mid_v

    delta_bb, delta_gg, delta_rr = get_bgr_delta(img_bgr, alpha_c,
                                                        alpha_m,
                                                        alpha_y,
                                                        alpha_k,
                                                        omiga)
    # 生成对应色彩的增量蒙板(float类型)
    mask = (max_v == img_bgr[:,channel:channel+1,:, :])

    rgb_plus_map = torch.cat([delta_bb * mask, delta_gg * mask , delta_rr * mask], 1)
    return rgb_plus_map

def get_sat_channel(img_bgr, method='xinzhi'):
    if method == 'default':
        eps = 1e-7
        saturation = ( img_bgr.max(1)[0] - img_bgr.min(1)[0] ) / ( img_bgr.max(1)[0] + eps )
        saturation[ img_bgr.max(1)[0]==0 ] = 0
        saturation = saturation.unsqueeze(1)

    elif method == 'xinzhi':
        red_plus_map   = rgb_selective_color(img_bgr, 2, 0, 0, 0, -1)
        green_plus_map = rgb_selective_color(img_bgr, 1, 0, 0, 0, -1)
        blue_plus_map  = rgb_selective_color(img_bgr, 0, 0, 0, 0, -1)
        yellow_plus_map  = cmy_selective_color(img_bgr, 0, 0, 0, 0, -1)
        magenta_plus_map = cmy_selective_color(img_bgr, 1, 0, 0, 0, -1)
        cyan_plus_map    = cmy_selective_color(img_bgr, 2, 0, 0, 0, -1)
        white_plus_map   = white_selective_color(img_bgr, 0, 0, 0, 1)
        neutral_plus_map = neutrals_selective_color(img_bgr, 0, 0, 0, 1)
        black_plus_map   = black_selective_color(img_bgr, 0, 0 , 0, 1)

        # 得到最后的map
        sum = red_plus_map + green_plus_map + blue_plus_map + yellow_plus_map + \
                magenta_plus_map + cyan_plus_map + white_plus_map + neutral_plus_map \
                + black_plus_map

        # 对原图进行操作
        dst = img_bgr + sum
        saturation = torch.clamp(dst, 0, 1)

    return saturation

def get_lum_channel(img_bgr, use_blur=False):
    value = img_bgr.max(1)[0]
    value = value.unsqueeze(1)
    if use_blur:
        value = kornia.gaussian_blur2d(value, (5,5), (1.5, 1.5))
    return value

def get_hue_channel(img_bgr, method='xinzhi'):
    if method == 'default':
        eps = 1e-7
        hue = torch.Tensor(img_bgr.shape[0], img_bgr.shape[2], img_bgr.shape[3]).to(img_bgr.device)
        hue[ img_bgr[:,0]==img_bgr.max(1)[0] ] = 4.0 + ( (img_bgr[:,2]-img_bgr[:,1]) / ( img_bgr.max(1)[0] - img_bgr.min(1)[0] + eps) ) [ img_bgr[:,0]==img_bgr.max(1)[0] ]
        hue[ img_bgr[:,1]==img_bgr.max(1)[0] ] = 2.0 + ( (img_bgr[:,0]-img_bgr[:,2]) / ( img_bgr.max(1)[0] - img_bgr.min(1)[0] + eps) ) [ img_bgr[:,1]==img_bgr.max(1)[0] ]
        hue[ img_bgr[:,2]==img_bgr.max(1)[0] ] = (0.0 + ( (img_bgr[:,1]-img_bgr[:,0]) / ( img_bgr.max(1)[0] - img_bgr.min(1)[0] + eps) ) [ img_bgr[:,2]==img_bgr.max(1)[0] ]) % 6

        hue[img_bgr.min(1)[0]==img_bgr.max(1)[0]] = 0.0
        hue = hue/6
        hue = hue.unsqueeze(1)
    elif method == 'xinzhi':
        #hue = get_color_map(img_bgr)
        hue=luminosity_blend(img_bgr)
    elif method == 'xinzhi_hsl':
        img_rgb = torch.flip(img_bgr, [1])
        hue = get_color_map_xinzhi(img_rgb)

    return hue

def apply_saturation_filter(comp_rgb, increment_params):
    eps = 1e-20
    comp_bgr = torch.flip(comp_rgb, [1])

    rgb_max,_ = torch.max(comp_rgb, axis=1, keepdim=True)
    rgb_min,_  = torch.min(comp_rgb, axis=1, keepdim=True)
    delta = rgb_max - rgb_min
    value = rgb_max + rgb_min
    L = value / 2
    mask1 = (L < 0.5) & (value > 0)
    mask2 = (L >= 0.5) & (value < 2)

    S = delta * mask1 / (value + eps) + delta * mask2 / (2 - value + eps)
    #S = S.detach()
    #L = L.detach()
    #delta = delta.detach()

    bgr_total = saturation_transfer_ps(comp_bgr, increment_params[:, 0:1, :, :], L, S, delta)
    map_delta_total = bgr_total - comp_bgr
    
    if increment_params.shape[1] == 7:
        input_Hmap = get_hue_channel(comp_bgr, method='default') * 360.0
        input_Hmap = input_Hmap.detach()
        for ci, cname in enumerate(COLORS):
            bgr_new = saturation_transfer_ps(comp_bgr, increment_params[:, ci+1:ci+2, :, :], L, S, delta)
            map_delta = color_saturation_delta(comp_bgr, input_Hmap, bgr_new, color_type=cname)
            map_delta_total += map_delta

    
    dst = comp_bgr + map_delta_total
    dst = torch.clamp(dst, min=0, max=1.0)

    output_rgb = torch.flip(dst, [1])
    return output_rgb


def saturation_transfer_ps(comp_bgr, cur_params, L, S, delta):
    eps = 1e-20

    cur_params = torch.clamp(cur_params, -0.99, 0.99)

    bgr_low = L + (comp_bgr - L) * (1 + cur_params)

    hmask2 = S >= (1 - cur_params)
    hmask3 = S < (1 - cur_params)
    bgr_high_1 = comp_bgr + (comp_bgr - L) * (1 / (S + eps) - 1)
    bgr_high_2 = comp_bgr + (comp_bgr - L) * (1 / (1 - cur_params + eps) - 1)
    bgr_high = bgr_high_1 * hmask2 + bgr_high_2 * hmask3


    bgr_new = torch.where(cur_params < 0, bgr_low, bgr_high)
    bgr_new = bgr_new * (delta != 0) + comp_bgr * (delta==0)
    bgr_new = torch.clamp(bgr_new, min=0, max=1.0)
    return bgr_new


def color_saturation_delta(comp_bgr, input_Hmap, bgr_new, color_type='red'):
    delta_total = bgr_new - comp_bgr

    s_l, r_l, r_r, s_r = COLOR_RANGES[color_type]
    if color_type == 'red':
        region_mask = (input_Hmap >= r_l) | (input_Hmap <= r_r)
    else:
        region_mask =  (input_Hmap >= r_l) & (input_Hmap <= r_r)

    # 软阈值内分别赋值
    s_lmask = (input_Hmap >= s_l) & (input_Hmap <= r_l)
    div = r_l - s_l
    alpha_map_left = ((input_Hmap - s_l) / div) * s_lmask

    s_rmask = (input_Hmap >= r_r) & (input_Hmap < s_r)
    div = s_r - r_r

    alpha_map_right = (1.0 - (input_Hmap - r_r) / div) * s_rmask

    delta_map = delta_total * region_mask + delta_total * alpha_map_left + \
                    delta_total * alpha_map_right
    return delta_map

def get_color_map(comp_bgr):
    alpha = 0.21
    Gmean = 0.5

    max, _ = torch.max(comp_bgr, axis=1, keepdim=True)
    min, _ = torch.min(comp_bgr, axis=1, keepdim=True)
    mean = (max + min) / 2

    # step1, 对需要减少的像素进行处理
    mask1 = mean >= Gmean
    delta1 = mean - Gmean
    delta1 = delta1 / alpha

    # step2, 对需要增大的像素进行处理
    mask2 = mean < Gmean
    delta2 = mean - Gmean
    delta2 = delta2 / (1 - alpha)

    delta_map = delta1 * mask1 + delta2 * mask2

    dst = comp_bgr + delta_map
    dst = torch.clamp(dst, 0, 1)

    return dst

def luminosity_blend(img_bgr):
    img_rgb = torch.flip(img_bgr, [1])
    img_hsv = rgb_to_hsv(img_rgb)
    img_hsv[:,2,:,:] = 0.9
    dst = hsv_to_rgb(img_hsv)
    dst = torch.flip(dst, [1])

    src_max, _ = torch.max(img_bgr, axis=1, keepdim=True)
    src_min, _ = torch.min(img_bgr, axis=1, keepdim=True)
    src_mean = (src_max + src_min) / 2

    dst_max, _ = torch.max(dst, axis=1, keepdim=True)
    dst_min, _ = torch.min(dst, axis=1, keepdim=True)
    dst_mean = (dst_max + dst_min) / 2

    delta = dst_mean - src_mean
    new_dst = img_bgr + delta
    new_dst = torch.clamp(new_dst, 0, 1)
    return new_dst


def hsv_to_rgb(hsv):
    h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
    #对出界值的处理
    h = h%1
    s = torch.clamp(s,0,1)
    v = torch.clamp(v,0,1)
    
    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))
    
    hi0 = hi==0
    hi1 = hi==1
    hi2 = hi==2
    hi3 = hi==3
    hi4 = hi==4
    hi5 = hi==5

    r = v * hi0 + q * hi1 + p * hi2 + p * hi3 + t * hi4 + v * hi5
    g = t * hi0 + v * hi1 + v * hi2 + q * hi3 + p * hi4 + p * hi5
    b = p * hi0 + p * hi1 + t * hi2 + v * hi3 + v * hi4 + q * hi5
    
    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    
    return rgb

def rgb_to_hsv(img):
    eps = 1e-7
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]
    
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)

    return hsv

def get_rgb_hue_channle(img):
    eps = 1e-7
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    return hue.unsqueeze(1)


def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    eps = 1e-7
    delta = cmax - cmin + eps
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    #cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    #hsl_h = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6) * (cmax_idx == 0) + \
    #    (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2) * (cmax_idx == 1) + \
    #    (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)* (cmax_idx == 2)
    hsl_h[delta == 0] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    #hsl_l_ma = (hsl_l > 0) & (hsl_l < 1)
    #hsl_l_s0_5 = hsl_l_ma & (hsl_l <= 0.5)
    #hsl_l_l0_5 = hsl_l_ma & (hsl_l > 0.5)   
    eps = 1e-7
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2. + eps))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2. + eps))[hsl_l_l0_5]
    
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    #rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    #rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    #rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    #rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    #rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    #rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb = torch.cat([_c, _x, _o], dim=1) * (idx == 0) + \
        torch.cat([_x, _c, _o], dim=1) * (idx ==1) + \
        torch.cat([_o, _c, _x], dim=1) * (idx == 2) + \
        torch.cat([_o, _x, _c], dim=1) * (idx == 3) + \
        torch.cat([_x, _o, _c], dim=1) * (idx == 4) + \
        torch.cat([_c, _o, _x], dim=1) * (idx == 5)
    rgb += _m
    return rgb

def get_color_map_xinzhi(img_rgb):
    im_hsl = rgb2hsl_torch(img_rgb)
    im_hsl[:, 2:3] = 204/255.0
    im_result = hsl2rgb_torch(im_hsl)

    src_max, _ = torch.max(img_rgb, axis=1, keepdim=True)
    src_min, _ = torch.min(img_rgb, axis=1, keepdim=True)
    src_mean = (src_max + src_min) / 2

    dst_max, _ = torch.max(im_result, axis=1, keepdim=True)
    dst_min, _ = torch.min(im_result, axis=1, keepdim=True)
    dst_mean = (dst_max + dst_min) / 2

    delta = dst_mean - src_mean
    new_dst = img_rgb + delta
    new_dst = torch.clamp(new_dst, 0, 1)
    return new_dst
