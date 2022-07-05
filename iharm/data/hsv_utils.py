import numpy as np
import cv2

class SelectedColorPara():
    def __init__(self):
        self.red_para     = [0.0, 0.0, 0.0, -1.0]
        self.yellow_para  = [0.0, 0.0, 0.0, -1.0]
        self.green_para   = [0.0, 0.0, 0.0, -1.0]
        self.cyan_para    = [0.0, 0.0, 0.0, -1.0]    #青色
        self.blue_para    = [0.0, 0.0, 0.0, -1.0]
        self.magenta_pata = [0.0, 0.0, 0.0, -1.0]
        self.white_para   = [0.0, 0.0, 0.0, 1.0]
        self.neutral_para = [0.0, 0.0, 0.0, 1.0]
        self.black_para   = [0.0, 0.0, 0.0, 1.0]

class PsSelectiveColor():
    def __init__(self, img_bgr, para:SelectedColorPara):
        self.img_bgr = img_bgr
        self.N = 255
        self.para = para

    def get_saturation_map_fast(self):
        """
        迅速得到saturation map
        Returns:
        """
        max_v = np.max(self.img_bgr, axis=2)
        min_v = np.min(self.img_bgr, axis=2)
        self.saturation_map = max_v - min_v
        return np.stack((self.saturation_map, self.saturation_map, self.saturation_map), axis=2)

    def get_saturation_map(self):
        """
        计算该rgb图像的saturation map
        Returns:
        """
        red_plus_map   = self.rgb_selective_color(2, 0, 0, 0, -1).astype(np.int32)
        green_plus_map = self.rgb_selective_color(1, 0, 0, 0, -1).astype(np.int32)
        blue_plus_map  = self.rgb_selective_color(0, 0, 0, 0, -1).astype(np.int32)
        yellow_plus_map  = self.cmy_selective_color(0, 0, 0, 0, -1).astype(np.int32)
        magenta_plus_map = self.cmy_selective_color(1, 0, 0, 0, -1).astype(np.int32)
        cyan_plus_map    = self.cmy_selective_color(2, 0, 0, 0, -1).astype(np.int32)
        white_plus_map   = self.white_selective_color(0, 0, 0, 1).astype(np.int32)
        neutral_plus_map = self.neutrals_selective_color(0, 0, 0, 1).astype(np.int32)
        black_plus_map   = self.black_selective_color(0, 0 , 0, 1).astype(np.int32)

        # 得到最后的map
        sum = red_plus_map + green_plus_map + blue_plus_map + yellow_plus_map + \
              magenta_plus_map + cyan_plus_map + white_plus_map + neutral_plus_map \
              + black_plus_map

        # 对原图进行操作
        dst = self.img_bgr + sum
        dst = np.clip(dst, 0, 255)

        return dst.astype(np.uint8)

    def color_enhance(self):
        red_plus_map   = self.rgb_selective_color(2, self.para.red_para[0],
                                                     self.para.red_para[1],
                                                     self.para.red_para[2],
                                                     self.para.red_para[3]).astype(np.int32)

        green_plus_map = self.rgb_selective_color(1, self.para.green_para[0],
                                                     self.para.green_para[1],
                                                     self.para.green_para[2],
                                                     self.para.green_para[3]).astype(np.int32)

        blue_plus_map  = self.rgb_selective_color(0, self.para.blue_para[0],
                                                     self.para.blue_para[1],
                                                     self.para.blue_para[2],
                                                     self.para.blue_para[3]).astype(np.int32)

        yellow_plus_map  = self.cmy_selective_color(0, self.para.yellow_para[0],
                                                       self.para.yellow_para[1],
                                                       self.para.yellow_para[2],
                                                       self.para.yellow_para[3]).astype(np.int32)

        magenta_plus_map = self.cmy_selective_color(1, self.para.magenta_pata[0],
                                                       self.para.magenta_pata[1],
                                                       self.para.magenta_pata[2],
                                                       self.para.magenta_pata[3]).astype(np.int32)

        cyan_plus_map    = self.cmy_selective_color(2, self.para.cyan_para[0],
                                                       self.para.cyan_para[1],
                                                       self.para.cyan_para[2],
                                                       self.para.cyan_para[3]).astype(np.int32)

        white_plus_map   = self.white_selective_color(self.para.white_para[0],
                                                      self.para.white_para[1],
                                                      self.para.white_para[2],
                                                      self.para.white_para[3]).astype(np.int32)

        neutral_plus_map = self.neutrals_selective_color(self.para.neutral_para[0],
                                                         self.para.neutral_para[1],
                                                         self.para.neutral_para[2],
                                                         self.para.neutral_para[3]).astype(np.int32)

        black_plus_map   = self.black_selective_color(self.para.black_para[0],
                                                      self.para.black_para[1],
                                                      self.para.black_para[2],
                                                      self.para.black_para[3]).astype(np.int32)

        # 得到最后的map
        sum = red_plus_map + green_plus_map + blue_plus_map + yellow_plus_map + \
              magenta_plus_map + cyan_plus_map + white_plus_map + neutral_plus_map \
              + black_plus_map

        # 对原图进行操作
        dst = self.img_bgr + sum
        dst = np.clip(dst, 0, 255)

        src_yuv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HLS)
        dst_yuv = cv2.cvtColor(dst.astype(np.uint8), cv2.COLOR_BGR2HLS)
        src_yuv[:,:,0] = dst_yuv[:,:,0]

        dst_new = cv2.cvtColor(src_yuv, cv2.COLOR_HLS2BGR)
        return dst_new

    def get_bgr_delta(self, alpha_c, alpha_m, alpha_y, alpha_k, omiga):
        # 计算r通道的增量
        img_bgr = self.img_bgr
        N       = self.N

        delta_r = (-1 - alpha_c) * alpha_k - alpha_c
        min_r = - 1.0 * img_bgr[:, :, 2] / N
        max_r = 1.0 + min_r
        delta_rr = np.clip(delta_r * omiga, np.around(min_r * omiga),
                           np.around(max_r * omiga))

        # 计算g通道的增量
        delta_g = (-1 - alpha_m) * alpha_k - alpha_m
        min_g = - 1.0 * img_bgr[:, :, 1] / N
        max_g = 1.0 + min_g
        delta_gg = np.clip(delta_g * omiga, np.around(min_g * omiga ),
                           np.around(max_g * omiga))

        # 计算b通道的增量
        delta_b = (-1 - alpha_y) * alpha_k - alpha_y
        min_b = - 1.0 * img_bgr[:, :, 0] / N
        max_b = 1.0 + min_b
        delta_bb = np.clip(delta_b * omiga, np.around(min_b * omiga),
                           np.around(max_b * omiga))

        return delta_bb, delta_gg, delta_rr

    def black_selective_color(self, alpha_c = 0.0, alpha_m = 0.0,
                                    alpha_y = 0.0, alpha_k = 1.0):
        """
        对黑色区域进行选择性颜色处理
        Args:
            alpha_c:
            alpha_m:
            alpha_y:
            alpha_k:
        Returns:
        """
        img_bgr = self.img_bgr
        N = 255
        half_n = 127.5
        max_v = np.max(img_bgr, axis=2)
        omiga = (N / 2 - max_v) * 2

        delta_bb, delta_gg, delta_rr = self.get_bgr_delta(alpha_c,
                                                         alpha_m,
                                                         alpha_y,
                                                         alpha_k,
                                                         omiga)

        # 生成对应色彩的增量蒙板(float类型)
        mask = ((img_bgr[:, :, 0] < half_n) & (
                 img_bgr[:, :, 1] < half_n) & (img_bgr[:, :, 2] < half_n))

        using_r_map = np.zeros(delta_rr.shape, delta_rr.dtype)
        using_r_map[mask] = np.around(delta_rr[mask])
        using_g_map = np.zeros(delta_gg.shape, delta_gg.dtype)
        using_g_map[mask] = np.around(delta_gg[mask])
        using_b_map = np.zeros(delta_bb.shape, delta_bb.dtype)
        using_b_map[mask] = np.around(delta_bb[mask])
        black_plus_map = np.stack((using_b_map, using_g_map, using_r_map),axis=2)
        return black_plus_map

    def neutrals_selective_color(self, alpha_c = 0.0, alpha_m = 0.0,
                                    alpha_y = 0.0, alpha_k = 1.0):
        """
        对灰度色彩区域进行选择性颜色处理
        Args:
            alpha_c:
            alpha_m:
            alpha_y:
            alpha_k:
        Returns:
        """
        img_bgr = self.img_bgr
        N = 255
        half = 127.5
        min_v = np.min(img_bgr, axis=2)
        max_v = np.max(img_bgr, axis=2)
        omiga = N - (np.abs(max_v - half) + np.abs(min_v - half))

        # 计算r通道的增量
        delta_bb, delta_gg, delta_rr = self.get_bgr_delta(alpha_c,
                                                         alpha_m,
                                                         alpha_y,
                                                         alpha_k,
                                                         omiga)

        # 生成对应色彩的增量蒙板(float类型)
        mask = (((img_bgr[:, :, 0] > 0) | (img_bgr[:, :, 1] > 0) | (
                 img_bgr[:, :, 2] > 0)) & ((img_bgr[:, :, 0] < N) | (
                 img_bgr[:, :, 1] < N) | (img_bgr[:, :, 2] < N)))

        using_r_map = np.zeros(delta_rr.shape, delta_rr.dtype)
        using_r_map[mask] = np.around(delta_rr[mask])
        using_g_map = np.zeros(delta_gg.shape, delta_gg.dtype)
        using_g_map[mask] = np.around(delta_gg[mask])
        using_b_map = np.zeros(delta_bb.shape, delta_bb.dtype)
        using_b_map[mask] = np.around(delta_bb[mask])
        neutrals_plus_map = np.stack((using_b_map, using_g_map, using_r_map),
                                  axis=2)
        return neutrals_plus_map

    def white_selective_color(self, alpha_c = 0.0, alpha_m = 0.0,
                                    alpha_y = 0.0, alpha_k = 1.0):
        """
        在白色通道进行可选颜色的调整
        Args:
            alpha_c:
            alpha_m:
            alpha_y:
            alpha_k:
        Returns:
        """
        img_bgr = self.img_bgr
        N = 255
        min_v = np.min(img_bgr, axis=2)
        omiga = (min_v - N / 2) * 2
        
        # 计算r通道的增量
        delta_bb, delta_gg, delta_rr = self.get_bgr_delta(alpha_c,
                                                         alpha_m,
                                                         alpha_y,
                                                         alpha_k,
                                                         omiga)
        # 生成对应色彩的增量蒙板(float类型)
        mask = ((img_bgr[:,:,0] > N /2) & (img_bgr[:,:,1] > N /2) & (img_bgr[:,:,2] > N /2))
        using_r_map = np.zeros(delta_rr.shape, delta_rr.dtype)
        using_r_map[mask] = np.around(delta_rr[mask])
        using_g_map = np.zeros(delta_gg.shape, delta_gg.dtype)
        using_g_map[mask] = np.around(delta_gg[mask])
        using_b_map = np.zeros(delta_bb.shape, delta_bb.dtype)
        using_b_map[mask] = np.around(delta_bb[mask])
        white_plus_map = np.stack((using_b_map, using_g_map, using_r_map), axis=2)
        return white_plus_map

    def cmy_selective_color(self, channel,  alpha_c = 0.0, alpha_m = 0.0,
                                  alpha_y = 0.0, alpha_k = -1.0):
        """
        对黄色、洋红、青色的颜色进行处理
        Args:
            channel: 0 - yellow 1- magentas 2- cyans
            alpha_c:
            alpha_m:
            alpha_y:
            alpha_k:
        Returns:
        """
        img_bgr = self.img_bgr
        N = 255
        min_v = np.min(img_bgr, axis=2)
        mid_v = np.median(img_bgr, axis=2)
        omiga = mid_v - min_v

        # 计算r通道的增量
        delta_bb, delta_gg, delta_rr = self.get_bgr_delta(alpha_c,
                                                         alpha_m,
                                                         alpha_y,
                                                         alpha_k,
                                                         omiga)
        # 生成对应色彩的增量蒙板(float类型)
        mask = (min_v == img_bgr[:,:,channel])
        using_r_map = np.zeros(delta_rr.shape, delta_rr.dtype)
        using_r_map[mask] = np.around(delta_rr[mask])
        using_g_map = np.zeros(delta_gg.shape, delta_gg.dtype)
        using_g_map[mask] = np.around(delta_gg[mask])
        using_b_map = np.zeros(delta_bb.shape, delta_bb.dtype)
        using_b_map[mask] = np.around(delta_bb[mask])
        cmy_plus_map = np.stack((using_b_map, using_g_map, using_r_map), axis=2)
        return cmy_plus_map

    def rgb_selective_color(self, channel, alpha_c = 0.0, alpha_m = 0.0,
                            alpha_y = 0.0, alpha_k = -1.0):
        """
        对rgb色彩区域做选择性增强
        Args:
            channel: 0- b 1 - g 2 - r
            alpha_c:
            alpha_m:
            alpha_y:
            alpha_k:
        Returns:
        """
        img_bgr = self.img_bgr
        N = 255
        max_v = np.max(img_bgr, axis=2)
        mid_v = np.median(img_bgr, axis=2)
        omiga = max_v - mid_v

        delta_bb, delta_gg, delta_rr = self.get_bgr_delta(alpha_c,
                                                         alpha_m,
                                                         alpha_y,
                                                         alpha_k,
                                                         omiga)
        # 生成对应色彩的增量蒙板(float类型)
        mask = (max_v == img_bgr[:,:,channel])
        using_r_map = np.zeros(delta_rr.shape, delta_rr.dtype)
        using_r_map[mask] = np.around(delta_rr[mask])
        using_g_map = np.zeros(delta_gg.shape, delta_gg.dtype)
        using_g_map[mask] = np.around(delta_gg[mask])
        using_b_map = np.zeros(delta_bb.shape, delta_bb.dtype)
        using_b_map[mask] = np.around(delta_bb[mask])
        rgb_plus_map = np.stack((using_b_map, using_g_map, using_r_map), axis=2)
        return rgb_plus_map

