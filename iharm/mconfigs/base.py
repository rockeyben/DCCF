from iharm.model.base.dih_model_v3 import DeepImageHarmonizationUpsampleHSL_V3
from iharm.model.base import DeepImageHarmonization, SSAMImageHarmonization
                    
BMCONFIGS = {

    'improved_dih256': {
        'model': DeepImageHarmonization,
        'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True}
    },

    'improved_ssam256': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 4, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 2}
    },

    'dccf_improved_dih256_LR_clamp': {
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 7,'batchnorm_from': 2, 'image_fusion': True, 'backbone_type' : 'idih',
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : False, 'use_hr_inter' : False, 'hue_filters' : 12, 'clamp_sat_modify' : True, 'hue_norm_rotation' : False, 'use_blur_L' : True}
    },

    'dccf_improved_dih256_HR_clamp':{
       'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 7,'batchnorm_from': 2, 'image_fusion': True, 'backbone_type' : 'idih',
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : True, 'use_hr_inter' : False, 'hue_filters' : 12, 'clamp_sat_modify' : True, 'hue_norm_rotation' : False, 'use_blur_L' : True}

    },
    
    'dccf_improved_ssam256_LR_clamp':{
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 4, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 2, 'backbone_type' : 'ssam',
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : False, 'use_hr_inter' : False, 'hue_filters' : 12, 'clamp_sat_modify' : True, 'hue_norm_rotation' : False, 'use_blur_L' : True}
    },
    
    'dccf_improved_ssam256_HR_clamp':{
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 4, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 2, 'backbone_type' : 'ssam',
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : True, 'use_hr_inter' : False, 'hue_filters' : 12, 'clamp_sat_modify' : True, 'hue_norm_rotation' : False, 'use_blur_L' : True}
    },
    
    
    'improved_dih256_hsl_refine_LR' : {
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True,
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : False, 'use_hr_inter' : False, 'hue_filters' : 12,
            'hue_norm_rotation' : False, 'use_blur_L' : True}
    },
    'improved_dih256_hsl_refine_HR' : {
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True,
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : True, 'use_hr_inter' : False, 'hue_filters' : 12,
            'hue_norm_rotation' : False, 'use_blur_L' : True}
    },


    'improved_dih256_hsl_refine_LR_sat_clamp' : {
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True,
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : False, 'use_hr_inter' : False, 'hue_filters' : 12,
            'hue_norm_rotation' : False, 'use_blur_L' : True, 'use_dbl' : False, 'clamp_sat_modify' : True}
    },
    'improved_dih256_hsl_refine_HR_sat_clamp' : {
        'model': DeepImageHarmonizationUpsampleHSL_V3,
        'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True,
            'npts': 8, 'norm_rotation': True, 'up_size':(768, 1024), 'use_disentangle' : True,
            'h_method' : 'xinzhi_hsl', 'tune_method' : 'merge', 'use_refine' : True, 'use_attn' :True,
            'use_tanh' : False, 'use_detach' : False, 'use_hr' : True, 'use_hr_inter' : False, 'hue_filters' : 12,
            'hue_norm_rotation' : False, 'use_blur_L' : True, 'use_dbl' : False, 'clamp_sat_modify' : True}
    }

}
