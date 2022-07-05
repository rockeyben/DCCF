from .base import BMCONFIGS
from iharm.model.backboned import DeepLabIHModel, HRNetIHModel, HRNetIHModelMhead, HRNetIHModelUpsample,\
                        HRNetIHModelHSL


MCONFIGS = {

    'hrnet18s_v2p_idih256': {
        'model': HRNetIHModel,
        'params': {'base_config': BMCONFIGS['improved_dih256'], 'pyramid_channels': 256}
    },

    'hrnet18s_v2p_idih256_upsample_hsl_refine_LR' : {
        'model' : HRNetIHModelHSL,
        'params' : {'base_config': BMCONFIGS['improved_dih256_hsl_refine_LR'], 'pyramid_channels': 256}
    },

    'hrnet18s_v2p_idih256_upsample_hsl_refine_HR' : {
        'model' : HRNetIHModelHSL,
        'params' : {'base_config': BMCONFIGS['improved_dih256_hsl_refine_HR'], 'pyramid_channels': 256}
    },

    'hrnet18s_v2p_idih256_upsample_hsl_refine_LR_sat_clamp' : { 
        'model' : HRNetIHModelHSL,
        'params' : {'base_config': BMCONFIGS['improved_dih256_hsl_refine_LR_sat_clamp'], 'pyramid_channels': 256}
    },

    'hrnet18s_v2p_idih256_upsample_hsl_refine_HR_sat_clamp' : { 
        'model' : HRNetIHModelHSL,
        'params' : {'base_config': BMCONFIGS['improved_dih256_hsl_refine_HR_sat_clamp'], 'pyramid_channels': 256}
    },

}
