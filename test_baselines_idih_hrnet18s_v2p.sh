weights='pretrained_models/idih_hrnet18s_v2p.pth'

# Using config_test_HR.yml would get worse results for BGU, so we use config_test_LR.yml as a stronger baseline
# Run BGU algorithm [https://github.com/google/bgu] with the saved intermediate low-res results
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py hrnet18s_v2p_idih256 ${weights} \
    --config-path config_test_LR.yml \
    --resize-strategy Fixed256 \
    --vis-dir harmonization_exps/vis_idih_v2p_BGU \
    --upsample-method BGU

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py hrnet18s_v2p_idih256 ${weights} \
    --config-path config_test_HR.yml \
    --resize-strategy Fixed256 \
    --upsample-method bilinear

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py hrnet18s_v2p_idih256 ${weights} \
    --config-path config_test_HR.yml \
    --resize-strategy Fixed256 \
    --upsample-method GF

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py hrnet18s_v2p_idih256 ${weights} \
    --config-path config_test_HR.yml \
    --resize-strategy None \
    --datasets HFlickr,HDay2Night,HCOCO \
    --upsample-method rawsize 
