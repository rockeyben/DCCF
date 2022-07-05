pretrain_path="pretrained_models/dccf_idih_hrnet18s_v2p_HR_pretrain.pth"
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_upsample_refiner.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${pretrain_path} \
    --resize-strategy Fixed256 \
    --version hsl \
    --config-path config_test_HR.yml 

pretrain_path='pretrained_models/dccf_idih_hrnet18s_v2p_LR_pretrain.pth'
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_upsample_refiner.py hrnet18s_v2p_idih256_upsample_hsl_refine_LR ${pretrain_path} \
    --resize-strategy Fixed256 \
    --res LR \
    --version hsl \
    --config-path config_test_LR.yml 
