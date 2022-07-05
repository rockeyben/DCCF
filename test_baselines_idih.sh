weights='pretrained_models/idih.pth'

# Using config_test_HR.yml would get worse results for BGU, so we use config_test_LR.yml as a stronger baseline
# Run BGU algorithm [https://github.com/google/bgu] with the saved intermediate low-res results
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py improved_dih256 ${weights} \
    --resize-strategy Fixed256 \
    --upsample-method BGU \
    --config-path config_test_LR.yml \
    --vis-dir harmonization_exps/vis_idih256_lowres_offical_LRconfig

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py improved_dih256 ${weights} \
    --resize-strategy Fixed256 \
    --upsample-method bilinear \
    --config-path config_test_HR.yml 

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py improved_dih256 ${weights} \
    --resize-strategy Fixed256 \
    --upsample-method GF \
    --config-path config_test_HR.yml 

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_model_rawsize.py improved_dih256 ${weights} \
    --resize-strategy None \
    --upsample-method rawsize \
    --config-path config_test_HR.yml \
    --datasets HFlickr,HDay2Night,HCOCO 