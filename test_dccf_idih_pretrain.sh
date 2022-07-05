weights="pretrained_models/dccf_idih_HR_pretrain.pth"
CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_upsample_refiner.py dccf_improved_dih256_HR_clamp ${weights} \
    --resize-strategy Fixed256 \
    --version hsl_nobb \
    --config-path config_test_HR.yml 