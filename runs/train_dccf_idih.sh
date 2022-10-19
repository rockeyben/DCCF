# train low-res
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    models/fixed256/idih_dccf_LR_hsv1_clamp.py \
    --ngpu 1 \
    --workers 8 \
    --batch-size 16 \
    --exp-name=v1

# resume the above low-res weights to perform high-res training
low_res_weights='harmonization_exps/fixed256/idih_dccf_LR_hsv1_clamp/000_v1/checkpoints/119.pth'
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 train.py \
    models/fixed256/idih_dccf_HR_hsv1_clamp.py \
    --ngpu 2 \
    --workers 8 \
    --batch-size 8 \
    --exp-name=v1 \
    --weights=${low_res_weights}

# test
weights="harmonization_exps/fixed256/idih_dccf_HR_hsv1_clamp/000_v1/checkpoints/069.pth"
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_upsample_refiner.py dccf_improved_dih256_HR_clamp ${weights} \
    --resize-strategy Fixed256 \
    --version hsl_nobb \
    --config-path config_test_HR.yml 
