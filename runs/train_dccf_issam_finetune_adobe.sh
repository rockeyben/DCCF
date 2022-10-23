# finetune on adobe's 1024*1024, 2048*2048 versions to compare with CDTNet
pretrain_weights="pretrained_models/dccf_issam_HR_pretrain_notv.pth"
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 train.py \
    models/fixed256/issam_dccf_HR_adobe1024.py \
    --ngpu 2 \
    --workers 8 \
    --batch-size 8 \
    --exp-name=v1 \
    --weights=${pretrain_weights}

CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 train.py \
    models/fixed256/issam_dccf_HR_adobe2048.py \
    --ngpu 2 \
    --workers 8 \
    --batch-size 8 \
    --exp-name=v1 \
    --weights=${pretrain_weights}

# test
weights1024="pretrained_models/dccf_issam_HR_pretrain_notv.pth"
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_upsample_refiner.py dccf_improved_ssam256_HR_clamp ${weights1024} \
    --resize-strategy Fixed256 \
    --version hsl_nobb \
    --config-path config_test_adobe1024.yml \
    --datasets HAdobe5k 

weights2048="harmonization_exps/fixed256/issam_dccf_HR_adobe2048/000_v1/checkpoints/049.pth"
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_upsample_refiner.py dccf_improved_ssam256_HR_clamp ${weights2048} \
    --resize-strategy Fixed256 \
    --version hsl_nobb \
    --config-path config_test_adobe2048.yml \
    --datasets HAdobe5k 
