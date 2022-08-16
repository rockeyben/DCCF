# train low-res
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -W ignore -m torch.distributed.launch --nproc_per_node=8 train.py \
    models/fixed256/hrnet18s_v2p_idih_upsample_hsl_refine_LR.py \
    --ngpu 8 \
    --workers 6 \
    --batch-size 32 \
    --exp-name=v1 \

low_res_weights='harmonization_exps/fixed256/hrnet18s_v2p_idih_upsample_hsl_refine_LR/000_v1/checkpoints/179.pth'
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 train.py \
    models/fixed256/hrnet18s_v2p_idih_upsample_hsl_refine_HR.py \
    --ngpu 2 \
    --workers 8 \
    --batch-size 8 \
    --exp-name=v1 \
    --weights=${low_res_weights}

pretrained_path='harmonization_exps/fixed256/hrnet18s_v2p_idih_upsample_hsl_refine_HR/000_v1/checkpoints/069.pth'
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_upsample_refiner.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${pretrain_path} \
    --resize-strategy Fixed256 \
    --version hsl \
    --config-path config_test_HR.yml 
