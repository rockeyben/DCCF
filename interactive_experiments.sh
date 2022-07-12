weights='pretrained_models/dccf_idih_hrnet18s_v2p_HR_pretrain.pth'

CUDA_VISIBLE_DEVICES=0 python scripts/predict_interactive.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${weights} \
    --images assets/interactive_images/ \
    --masks assets/interactive_images/ \
    --channel L \
    --results-path harmonization_exps/interactive_results_L/ 

CUDA_VISIBLE_DEVICES=0 python scripts/predict_interactive.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${weights} \
    --images assets/interactive_images/ \
    --masks assets/interactive_images/ \
    --channel S \
    --results-path harmonization_exps/interactive_results_S/ 

CUDA_VISIBLE_DEVICES=0 python scripts/predict_interactive.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${weights} \
    --images assets/interactive_images_H/ \
    --masks assets/interactive_images_H/ \
    --channel H \
    --results-path harmonization_exps/interactive_results_H/ 

