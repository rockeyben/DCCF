weights='pretrained_models/dccf_idih_hrnet18s_v2p_HR_pretrain.pth'

CUDA_VISIBLE_DEVICES=0 python scripts/predict_interactive.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${weights} \
    --images $HOME/assets/ \
    --masks $HOME/assets/ \
    --channel L \
    --results-path $HOME/results/

CUDA_VISIBLE_DEVICES=0 python scripts/predict_interactive.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${weights} \
    --images  $HOME/assets/ \
    --masks  $HOME/assets/ \
    --channel S \
    --results-path $HOME/results/

#CUDA_VISIBLE_DEVICES=0 python scripts/predict_interactive.py hrnet18s_v2p_idih256_upsample_hsl_refine_HR ${weights} \
#    --images assets/interactive_images_H/ \
#    --masks assets/interactive_images_H/ \
#    --channel H \
#    --results-path harmonization_exps/interactive_results_H/

