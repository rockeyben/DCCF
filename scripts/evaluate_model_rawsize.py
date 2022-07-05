import argparse
import sys

sys.path.insert(0, '.')

import torch
import os
from pathlib import Path
from albumentations import Resize, NoOp
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from iharm.inference.predictor import Predictor
from iharm.inference.evaluation import evaluate_dataset_rawsize
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, N, AvgPredictTime
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.exp import load_config_file
from iharm.utils.log import logger, add_new_file_output_to_logger


RESIZE_STRATEGIES = {
    'None': NoOp(),
    'LimitLongest1024': LongestMaxSizeIfLarger(1024),
    'Fixed256': Resize(256, 256),
    'Fixed512': Resize(512, 512),
    'Fixed1024' : Resize(1024, 1024),
    'Fixed2048' : Resize(2048, 2048),
    'Fixed3072' : Resize(3072, 3072)
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--datasets', type=str, default='HDay2Night,HFlickr,HCOCO,HAdobe5k',
                        help='Each dataset name must be one of the prefixes in config paths, '
                             'which look like DATASET_PATH.')
    parser.add_argument('--resize-strategy', type=str, choices=RESIZE_STRATEGIES.keys(), default='Fixed256')
    parser.add_argument('--use-flip', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--vis-dir', type=str, default=None, help='output dir of visualization')
    parser.add_argument('--upsample-method', type=str, default=None, help='GF, bilinear, rawsize')

    parser.add_argument('--eval-prefix', type=str, default='')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    return args, cfg


def main():
    args, cfg = parse_args()
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    add_new_file_output_to_logger(
        logs_path=Path(cfg.EXPS_PATH) / 'evaluation_logs',
        prefix=f'{Path(checkpoint_path).stem}_',
        only_message=True
    )
    logger.info(vars(args))

    device = torch.device(f'cuda:{args.gpu}')
    net = load_model(args.model_type, checkpoint_path, verbose=True)

    predictor = Predictor(net, device, with_flip=args.use_flip)

    datasets_names = args.datasets.split(',')
    datasets_metrics_low = []
    datasets_metrics_full = []

    if args.vis_dir:
        try:
            os.makedirs(args.vis_dir)
        except:
            pass

    for dataset_indx, dataset_name in enumerate(datasets_names):
        dataset = HDataset(
            cfg.get(f'{dataset_name.upper()}_PATH'), split='test',
            augmentator=HCompose([RESIZE_STRATEGIES[args.resize_strategy]]),
            keep_background_prob=-1
        )

        dataset_metrics_lowres = MetricsHub([N(), MSE(), PSNR(), AvgPredictTime()],
                                     name=dataset_name)
        dataset_metrics_fullres = MetricsHub([N(), MSE(), PSNR(), AvgPredictTime()],
                                     name=dataset_name)
        evaluate_dataset_rawsize(dataset, predictor, dataset_metrics_lowres, dataset_metrics_fullres,
            upsample_method=args.upsample_method, visdir=args.vis_dir)
        datasets_metrics_low.append(dataset_metrics_lowres)
        datasets_metrics_full.append(dataset_metrics_fullres)


        if args.upsample_method != 'BGU':
            if dataset_indx == 0:
                logger.info(dataset_metrics_lowres.get_table_header())
        
            # logger.info(dataset_metrics_lowres)
            logger.info(dataset_metrics_fullres)

    if len(datasets_metrics_low) > 1:

        if args.upsample_method != 'BGU':
            # overall_metrics = sum(datasets_metrics_low, MetricsHub([], 'Overall low'))
            # logger.info('-' * len(str(overall_metrics)))
            # logger.info(overall_metrics)

            overall_metrics = sum(datasets_metrics_full, MetricsHub([], 'Overall full'))
            logger.info('-' * len(str(overall_metrics)))
            logger.info(overall_metrics)
if __name__ == '__main__':
    main()
