# DCCF: Deep Comprehensible Color Filter Learning Framework

This is the official repository for our paper:

> **DCCF: Deep Comprehensible Color Filter Learning Framework for High-Resolution Image Harmonization**
> [https://arxiv.org/abs/2207.04788](https://arxiv.org/abs/2207.04788)
>
> Ben Xue, Shenghui Ran, Quan Chen, Rongfei Jia, Binqiang Zhao, Binqiang Zhao </br>
> (Accepted by ECCV 2022, Oral) 



## Enviroment

This code is built using Python 3.6 and relies on the PyTorch 1.7.1 & cuda 10.2. The following command installs all necessary packages:

```.bash
pip3 install -r requirements.txt --no-dependencies
```



## Datasets
We train and evaluate all our models on the [iHarmony4 dataset](https://github.com/bcmi/Image_Harmonization_Datasets). Before training low-resolution parts, we resize HAdobe5k subdataset so that   the longest side is smaller than 1024. If you want to run training or testing, please configure the paths to the dataset in [config.yml](./config.yml).
The resizing script is provided in [resize_dataset.py](./notebooks/resize_dataset.py). Note that we use the resized version of dataset for low-resolution phase training, and use the raw size version of dataset for high-resolution phase training. If you want to evaluate reuslts on both low and high resolution, the testing config is different for [config_test_LR.yml](./config_test_LR.yml) and [config_test_HR.yml](./config_test_HR.yml). 

To further compare the result with [CDTNet](https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization), you should also resize the HAdobe5k into (1024,1024) and (2048,2048) version, which can be done by [resize_dataset_1024_1024.py](./notebooks/resize_dataset_1024_1024.py) and [resize_dataset_2048_2048.py](./notebooks/resize_dataset_2048_2048.py).

## Training

For each experiment, a separate folder is created in the `./harmonization_exps` with Tensorboard logs, text logs, visualization and model's checkpoints.
You can specify another path in the [config.yml](./config.yml) (see `EXPS_PATH` variable).

Start training with the following commands:
```.bash
./runs/train_dccf_idih_hrnet18s_v2p.sh
./runs/train_dccf_idih.sh
./runs/train_dccf_issam.sh
```

The low-res training script only train at LR scale, the HR script will do joint training at LR & HR scale. Note that batch size may also affect final results.

To compare with [CDTNet](https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization), you should further finetune the dccf-issam model on HAdobe5k subset with this script:
```.bash
./runs/train_dccf_issam_finetune_adobe.sh
```

We used pre-trained HRNetV2 models from the [official repository](https://github.com/HRNet/HRNet-Image-Classification).
To train one of our models with HRNet backbone, download HRNet weights and specify their path in [config.yml](./config.yml) (see `IMAGENET_PRETRAINED_MODELS` variable).

The main configuration can be changed at [mconfigs/base.py](./mconfigs/base.py)
```
'npts': 8, # The nonlinearity of Lum Filter's curve
'up_size':(768, 1024),  # upsampled size during train
'use_attn' :True & 'use_refine' : True,   # use attentive rendering module
'use_hr' : True,      # open HR branch during train
'use_blur_L' : True   # blur on L map
```

## Evaluation

You can put pretrained model at [pretrained_models/](./pretrained_models), which can be directly used to re-implement results in the paper. It can be downloaded from [BaiduCloud (w8vg)](https://pan.baidu.com/s/1u5q94SzYciYkq5rspBaN7w) or [Google Drive](https://drive.google.com/drive/folders/1RFuf2zMU55ISYWSqu6HDELSxZTne8_u_?usp=sharing). 
```.bash
./runs/test_dccf_idih_hrnet18s_v2p_pretrain.sh
./runs/test_dccf_idih_pretrain.sh
./runs/test_dccf_issam_pretrain.sh
```

To get baseline results, you can download the baseline models from [BaiduCloud (w8vg)](https://pan.baidu.com/s/1u5q94SzYciYkq5rspBaN7w) or [Google Drive](https://drive.google.com/drive/folders/1RFuf2zMU55ISYWSqu6HDELSxZTne8_u_?usp=sharing) and put them at [pretrained_models/](./pretrained_models), then run:
```.bash
./runs/test_baselines_*.sh
```

To get BGU high-resolution results, please refer to its official MATLAB implementation [code](https://github.com/google/bgu). Our code only generates intermediate low-resolution predictions for this baseline.

## Interactive Experiments

Run [interactive_experiments.sh](./runs/interactive_experiments.sh) to perform filter fusion with our deep comprehensible color filter. You can put customized images with their input masks in a directory and run this script.

## Acknowledgement

Our code is based on Konstantin Sofiiuk's [iDIH](https://github.com/saic-vul/image_harmonization).

## Citation

```
@inproceedings{xue2022dccf,
 title={DCCF: Deep Comprehensible Color Filter Learning Framework for High-Resolution Image Harmonization},
 author={Ben Xue and Shenghui Ran and Quan Chen and Rongfei Jia and Binqiang Zhao and Binqiang Zhao},
 year={2022},
 booktitle={ECCV},
}
```
