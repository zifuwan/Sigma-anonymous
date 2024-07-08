
<div align="center">
<h1> Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation </h1>

Anonymous WACV 2025 Submission



</div>


## 👀Introduction

This repository contains the code for our paper `Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation`. 

![](figs/sigma.png)

`Sigma`, as a lightweight and efficient method, reaches a balance between accuracy and speed. (Results below are calculated on [MFNet](https://github.com/haqishen/MFNet-pytorch) dataset)

![](figs/overall_flops.png)

## 💡Environment

We test our codebase with `PyTorch 1.13.1 + CUDA 11.7` as well as `PyTorch 2.2.1 + CUDA 12.1`. Please install corresponding PyTorch and CUDA versions according to your computational resources. We showcase the environment creating process with PyTorch 1.13.1 as follows.

1. Create environment.
    ```shell
    conda create -n sigma python=3.9
    conda activate sigma
    ```

2. Install all dependencies.
Install pytorch, cuda and cudnn, then install other dependencies via:
    ```shell
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
    ```shell
    pip install -r requirements.txt
    ```

3. Install Mamba
    ```shell
    cd models/encoders/selective_scan && pip install . && cd ../../..
    ```

## ⏳Setup

### Datasets

1. We use four datasets, including both RGB-Thermal and RGB-Depth datasets:
    - [RGB-Thermal MFNet](https://github.com/haqishen/MFNet-pytorch)
    - [RGB-Thermal PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)
    - [RGB-Depth NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
    - [RGB-Depth SUN-RGBD](https://rgbd.cs.princeton.edu/)

    Please refer to the original dataset websites for more details. You can directly download the processed RGB-Depth datasets from [DFormer](https://github.com/VCIP-RGBD/DFormer?tab=readme-ov-file), though you may need to make small modifications to the txt files.

<!-- 2. <u>We also provide the processed datasets (including RGB-Thermal and RGB-Depth) we use here: [Google Drive Link](https://drive.google.com/drive/folders/1GD4LYF208h9-mHJ_lxW11UM0TPlRmv0z?usp=drive_link).</u> -->

2. If you are using your own datasets, please orgnize the dataset folder in the following structure:
    ```shell
    <datasets>
    |-- <DatasetName1>
        |-- <RGBFolder>
            |-- <name1>.<ImageFormat>
            |-- <name2>.<ImageFormat>
            ...
        |-- <ModalXFolder>
            |-- <name1>.<ModalXFormat>
            |-- <name2>.<ModalXFormat>
            ...
        |-- <LabelFolder>
            |-- <name1>.<LabelFormat>
            |-- <name2>.<LabelFormat>
            ...
        |-- train.txt
        |-- test.txt
    |-- <DatasetName2>
    |-- ...
    ```

    `train.txt/test.txt` contains the names of items in training/testing set, e.g.:

    ```shell
    <name1>
    <name2>
    ...
    ```


## 📦Usage

### Training
1. Please download the pretrained [VMamba](https://github.com/MzeroMiko/VMamba) weights:

    - [VMamba_Tiny](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth).
    - [VMamba_Small](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth).
    - [VMamba_Base](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp06_ckpt_epoch_241.pth).

    <u> Please put them under `pretrained/vmamba/`. </u>


2. Config setting.

    Edit config file in the `configs` folder.    
    Change C.backbone to `sigma_tiny` / `sigma_small` / `sigma_base` to use the three versions of Sigma. 

3. Run multi-GPU distributed training:

    ```shell
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4  --master_port 29502 train.py -p 29502 -d 0,1,2,3 -n "dataset_name"
    ```

    Here, `dataset_name=mfnet/pst/nyu/sun`, referring to the four datasets.

4. You can also use single-GPU training:

    ```shell
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun -m --nproc_per_node=1 train.py -p 29501 -d 0 -n "dataset_name" 
    ```
5. Results will be saved in `log_final` folder.


### Evaluation
1. Run the evaluation by:

    ```shell
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
    ```

    Here, `dataset_name=mfnet/pst/nyu/sun`, referring to the four datasets.\
    `epoch_number` refers to a number standing for the epoch number you want to evaluate with. You can also use a `.pth` checkpoint path directly for `epoch_number` to test for a specific weight.

2. If you want to use multi GPUs please specify multiple Device IDs:

    ```shell
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0,1,2,3,4,5,6,7" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
    ```

3. Results will be saved in `log_final` folder.

## 📈Results

We provide our trained weights on the four datasets:

### MFNet (9 categories)
| Architecture | Backbone | mIOU |
|:---:|:---:|:---:|
| Sigma | VMamba-T | 60.2% |
| Sigma | VMamba-S | 61.1% |
| Sigma | VMamba-B | 61.3% |

### PST900 (5 categories)
| Architecture | Backbone | mIOU |
|:---:|:---:|:---:|
| Sigma | VMamba-T | 88.6% |
| Sigma | VMamba-S | 87.8% |

### NYU Depth V2 (40 categories)
| Architecture | Backbone | mIOU |
|:---:|:---:|:---:|
| Sigma | VMamba-T | 53.9% |
| Sigma | VMamba-S | 57.0% | 

### SUN RGB-D (37 categories)
| Architecture | Backbone | mIOU |
|:---:|:---:|:---:|
| Sigma | VMamba-T | 50.0% | 
| Sigma | VMamba-S | 52.4% | 

