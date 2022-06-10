# HODOR: High-Level Object Descriptors for Object Re-segmentation in Video Learned from Static Images (CVPR 2022)

Ali Athar, Jonathon Luiten, Alexander Hermans, Deva Ramanan, Bastian Leibe

\[[`arXiv`](https://arxiv.org/abs/2112.09131)\] \[[`Related Workshop Paper`](https://arxiv.org/pdf/2206.00182.pdf)\]

This repository contains official code for the above-mentioned publication, as well as the related workshop paper titled 'Differentiable Soft-Maskted Attention' which was presented at the Transformers for Vision (T4V) Workshop at CVPR 2022.

## Conceptual Overview

![image](https://user-images.githubusercontent.com/14821941/173097679-184e4951-c0f6-4d7e-be1e-18441ff78d73.png)

## Installation

The following packages are required:

- PyTorch (v1.9.0)
- Torchvision (v0.10.0)
- Pillow
- opencv-python
- imgaug
- tqdm
- pyyaml
- tensorboardX
- pycocotools

#### Directory Setup

1. **Set the environment variable `HODOR_WORKSPACE_DIR`:** This is the top-level directory under which all checkpoints will be loaded/saved, and also where the datasets are expected to be located. The directory structure should look like this:

```
$HODOR_WORKSPACE_DIR
    - dataset_images
        - coco_2017_train
        - davis_trainval
            - seq1
            - seq2
            - ...
        - davis_testdev
            - seq1
            - seq2
            - ...
        - youtube_vos_2019_train
            - seq1
            - seq2
            - ...
        - youtube_vos_2019_val
            - seq1
            - seq2
            - ...
    - dataset_json_annotations
        - coco_2017_train.json
        - davis_train.json
        - davis_val.json
        - youtube_vos_2019_train.json
        - youtube_vos_2019_val.json
    - pretrained_backbones
        - swin_tiny_pretrained.pth
    - checkpoints
        - my_training_session
        - another_training_session
```

2. **Download annotations and pretrained models**: A basic setup of the above directory including all dataset annotations and pretrained models can be downloaded from [HERE]().
