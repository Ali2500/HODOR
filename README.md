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

Note that we convert all annotations for COCO, DAVIS and YouTube-VOS into a somewhat standardized JSON format so that data loading code can be easily re-used.

2. **Download annotations and pretrained models**: A basic setup of the above directory including dataset annotations (but not the image frames) and pretrained models can be downloaded from [HERE](https://omnomnom.vision.rwth-aachen.de/data/HODOR/everything_zipped.zip). You can also download separate files from by browsing through them from [here](https://omnomnom.vision.rwth-aachen.de/data/HODOR). 


## Inference

**DAVIS 2017 val**: Run the following from the repository base directory:

```
python hodor/inference/main.py $HODOR_WORKSPACE_DIR/checkpoints/static_image/250000.pth --dataset davis_val --output_dir davis_inference_output --temporal_window 7 --min_image_dim 512
```

This will create a directory called `davis_inference_output` in `$HODOR_WORKSPACE_DIR/checkpoints/static_image` and write the output masks to it. For Likewise you can point the script to the checkpoints in `video_dense` or `video_sparse` to evaluate those. 

**YouTube-VOS or DAVIS testdev**: To run inference on a different dataset, set the `--dataset` argument to `davis_testdev` or `youtube_vos_val`. For detailed inference options, run the script with `--help'. Note that you may need to adjust the `--min_image_dim` and `--temporal_window` options to get the exact results mentioned in the paper for different datasets.

## Training

