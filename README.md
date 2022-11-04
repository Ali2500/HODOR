# HODOR: High-Level Object Descriptors for Object Re-segmentation in Video Learned from Static Images (CVPR 2022 Oral)

Ali Athar, Jonathon Luiten, Alexander Hermans, Deva Ramanan, Bastian Leibe

\[[`arXiv`](https://arxiv.org/abs/2112.09131)\] \[[`BibTeX`](https://github.com/Ali2500/HODOR/blob/main/README.md#cite)\] \[[`Related Workshop Paper`](https://arxiv.org/pdf/2206.00182.pdf)\]

This repository contains official code for the above-mentioned publication, as well as the related workshop paper titled 'Differentiable Soft-Maskted Attention' which was presented at the Transformers for Vision (T4V) Workshop at CVPR 2022.

## Conceptual Overview

- **Idea:** Can we learn to do VOS by training only on static images?
- Unlike existing VOS methods which learn pixel-to-pixel correspondences, HODOR learns to encode object appearance information from an image frame into a concise object descriptors which can then be decoded into another video frame to "re-segment" that object.
- We can also train using cyclic consistency on video clips where just one frame is annotated.

![image](https://user-images.githubusercontent.com/14821941/173097679-184e4951-c0f6-4d7e-be1e-18441ff78d73.png)

## Differentiable Soft-Masked Attention

In case you followed the workshop paper and want to nose-dive into the implementation of our novel differentiable soft-masked attention, take a look at the PyTorch module here: [hodor/modeling/encoder/soft_masked_attention.py](https://github.com/Ali2500/HODOR/blob/443903b07fbed6dac57668c2c63a58417f82003a/hodor/modelling/encoder/soft_masked_attention.py). The API is similar to PyTorch's native `nn.MultiHeadAttention` with the main difference that the `forward` requires a soft mask for the attention to be given as input.

## Installation

The following packages are required:

- Python v3.7.10
- PyTorch (v1.9.0)
- Torchvision (v0.10.0)
- Pillow
- opencv-python
- imgaug
- einops
- timm
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

2. **Download annotations and pretrained models**: Links to downloadable resources are given below. For the easiest setup, download the entire zipped workspace. This includes all model checkpoints (COCO training + finetuning on sparse and dense video) as well as train/val/test annotations in JSON format for all 3 datasets (COCO, DAVIS, YouTube-VOS). Note that you'll still have to copy the dataset images to the relevant dataset in `$HODOR_WORKSPACE_DIR/dataset_images`.

| Content                                                    | URLs                                                                                |
|------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Zipped Workspace (Model Checkpoints + Dataset Annotations) | [LINK](https://omnomnom.vision.rwth-aachen.de/data/HODOR/everything_zipped.zip)     |
| Dataset Annotations                                        | [LINK](https://omnomnom.vision.rwth-aachen.de/data/HODOR/dataset_json_annotations/) |
| Model Checkpoints                                          | [LINK](https://omnomnom.vision.rwth-aachen.de/data/HODOR/checkpoints/)              |


## Inference

**DAVIS 2017 val**: Run the following from the repository base directory:

```
python hodor/inference/main.py $HODOR_WORKSPACE_DIR/checkpoints/static_image/250000.pth --dataset davis_val --output_dir davis_inference_output --temporal_window 7 --min_image_dim 512
```

This will create a directory called `davis_inference_output` in `$HODOR_WORKSPACE_DIR/checkpoints/static_image` and write the output masks to it. For Likewise you can point the script to the checkpoints in `video_dense` or `video_sparse` to evaluate those. 

**YouTube-VOS or DAVIS testdev**: To run inference on a different dataset, set the `--dataset` argument to `davis_testdev` or `youtube_vos_val`. For detailed inference options, run the script with `--help`. Note that you may need to adjust the `--min_image_dim` and/or `--temporal_window` options to get the exact results mentioned in the paper for different datasets.

## Training

#### Static Images

For single GPU training on static images from COCO:

```
python hodor/training/main.py --model_dir my_training_on_static_images --cfg static_image.yaml
```

For multi-GPU training (e.g. 8 GPUs) on static images from COCO:

```
python -m torch.distributed.launch --nproc_per_node=8 hodor/training/main.py --model_dir my_training_on_static_images --cfg static_image.yaml --allow_multigpu
```

The checkpoints provided above were usually trained on 4 or 8 GPUs. Note that we use gradient accumulation so it is possible to train with the default batch size of 8 even on a single GPU, but the results will not be exactly reproducible.

#### Video

To fine-tuning the COCO trained model on sparse video (i.e. assuming that only one frame per video is annotated in the DAVIS and YouTube-VOS training sets):

```
python -m torch.distributed.launch --nproc_per_node=8 hodor/training/main.py --model_dir my_finetuning_on_video --cfg video_sparse.yaml --allow_multigpu --restore_path /path/to/coco/trained/checkpoint.pth
```

Likewise you can set `--cfg video_dense.yaml` to train with the full set of available training annotations.

## Cite

```
@inproceedings{athar2022hodor,
  title={HODOR: High-level Object Descriptors for Object Re-segmentation in Video Learned from Static Images},
  author={Athar, Ali and Luiten, Jonathon and Hermans, Alexander and Ramanan, Deva and Leibe, Bastian},
  booktitle={CVPR},
  year={2022}
}
```
