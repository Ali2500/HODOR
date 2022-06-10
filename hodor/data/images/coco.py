from torch.utils.data import Dataset
from hodor.config import cfg
from hodor.data.images.image_dataset_base import ImageDatasetBase
from hodor.data.images.json_parser import parse_generic_image_dataset

import numpy as np


class CocoDataset(ImageDatasetBase, Dataset):
    def __init__(self, base_dir, ids_json_file, samples_to_create, max_num_instances, resize_mode,
                 min_num_instances: int = 1):

        ImageDatasetBase.__init__(self, "coco",
                                  min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
                                  min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
                                  resize_mode=resize_mode)
        Dataset.__init__(self)

        images, meta_info = parse_generic_image_dataset(base_dir, ids_json_file)

        self.category_labels = meta_info["category_labels"]

        # remove very small instances. Otherwise the mask becomes all-zeros when we down-sample the mask by 8x before
        # feeding it to the encoder.
        for ii in range(len(images)):
            areas = images[ii].mask_areas()
            boxes = images[ii].box_coordinates()

            keep_idxes = [
                j for j, (area, box) in enumerate(zip(areas, boxes))
                if area > self.min_ref_mask_area and min(box[2:]) > self.min_ref_mask_dim
            ]

            images[ii].segmentations = [images[ii].segmentations[j] for j in keep_idxes]
            images[ii].categories = [images[ii].categories[j] for j in keep_idxes]
            images[ii].bboxes = [images[ii].bboxes[j] for j in keep_idxes]
            images[ii].areas = [images[ii].areas[j] for j in keep_idxes]

        images = list(filter(lambda im: im.num_instances > 0, images))
        self.images = {im.id: im for im in images}

        # create training samples
        self.samples, self.sample_image_dims, self.sample_num_instances = self.create_training_samples(
            images, samples_to_create, min_num_instances, max_num_instances
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]

        image_sample = self.images[s['img_id']]
        image = image_sample.load_image()

        selected_instance_ids = s['instance_ids']
        sample_masks = image_sample.load_masks()

        selected_masks = np.zeros(image.shape[:2], np.int8)
        category_ids = []

        current_inst_id = 1
        for i, (mask_i, category_i) in enumerate(zip(sample_masks, image_sample.categories)):
            if i not in selected_instance_ids:
                continue

            selected_masks = np.where(mask_i, np.full_like(selected_masks, current_inst_id), selected_masks)
            current_inst_id += 1
            category_ids.append(category_i)

        parsed_sample = self.process_image_mask_pair(image, selected_masks)

        parsed_sample["meta"].update({
            "seq_id": s['img_id'],
            "num_instances": len(selected_instance_ids),
            "frame_idxes": list(range(self.num_augmented_frames + 1)),
            "category_ids": category_ids,
            "category_labels": [self.category_labels[cat_id] for cat_id in category_ids]
        })

        return parsed_sample
