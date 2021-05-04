import json
import os
import numpy as np
import torch
from torchvision import datasets


class ImageNetCategory():
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __init__(self):
        pass

    def __call__(self, full_path):
        img_name = full_path.split("/")[-1]
        category = full_path.split("/")[-2]
        return category


class ImageNetDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, *args, **kwargs):
        super(ImageNetDataset, self).__init__(*args, **kwargs)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        sample, target = super(ImageNetDataset, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        new_target = ImageNetCategory()(path)
        original_tuple = (sample, new_target)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ImageNetClipDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """
    SOFT_LABELS = "soft_labels"
    HARD_LABELS = "hard_labels"

    def __init__(self, label_type, mappings, *args, **kwargs):
        self.label_type = label_type
        if len(mappings) > 1:
            self.clip_class_mapping = mappings
        else:
            self.clip_class_mapping = self._load_clip_mappings(label_type)
        super(ImageNetClipDataset, self).__init__(*args, **kwargs)

    def _load_clip_mappings(self):
        mappings = {}
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        if self.label_type == ImageNetClipDataset.HARD_LABELS:
            file_name = os.path.join(cur_dir, "clip_hard_labels.json")
            with open(file_name) as f:
                for line in f:
                    mappings.update(json.loads(line[:-2]))
        elif self.label_type == ImageNetClipDataset.SOFT_LABELS:
            file_name = os.path.join(cur_dir, "clip_soft_labels.json")
            with open(file_name) as f:
                for line in f:
                    temp_tuple = list(json.loads(line[:-2]).items())[0]
                    mappings[temp_tuple[0]] = np.array(temp_tuple[1], dtype=np.float32)
        else:
            raise ValueError("Invalid label type")
        return mappings

    def _get_new_templatei_hard_labels(self, image_path):
        file_name = os.path.basename(image_path)
        target_class = self.clip_class_mapping[file_name]
        target_index = self.class_to_idx[target_class]
        return target_index

    def _get_new_template_soft_labels(self, image_path):
        file_name = os.path.basename(image_path)
        target_class = self.clip_class_mapping[file_name]
        return target_class
    
    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageNetClipDataset, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        if self.label_type == ImageNetClipDataset.HARD_LABELS:
            new_target = self._get_new_template_hard_labels(path)
        elif self.label_type == ImageNetClipDataset.SOFT_LABELS:
    	    new_target = self._get_new_template_soft_labels(path)
        original_tuple = (sample, new_target,)
        return original_tuple


def data_loader(transform, args):
    imagenet_data = ImageNetDataset(args.data_dir, transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    return data_loader
