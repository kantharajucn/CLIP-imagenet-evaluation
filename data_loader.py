import os
import json
import torchvision.datasets as datasets
import torch

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

    def __init__(self, *args, **kwargs):

        self.clip_class_mapping = self._load_clip_mappings()
        super(ImageNetClipDataset, self).__init__(*args, **kwargs)

    def _load_clip_mappings(self):
        mappings = {}
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cur_dir, "clip_hard_labels.json")) as f:
            for line in f:
                mappings.update(json.loads(line[:-2]))
        return mappings

    def _get_new_template(self, image_path):
            file_name = os.path.basename(image_path)
            target_class = self.clip_class_mapping[file_name]
            target_index = self.class_to_idx[target_class]
            return target_index


    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageNetClipDataset, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        new_target = self._get_new_template(path)
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




