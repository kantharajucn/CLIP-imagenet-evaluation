# CLIP inference for ImageNet and train other models using soft and hard labels from CLIP

This repo contains CLIP model inference for ImageNet dataset and using the resulting hard and soft labels to train any other models.

## Quick Start

Clone the repository and install the requirements using pip

```
git clone https://github.com/kantharajucn/clip-imagenet-evaluation

pip install -r requirements.txt

```


## Running CLIP inference on ImageNet

```
python3 clip_imagenet_inference.py --data-dir /path/to/dataset/dir --num-workers 25 --batch-size 40

```

Inference results will be stored into files `soft_labels.json` and `hard_labels.json` in the current directory. 

## Running training

You can train any model from torch model zoo on both `Soft Labels` and `Hard Labels`

To train using soft labels

```
train.py /path/to/imagenet -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type soft_labels

```

To train using hard labels

```
train.py /path/to/imagenet -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 16  --label-type hard_labels

```


## Evaluate model trained using CLIP hard and Soft Labels

After training any `pytorch zoo model` using CLIP labels, you can evaluate the model on ImageNet with original ImageNet labels or CLIP labels using the below script.

### Evaluate on ImageNet labels

```
test.py /path/to/imagenet -a resnet50 --evaluate --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --checkpoint ./hard_labels_checkpoint.pth.tar
```


### Evaluate on CLIP hard labels
```
test.py /path/to/imagenet -a resnet50 --evaluate --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type hard_labels --labels ./IN_val_clip_hard_labels.json --checkpoint ./hard_labels_checkpoint.pth.tar
```

### Evaluate on CLIP soft labels
```
test.py /path/to/imagenet -a resnet50 --evaluate --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type soft_labels --labels ./IN_val_clip_soft_labels.json --checkpoint ./soft_labels_checkpoint.pth.tar
```

