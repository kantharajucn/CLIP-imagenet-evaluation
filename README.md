# CLIP inference for ImageNet and train other models using soft and hard labels from CLIP

This repo contains CLIP model inference for ImageNet dataset and using the resulting hard and soft labels to train any other models.

## Quick Start

Clone the repository and install the requirements using pip

```
git clone https://github.com/kantharajucn/clip-imagenet-evaluation

pip install -r requirements.txt

```


## Running inference

```
python3 test.py --data-dir /path/to/dataset/dir --num-workers 25 --batch-size 40

```

Inference results will be stored into files `soft_labels.json` and `hard_labels.json` in the current directory. 

## Running training

You can train any model from torch model zoo on both `Soft Labels` and `Hard Labels`

To train using soft labels

```
train.py /scratch_local/datasets/ImageNet2012 -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type soft_labels

```

To train using hard labels

```
train.py /scratch_local/datasets/ImageNet2012 -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 16  --label-type hard_labels

```